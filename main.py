import random
import time
import cv2
import numpy as np
import carla    #for simmulation
from ultralytics import YOLO    #for object detection
import torch    #check whether GPU is available

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.backends.cudnn.benchmark = True

# ==================================
# SETTINGS required in car behavior
# ==================================
WEATHERS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.SoftRainSunset,
]

# Lane steering

STEER_MAX = 0.60
LOOKAHEAD_Y_RATIO = 0.68
LANE_WIDTH_EST_RATIO = 0.42
STEER_GAIN = 0.78
STEER_DEADBAND = 0.018
STEER_RATE_LIMIT_BOTH = 0.030
STEER_RATE_LIMIT_ONE = 0.045
STEER_RATE_LIMIT_MEMORY = 0.020

# Road behavior

CURVE_SLOW_T1 = 0.03
CURVE_SLOW_T2 = 0.07
LANE_LOSS_GRACE = 12

# One-lane behavior

ONE_LANE_SMOOTH = 0.25
BOTH_LANE_SMOOTH = 0.60
ONE_LANE_STEER_BOOST = 1.15

# Free-space avoidance

FREE_SPACE_CENTER_MIN = 0.58
FREE_SPACE_SIDE_MARGIN = 0.04
AVOID_STEER = 0.35

# YOLO obstacle detection

DANGER_CLASSES = {
    "car",
    "truck",
    "bus",
    "person",
    "motorcycle",
    "bicycle",
}
CONF_THRES = 0.35
CENTER_BAND_PX = 180
YOLO_EVERY_N = 3

STOP_AREA = {
    "person": 30000,
    "bicycle": 30000,
    "motorcycle": 45000,
    "car": 120000,
    "truck": 150000,
    "bus": 150000,
}

SLOW_AREA = {
    "person": 12000,
    "bicycle": 12000,
    "motorcycle": 20000,
    "car": 50000,
    "truck": 70000,
    "bus": 70000,
}

# Front obstacle sensor
OBSTACLE_WARN_DISTANCE = 12.0
OBSTACLE_BRAKE_DISTANCE = 6.0

# Performance

YOLO_IMG_SIZE = 512
CAMERA_W = 512
CAMERA_H = 384
SENSOR_TICK = 0.12
NPC_COUNT = 0  # keep 0 while testing

# =========================
# YOLO MODEL
# =========================
model = YOLO("yolov8n.pt")
if USE_CUDA:
    model.to("cuda")

# =========================
# GLOBALS
# =========================
prev_steer = 0.0    #for storing last steer angle
npc_vehicles = []
vehicle = None
camera = None
collision_sensor = None
obstacle_sensor = None

frame_idx = 0   #counts camera frames
prev_time = time.time()
last_obstacle_speed = 60
last_obstacle_frame = None

prev_left_model = None
prev_right_model = None
lane_miss_count = 0     #tracks how many frames the lane was lost

collision_hold_until = 0.0      #force an emergency brake for short time after a crash
collision_reason = ""

obstacle_distance = None
obstacle_actor = "unknown"
last_obstacle_time = 0.0


# =========================
# HELPERS
# =========================
def clamp(val, lo, hi):     #clamp keeps a number inside a safe range
    return max(lo, min(hi, val))


def adjust_gamma(image, gamma=1.30):        #brightens or darkens an image so lane line are easier to see
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)],
        dtype=np.uint8
    )
    return cv2.LUT(image, table)


def spawn_random_vehicle(world, blueprints):
    vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute(
            "color",
            random.choice(vehicle_bp.get_attribute("color").recommended_values)
        )

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        v = world.try_spawn_actor(vehicle_bp, spawn_point)
        if v is not None:
            return v

    return None


def spawn_npc_vehicles(world, blueprints, client, count=0):
    spawned = []
    if count <= 0:
        return spawned

    vehicle_bps = blueprints.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    traffic_manager = client.get_trafficmanager(8000)

    for spawn_point in spawn_points[1:]:
        bp = random.choice(vehicle_bps)

        if bp.has_attribute("color"):
            bp.set_attribute(
                "color",
                random.choice(bp.get_attribute("color").recommended_values)
            )

        npc = world.try_spawn_actor(bp, spawn_point)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            spawned.append(npc)

        if len(spawned) >= count:
            break

    return spawned


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.90), height),
        (int(width * 0.62), int(height * 0.55)),
        (int(width * 0.38), int(height * 0.55)),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)

#below two function fit lane model and x at y help to draw lane lines from detected road markings 

def fit_lane_model(points):
    if len(points) < 4:
        return None

    x_vals = np.array([p[0] for p in points], dtype=np.float32)
    y_vals = np.array([p[1] for p in points], dtype=np.float32)

    a, b = np.polyfit(y_vals, x_vals, 1)
    return a, b


def x_at_y(model_line, y):
    return int(model_line[0] * y + model_line[1])


def build_lane_features(frame): # prepares image for lane detection by improving brightness and extracting lane like edges and colors

    bright = adjust_gamma(frame, 1.30)

    hls = cv2.cvtColor(bright, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    white_mask = cv2.inRange(
        hls,
        np.array([0, 160, 0], dtype=np.uint8),
        np.array([180, 255, 90], dtype=np.uint8)
    )

    yellow_mask = cv2.inRange(
        hls,
        np.array([10, 70, 70], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8)
    )

    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    edges = cv2.Canny(gray_eq, 50, 150)
    combined = cv2.bitwise_or(edges, color_mask)

    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    return combined


def lane_follow(frame):
    global prev_steer, prev_left_model, prev_right_model, lane_miss_count

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    frame_center = width // 2
    y_bottom = height - 1
    y_lookahead = int(height * LOOKAHEAD_Y_RATIO)
    approx_lane_width = int(width * LANE_WIDTH_EST_RATIO)

    features = build_lane_features(frame)
    cropped = region_of_interest(features)

    lines = cv2.HoughLinesP(
        cropped,
        1,
        np.pi / 180,
        35,
        minLineLength=25,
        maxLineGap=80
    )

    left_points = []
    right_points = []

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.35:
                continue

            if slope < 0 and x1 < width * 0.55 and x2 < width * 0.60:
                left_points.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and x1 > width * 0.40 and x2 > width * 0.35:
                right_points.extend([(x1, y1), (x2, y2)])

    current_left_model = fit_lane_model(left_points)
    current_right_model = fit_lane_model(right_points)

    if current_left_model is not None:
        prev_left_model = current_left_model
    if current_right_model is not None:
        prev_right_model = current_right_model

    use_memory = lane_miss_count < LANE_LOSS_GRACE

    left_model = current_left_model if current_left_model is not None else (prev_left_model if use_memory else None)
    right_model = current_right_model if current_right_model is not None else (prev_right_model if use_memory else None)

    if current_left_model is not None and current_right_model is not None:
        lane_mode = "both"
    elif current_left_model is not None:
        lane_mode = "left_only"
    elif current_right_model is not None:
        lane_mode = "right_only"
    elif use_memory and (prev_left_model is not None or prev_right_model is not None):
        lane_mode = "memory"
    else:
        lane_mode = "lost"

    lane_center_bottom = frame_center
    lane_center_lookahead = frame_center

    if left_model is not None and right_model is not None:
        left_bottom = clamp(x_at_y(left_model, y_bottom), 0, width - 1)
        left_look = clamp(x_at_y(left_model, y_lookahead), 0, width - 1)
        right_bottom = clamp(x_at_y(right_model, y_bottom), 0, width - 1)
        right_look = clamp(x_at_y(right_model, y_lookahead), 0, width - 1)

        cv2.line(annotated, (left_bottom, y_bottom), (left_look, y_lookahead), (0, 255, 255), 4)
        cv2.line(annotated, (right_bottom, y_bottom), (right_look, y_lookahead), (0, 255, 255), 4)

        lane_center_bottom = (left_bottom + right_bottom) // 2
        lane_center_lookahead = (left_look + right_look) // 2

    elif left_model is not None:
        left_bottom = clamp(x_at_y(left_model, y_bottom), 0, width - 1)
        left_look = clamp(x_at_y(left_model, y_lookahead), 0, width - 1)
        cv2.line(annotated, (left_bottom, y_bottom), (left_look, y_lookahead), (0, 255, 255), 4)
        lane_center_bottom = left_bottom + int(approx_lane_width * 0.5)
        lane_center_lookahead = left_look + int(approx_lane_width * 0.5)

    elif right_model is not None:
        right_bottom = clamp(x_at_y(right_model, y_bottom), 0, width - 1)
        right_look = clamp(x_at_y(right_model, y_lookahead), 0, width - 1)
        cv2.line(annotated, (right_bottom, y_bottom), (right_look, y_lookahead), (0, 255, 255), 4)
        lane_center_bottom = right_bottom - int(approx_lane_width * 0.5)
        lane_center_lookahead = right_look - int(approx_lane_width * 0.5)

    lane_center_bottom = clamp(lane_center_bottom, 0, width - 1)
    lane_center_lookahead = clamp(lane_center_lookahead, 0, width - 1)

    error_now = (lane_center_lookahead - frame_center) / float(frame_center)
    error_bottom = (lane_center_bottom - frame_center) / float(frame_center)

    if abs(error_now) < STEER_DEADBAND and abs(error_bottom) < STEER_DEADBAND and lane_mode == "both":
        steer_raw = 0.0
    else:
        steer_raw = clamp(
            STEER_GAIN * (0.70 * error_now + 0.30 * error_bottom),
            -STEER_MAX,
            STEER_MAX
        )

    curve_strength = abs(lane_center_lookahead - lane_center_bottom) / float(width)

    if lane_mode == "both":
        steer_target = BOTH_LANE_SMOOTH * prev_steer + (1.0 - BOTH_LANE_SMOOTH) * steer_raw
        if curve_strength < CURVE_SLOW_T1:
            curve_speed_cap = 60
            curve_label = "STRAIGHT"
        elif curve_strength < CURVE_SLOW_T2:
            curve_speed_cap = 40
            curve_label = "GENTLE CURVE"
        else:
            curve_speed_cap = 25
            curve_label = "SHARP CURVE"

    elif lane_mode in ("left_only", "right_only"):
        steer_target = ONE_LANE_SMOOTH * prev_steer + (1.0 - ONE_LANE_SMOOTH) * steer_raw
        steer_target = clamp(steer_target * ONE_LANE_STEER_BOOST, -STEER_MAX, STEER_MAX)

        if curve_strength < CURVE_SLOW_T1:
            curve_speed_cap = 18
            curve_label = "ONE LANE"
        elif curve_strength < CURVE_SLOW_T2:
            curve_speed_cap = 12
            curve_label = "ONE LANE CURVE"
        else:
            curve_speed_cap = 10
            curve_label = "HARD TURN"

    elif lane_mode == "memory":
        steer_target = prev_steer * 0.92
        curve_speed_cap = 15
        curve_label = "LANE MEMORY"

    else:
        steer_target = 0.0
        curve_speed_cap = 0
        curve_label = "LANE LOST"

    if lane_mode == "both":
        delta_limit = STEER_RATE_LIMIT_BOTH
    elif lane_mode in ("left_only", "right_only"):
        delta_limit = STEER_RATE_LIMIT_ONE
    elif lane_mode == "memory":
        delta_limit = STEER_RATE_LIMIT_MEMORY
    else:
        delta_limit = STEER_RATE_LIMIT_MEMORY

    steer = prev_steer + clamp(steer_target - prev_steer, -delta_limit, delta_limit)
    steer = clamp(steer, -STEER_MAX, STEER_MAX)
    prev_steer = float(steer)

    if lane_mode == "lost":
        lane_miss_count += 1
    else:
        lane_miss_count = 0

    cv2.line(annotated, (frame_center, 0), (frame_center, height), (255, 255, 0), 1)
    cv2.line(annotated, (lane_center_lookahead, 0), (lane_center_lookahead, height), (255, 180, 0), 2)

    cv2.putText(annotated, f"Steer: {prev_steer:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(annotated, f"Road: {curve_label}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return prev_steer, curve_speed_cap, lane_mode, annotated


def obstacle_detection(raw_frame, draw_frame=None):
    annotated = raw_frame.copy() if draw_frame is None else draw_frame.copy()

    height, width = raw_frame.shape[:2]
    frame_center = width // 2

    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    results = model.predict(
        rgb_frame,
        conf=CONF_THRES,
        imgsz=YOLO_IMG_SIZE,
        device=0 if USE_CUDA else "cpu",
        half=USE_CUDA,
        verbose=False
    )

    best_box = None
    best_name = None
    best_conf = 0.0
    best_area = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            conf = float(box.conf[0])

            if class_name not in DANGER_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2

            if abs(center_x - frame_center) > CENTER_BAND_PX:
                continue

            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
                best_name = class_name
                best_conf = conf

    speed = 60

    cv2.line(annotated, (frame_center - CENTER_BAND_PX, 0),
             (frame_center - CENTER_BAND_PX, height), (255, 0, 0), 2)
    cv2.line(annotated, (frame_center + CENTER_BAND_PX, 0),
             (frame_center + CENTER_BAND_PX, height), (255, 0, 0), 2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{best_name} {best_conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        stop_thr = STOP_AREA.get(best_name, 120000)
        slow_thr = SLOW_AREA.get(best_name, 50000)

        if best_area >= stop_thr:
            speed = 0
        elif best_area >= slow_thr:
            speed = 20
        else:
            speed = 60

    cv2.putText(annotated, f"Speed: {speed}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return speed, annotated


def free_space_avoidance(frame, draw_frame=None):
    annotated = frame.copy() if draw_frame is None else draw_frame.copy()

    h, w = frame.shape[:2]
    roi_y = int(h * 0.45)
    roi = frame[roi_y:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    third = w // 3
    left = edges[:, :third]
    center = edges[:, third:2 * third]
    right = edges[:, 2 * third:]

    def free_score(part):
        if part.size == 0:
            return 1.0
        occupied = np.count_nonzero(part) / float(part.size)
        return 1.0 - occupied

    left_free = free_score(left)
    center_free = free_score(center)
    right_free = free_score(right)

    avoid_steer = 0.0
    speed_cap = 60
    action = "CLEAR"

    if center_free < FREE_SPACE_CENTER_MIN:
        if left_free > right_free + FREE_SPACE_SIDE_MARGIN:
            avoid_steer = -AVOID_STEER
            speed_cap = 25
            action = "TURN LEFT"
        elif right_free > left_free + FREE_SPACE_SIDE_MARGIN:
            avoid_steer = AVOID_STEER
            speed_cap = 25
            action = "TURN RIGHT"
        else:
            speed_cap = 0
            action = "STOP"

    cv2.putText(annotated, f"Free L/C/R: {left_free:.2f} {center_free:.2f} {right_free:.2f}",
                (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(annotated, f"Avoid: {action}",
                (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return avoid_steer, speed_cap, annotated


def on_collision(event):
    global collision_hold_until, collision_reason
    collision_reason = getattr(event.other_actor, "type_id", "unknown")
    collision_hold_until = time.time() + 2.5
    print("Collision with:", collision_reason)


def on_obstacle(event):
    global obstacle_distance, obstacle_actor, last_obstacle_time
    obstacle_distance = float(event.distance)
    obstacle_actor = getattr(event.other_actor, "type_id", "unknown")
    last_obstacle_time = time.time()
    print(f"Obstacle ahead: {obstacle_actor} at {obstacle_distance:.2f} m")


# =========================
# CARLA CONNECTION
# =========================
client = carla.Client("localhost", 2000)
client.set_timeout(60.0)
world = client.get_world()

# keep the current map already open in CARLA for smoother startup
world.set_weather(random.choice(WEATHERS))
blueprints = world.get_blueprint_library()

# =========================
# SPAWN MAIN VEHICLE
# =========================
vehicle = spawn_random_vehicle(world, blueprints)
if vehicle is None:
    print("Could not spawn vehicle")
    raise SystemExit

print("Vehicle spawned")

vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))

spectator = world.get_spectator()

# =========================
# RGB CAMERA
# =========================
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(CAMERA_W))
camera_bp.set_attribute("image_size_y", str(CAMERA_H))
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", str(SENSOR_TICK))

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

# =========================
# NPC TRAFFIC
# =========================
npc_vehicles = spawn_npc_vehicles(world, blueprints, client, count=NPC_COUNT)

# =========================
# COLLISION SENSOR
# =========================
collision_bp = blueprints.find("sensor.other.collision")
collision_sensor = world.spawn_actor(
    collision_bp,
    carla.Transform(),
    attach_to=vehicle
)
collision_sensor.listen(on_collision)

# =========================
# FRONT OBSTACLE SENSOR
# =========================
obstacle_bp = blueprints.find("sensor.other.obstacle")
obstacle_bp.set_attribute("distance", "20.0")
obstacle_bp.set_attribute("hit_radius", "0.5")
obstacle_bp.set_attribute("sensor_tick", "0.05")
if obstacle_bp.has_attribute("only_dynamics"):
    obstacle_bp.set_attribute("only_dynamics", "False")

obstacle_sensor = world.spawn_actor(
    obstacle_bp,
    carla.Transform(carla.Location(x=2.5, z=1.0)),
    attach_to=vehicle
)
obstacle_sensor.listen(on_obstacle)

# =========================
# CAMERA CALLBACK
# =========================
def process_image(image):
    global vehicle, frame_idx, prev_time, last_obstacle_speed, last_obstacle_frame
    global collision_hold_until, obstacle_distance, obstacle_actor, last_obstacle_time

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))
    frame = img[:, :, :3].copy()

    # hard emergency stop after collision
    if time.time() < collision_hold_until:
        emergency = frame.copy()
        cv2.putText(emergency, "COLLISION! EMERGENCY BRAKE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        cv2.imshow("CARLA Camera", emergency)
        cv2.waitKey(1)
        frame_idx += 1
        return

    lane_steer, curve_speed_cap, lane_mode, lane_frame = lane_follow(frame)

    if frame_idx % YOLO_EVERY_N == 0:
        obstacle_speed, obstacle_frame = obstacle_detection(frame, draw_frame=lane_frame)
        last_obstacle_speed = obstacle_speed
        last_obstacle_frame = obstacle_frame
    else:
        obstacle_speed = last_obstacle_speed
        obstacle_frame = last_obstacle_frame if last_obstacle_frame is not None else lane_frame

    avoid_steer, avoid_speed_cap, final_frame = free_space_avoidance(
        frame,
        draw_frame=obstacle_frame
    )

    # obstacle sensor safety layer
    obstacle_active = (
        obstacle_distance is not None
        and (time.time() - last_obstacle_time) < 0.35
    )

    obstacle_speed_cap = 60
    if obstacle_active:
        cv2.putText(
            final_frame,
            f"Obstacle: {obstacle_actor} {obstacle_distance:.1f}m",
            (20, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        if obstacle_distance <= OBSTACLE_BRAKE_DISTANCE:
            obstacle_speed_cap = 0
            final_steer = 0.0
            final_speed = 0
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            cv2.putText(final_frame, "OBSTACLE TOO CLOSE", (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            cv2.putText(final_frame, f"FPS: {fps:.1f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("CARLA Camera", final_frame)
            cv2.waitKey(1)
            frame_idx += 1
            return
        elif obstacle_distance <= OBSTACLE_WARN_DISTANCE:
            obstacle_speed_cap = 12

    # combine steering
    if lane_mode in ("left_only", "right_only"):
        final_steer = clamp(0.90 * lane_steer + 0.10 * avoid_steer, -0.70, 0.70)
    elif lane_mode == "memory":
        final_steer = clamp(0.95 * lane_steer + 0.05 * avoid_steer, -0.65, 0.65)
    else:
        final_steer = clamp(0.75 * lane_steer + 0.25 * avoid_steer, -0.60, 0.60)

    final_speed = min(obstacle_speed, curve_speed_cap, avoid_speed_cap, obstacle_speed_cap)

    # better behavior when lane is lost
    if lane_mode == "lost":
        if lane_miss_count <= LANE_LOSS_GRACE:
            final_speed = 12
            final_steer = clamp(prev_steer * 0.92, -STEER_MAX, STEER_MAX)
        else:
            final_speed = 0

    if lane_mode in ("left_only", "right_only") and final_speed > 12:
        final_speed = 12

    if final_speed == 0:
        throttle = 0.0
        brake = 1.0
    elif final_speed <= 12:
        throttle = 0.04
        brake = 0.0
    elif final_speed <= 20:
        throttle = 0.10
        brake = 0.0
    elif final_speed <= 40:
        throttle = 0.25
        brake = 0.0
    else:
        throttle = 0.55
        brake = 0.0

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=final_steer
        )
    )

    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    cv2.putText(final_frame, f"FPS: {fps:.1f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_frame, f"Lane mode: {lane_mode}", (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(final_frame, f"Lane steer: {lane_steer:.2f}", (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(final_frame, f"Final steer: {final_steer:.2f}", (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(final_frame, f"Final speed: {final_speed}", (20, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("CARLA Camera", final_frame)
    cv2.waitKey(1)

    frame_idx += 1


prev_time = time.time()
camera.listen(process_image)

# =========================
# RUN LOOP
# =========================
try:
    while True:
        loc = vehicle.get_location()

        spectator.set_transform(
            carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=20),
                carla.Rotation(pitch=-90)
            )
        )

        print(f"Vehicle Location: {loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}")
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    try:
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()
    except:
        pass

    try:
        if obstacle_sensor is not None:
            obstacle_sensor.stop()
            obstacle_sensor.destroy()
    except:
        pass

    try:
        if camera is not None:
            camera.stop()
            camera.destroy()
    except:
        pass

    try:
        if vehicle is not None:
            vehicle.destroy()
    except:
        pass

    for npc in npc_vehicles:
        try:
            npc.destroy()
        except:
            pass

    cv2.destroyAllWindows()