import cv2
import numpy as np
import carla

client=carla.Client('localhost',2000)
client.set_timeout(10.0)
world=client.get_world()

blueprints=world.get_blueprint_library()

vehicle_bp=blueprints.filter('vehicle.tesla.model3')[0]

spawn_point=world.get_map().get_spawn_points()[0]

vehicle=world.spawn_actor(vehicle_bp,spawn_point)

camera_bp = blueprints.find('sensor.camera.rgb')

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)


def obstacle_detection(frame): #source is the video which is going to be played
    
    orignal_frame=frame.copy()

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.GaussianBlur(frame,(5,5),0)

    #otsu thresholding
    threshold_value,new_frame=cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,hierarchy=cv2.findContours(new_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_box = None

    for contour in contours: #contours is a list of all detected object boundaries. like obstacle 1, obstacle 2, obstacle 3
        
        area = cv2.contourArea(contour)

        if area<1000:       #to remove unwanted noise(like a filter)
            continue

        if area > largest_area:
            x, y, w, h = cv2.boundingRect(contour)
            largest_area=area
            largest_box=(x,y,w,h)
            
    if largest_box is not None:

        x,y,w,h=largest_box

        cv2.rectangle(orignal_frame,(x,y),(x+w,y+h),(0,255,0),2)

        box_area=w*h    #we take the area taken on the ground only

        if box_area>30000:
            speed=0
        elif box_area>10000:
            speed=20
        else:
            speed=60
            
    return speed
                