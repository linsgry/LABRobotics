from GUI import GUI
from HAL import HAL
import cv2


def haar_image():
  
    detected = False
    # get the haars cascade face xml
    haars_cascade = cv2.CascadeClassifier('/RoboticsAcademy/exercises/static/exercises/rescue_people_newmanager/haarcascade_frontalface_default.xml')
    # get the image of the dron
    img = HAL.get_ventral_image()

  
    # Rotate the image, change it to gray and search the faces
    for angle in [0, 45, -45, 90, -90, 135, -135, 180]:  
        # rotate for every 45º
        rotated_img = cv2.rotate(img, angle)
        rotated_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # search to the faces in the rotated image
        rotated_faces = haars_cascade.detectMultiScale(rotated_gray, scaleFactor=1.02, minNeighbors=2)
        
        # if it detect a face it return True
        for i in range(len(rotated_faces)):
            detected = True

    return img, detected


def arrive_to_pose(target_pos_x, target_pos_y):
    actual_pos = HAL.get_position()
    
    while not ((target_pos_x == round(actual_pos[0])) and (target_pos_y == round(actual_pos[1]))):
        actual_pos = HAL.get_position()
        continue

def repeated_persons(poses, new_pose):
    repeat = False
    
    for i in range(len(poses)):
        if ((abs(poses[i][0] - new_pose[0]) <= 2) and (abs(poses[i][1] - new_pose[1]) <= 2)):
            repeat = True
            
    if not (repeat):
        poses.append(new_pose)
    
    return poses
    
def send_people_position(poses):
    
    print("detected")
    # survivors poses
    for i in range(len(poses)):
        print("Person ", i ," detected at: (", poses[i][0], poses[i][1], ")")



# set constants and values
people_x = 30
people_y = -40
pos = HAL.get_position()
boat_x = pos[0]
boat_y = pos[1]
spiral_vel_x = 1 
spiral_vel_yaw = 0.3
survivors_poses = []
final = False
sended = False

# set velocities to 0
HAL.set_cmd_vel(0, 0, 0, 0)
# take off the dron 
HAL.takeoff(10)
# go to the GPS positon of the survivors
HAL.set_cmd_pos(people_x, people_y, 5, 0)
arrive_to_pose(people_x, people_y)

while True:
    if not (final):
        # while all the people wasn't detected continue 
        pos = HAL.get_position()
        vel = HAL.get_velocity()
        rpy = HAL.get_orientation()
        yaw = HAL.get_yaw_rate()
        HAL.set_cmd_vel(spiral_vel_x, 0, 0, spiral_vel_yaw)
        
        # search for people faces
        ventral_img, detected = haar_image()
        if detected:
            if (len(survivors_poses) > 0):
                repeated_persons(survivors_poses, pos)
            else:
                survivors_poses.append(pos)
        
        # if all people was detected go to the boat
        if (len(survivors_poses) == 6):
            final = True
        
        
        spiral_vel_x += 0.008
        GUI.showLeftImage(ventral_img)
    
    if (final):
        # if all people was detected send their positions andreturn to the boat
        if not (sended):
            send_people_position(survivors_poses)
            sended = True
        HAL.set_cmd_vel(0, 0, 0, 0)
        HAL.set_cmd_pos(boat_x, boat_y, 10, 0)
        arrive_to_pose(boat_x, boat_y)
        # land and finish
        HAL.land()
