import os
import cv2
import numpy as np
from dynamixel_sdk import *

# simulation
# right side_simul
#mp4_file = "/home/choi/Desktop/choi_cam/8_lt_cw_100rpm_in.mp4"

# left side_simul
# mp4_file = "/home/choi/Desktop/choi_cam/6_lt_ccw_100rpm_out.mp4"
# cap = cv2.VideoCapture(mp4_file)

# real cam
# Open the webcam (0 is the default camera, change if necessary)
# Camera and Frame Parameters
cap = cv2.VideoCapture(0)

frame_width = 500
frame_height = 500
roi_x_start, roi_x_end = 100, 399
roi_y_start, roi_y_end = 400, 499

# Dynamixel Parameters
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
LEN_GOAL_VELOCITY = 4

BAUDRATE = 1000000

PROTOCOL_VERSION = 2.0

DXL1_ID = 1
DXL2_ID = 2

DEVICENAME = '/dev/ttyACM0'

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DXL_VELOCITY_LIMIT = 200 # limit velocity dynamixel error value

# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

dxl1_goal_velocity = 100
dxl2_goal_velocity = 100
           
# Control logic based on centroid
center_line = (roi_x_end - roi_x_start) // 2
box_width = roi_x_end - roi_x_start

# deal with error situation
def initialize_dynamixel():
    if not portHandler.openPort():
        print("Failed to open the port")
        quit()

    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to set baudrate")
        quit()

    for dxl_id in [DXL1_ID, DXL2_ID]:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(f"Error enabling torque for Dynamixel ID {dxl_id}")
            quit()

    print("Dynamixels initialized and torque enabled.")

# set Dynamixel error value (fix value = 100)
def set_velocity(dxl_id, velocity):
    param_goal_velocity = [DXL_LOBYTE(DXL_LOWORD(velocity)), DXL_HIBYTE(DXL_LOWORD(velocity)), \
                           DXL_LOBYTE(DXL_HIWORD(velocity)), DXL_HIBYTE(DXL_HIWORD(velocity))]
    # def writeTxRx(self, port, dxl_id, address, length, data):
    dxl_comm_result, dxl_error = packetHandler.writeTxRx(portHandler, dxl_id, ADDR_GOAL_VELOCITY, len(param_goal_velocity), param_goal_velocity)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        print(f"Error setting velocity for Dynamixel ID {dxl_id}")

# Initialize Dynamixel
initialize_dynamixel()

if not cap.isOpened():
    print("Error: Could not open camera.")
    quit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize and process frame
    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    _, binary_frame = cv2.threshold(blurred_frame, 127, 255, cv2.THRESH_BINARY_INV)

    # Define ROI and calculate its centroid
    roi = binary_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("binary_frame",binary_frame)
    # line_tracer algorithm
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Map centroid to frame coordinates
            cx_global = cx + roi_x_start

            dxl1_goal_velocity = int(DXL_VELOCITY_LIMIT * (1 + (cx - center_line) / center_line) / 2)
            dxl2_goal_velocity = int(DXL_VELOCITY_LIMIT * (1 - (cx - center_line) / center_line) / 2)
            set_velocity(DXL1_ID, dxl1_goal_velocity)
            set_velocity(DXL2_ID, dxl2_goal_velocity)
           
            # Draw centroid
            cv2.circle(frame_resized, (cx_global, roi_y_start + cy), 5, (0, 0, 255), -1)

    # Show frames
    cv2.rectangle(frame_resized, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)
    cv2.imshow("Frame", frame_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
for dxl_id in [DXL1_ID, DXL2_ID]:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()