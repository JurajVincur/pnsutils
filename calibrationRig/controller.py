import serial
import cv2
import numpy as np

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
arduino.set_buffer_size(rx_size = 1, tx_size = 1)

#states
#1 idle
#2 moving up
#3 moving down
#4 tilt up
#5 tilt down

fontScale = 1
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
thickness = 2
img = np.zeros((480,640))
cv2.putText(img, 'Press 2 for camera up', (10,30), font, fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(img, 'Press 3 for camera down', (10,80), font, fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(img, 'Press 4 to tilt up', (10,130), font, fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(img, 'Press 5 to tilt down', (10,180), font, fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(img, 'Executing (1 == nothing): ', (10,280), font, fontScale, color, thickness, cv2.LINE_AA)

while True:
    arduino.reset_input_buffer()
    x = arduino.read()
    currentState = int.from_bytes( x, "little")
    img[255:305, 450:640] = 0
    cv2.putText(img, str(currentState), (450,280), font, fontScale, color, thickness, cv2.LINE_AA)
    key = cv2.waitKey(100) & 0xFF
    if(x == b'\x01'):
        if key == ord('2'):
            arduino.write(b'\x02')
        elif key == ord('3'):
            arduino.write(b'\x03')
        elif key == ord('4'):
            arduino.write(b'\x04')
        elif key == ord('5'):
            arduino.write(b'\x05')
        elif key == ord('6'):
            arduino.write(b'\x06')
        elif key == ord('7'):
            arduino.write(b'\x07')
        elif key == ord('8'):
            arduino.write(b'\x08')
        elif key == ord('9'):
            arduino.write(b'\x09')
    if key == ord('q') or key == 27:
        break
    cv2.imshow('Rig controls', img)