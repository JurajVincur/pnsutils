import numpy as np
import cv2
#from graycodeCalibration import CV2Camera
from graycodeCalibration import T265Camera

#camera = CV2Camera(1, 720, 2560)
camera = T265Camera()

def rot2euler(R):
    p = np.hstack((R, np.zeros((3,1))))
    return cv2.decomposeProjectionMatrix(p)[-1]

while(True):
    newFrame, frame = camera.readCapture()
    if(newFrame):
        leftRightImage = np.split(frame, 2, 1)
        ivecs = []
        colorFrames = []
        for ii, image in enumerate(("left", "right")):

            camera_matrix = camera.calibration[image + "CameraMatrix"]
            dist_coeffs = camera.calibration[image + "DistCoeffs"]
        
            r = camera.calibration["R" + str(ii+1)]
            print(rot2euler(r))
            #p = camera.calibration["P" + str(ii+1)]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, r, camera_matrix, (camera.resolution[1] >> 1, camera.resolution[0]), cv2.CV_32FC1)
            leftRightImage[ii] = cv2.remap(leftRightImage[ii], map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        cv2.imshow(type(camera).__name__ + ' Frame', cv2.resize(np.hstack(leftRightImage), dsize=(camera.resolution[1], camera.resolution[0])))#
    waitedKey = cv2.waitKey(100) & 0xFF
    if(waitedKey == ord('q')):
        camera.release()
        break