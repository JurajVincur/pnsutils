import numpy as np
import cv2
import math
import csv
from cameras import T265Camera

from procrustes import orthogonal

def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def solvePose(gray, objectPoints, cm, dc, blobDetector):
    keypoints = blobDetector.detect(gray)
    
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    ret, corners = cv2.findCirclesGrid(gray, (4,11), None, cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector)
    if ret == True:
        im_with_keypoints = cv2.drawChessboardCorners(im_with_keypoints, (4,11), corners, ret)
        
        # 3D posture
        #retval, rvec, tvec, _ = cv2.solvePnPRansac(objectPoints, corners, cm, dc)
        retval, rvec, tvec = cv2.solvePnP(objectPoints, corners, cm, dc)
        return True, rvec, tvec, im_with_keypoints
    return False, None, None, im_with_keypoints

def getPoints(rvec, tvec, objCoords):
    points = []
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rotation_matrix, tvec))
    for i in objCoords:
        op = np.array([np.append(i,[1])]).T
        points.append(np.dot(pose_mat, op).ravel())
    return np.array(points)

if __name__=="__main__":

    import uvc

    #blob detector
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.minThreshold = 23
    blobParams.maxThreshold = 255
    blobParams.filterByArea = True
    blobParams.minArea = 32     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 256   # maxArea may be adjusted to suit for your experiment
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.5
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.95
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    objectPoints = np.zeros((44, 3))  # In this asymmetric circle grid, 44 circles are adopted
    dist = 2
    for i in range(len(objectPoints)):
        objectPoints[i] = ((i>>2)*dist*0.5, (i%4)*dist + (((i%8)>>2)*dist*0.5), 0)
    objectPoints += [-2.5*dist, -1.75*dist, 0]

    objectPointsT265 = objectPoints#np.array([(-2.5,1.75,0), (-2.5,-1.75,0), (0,0,0), (2.5,1.75,0), (2.5,-1.75,0)]) * dist
    objectPointsEye = objectPointsT265 * [-1,1,1] #marker being seen from back

    print(objectPointsT265)
    print(objectPointsEye)

    calcPoints = False
    side = "right"
    uvcId = 3
    sideId = 0 if side == "left" else 1

    camera = T265Camera(autoExposure=True)
    calibT265 = camera.calibration
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(calibT265[f"{side}CameraMatrix"], calibT265[f"{side}DistCoeffs"], None, calibT265[f"{side}CameraMatrix"], (camera.resolution[1] >> 1, camera.resolution[0]), cv2.CV_32FC1)

    dev_list = uvc.device_list()
    cameraEye = uvc.Capture(dev_list[uvcId]["uid"])
    cameraEye.frame_mode = (400, 400, 120)
    for cont in cameraEye.controls:
        if cont.display_name == "Absolute Exposure Time":
            cont.value = 30
    calibEye = np.load(f"./cameraCalibrationET_{side}.npz", )

    map1Eye, map2Eye = cv2.initUndistortRectifyMap(calibEye["cameraMatrix"], calibEye["distCoeffs"], None, calibEye["cameraMatrix"], cameraEye.frame_mode[:-1], cv2.CV_32FC1)

    pointsT265 = []
    pointsEye = []

    while(True):
        newFrame, frame = camera.readCapture()
        
        waitedKey = cv2.waitKey(100) & 0xFF
        if(waitedKey == ord('q')):
            camera.release()
            break
            
        if(newFrame):
            imgslice = np.split(frame, 2, 1)[sideId]
            imgslice = cv2.remap(imgslice, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            
            ret, rvec, tvec, im_with_keypoints = solvePose(imgslice, objectPointsT265, calibT265[f"{side}CameraMatrix"], np.zeros(5), blobDetector)
            
            imgEye = cameraEye.get_frame_robust().img #cv2.flip(cameraEye.get_frame_robust().img, 1) # NO!
            imgEye = cv2.cvtColor(imgEye, cv2.COLOR_BGR2GRAY)
            imgEye = cv2.remap(imgEye, map1Eye, map2Eye, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            ret1, rvec1, tvec1, im_with_keypoints_eye = solvePose(imgEye, objectPointsEye, calibEye["cameraMatrix"], np.zeros(5), blobDetector)
            
            if ret is True and ret1 is True:
                if(waitedKey == ord('c')):
                    tvec = np.dot(calibT265[f"R{sideId+1}"], tvec) #bring tvec to rectified cs - check
                    if calcPoints:
                        for t in getPoints(rvec, tvec, objectPoints):
                            pointsT265.append(t.ravel())
                        for t in getPoints(rvec1, tvec1, objectPoints):
                            pointsEye.append(t.ravel())
                    else:
                        pointsT265.append(tvec.ravel())
                        pointsEye.append(tvec1.ravel())
                    print(tvec)
            cv2.imshow("T265Frame", im_with_keypoints)
            cv2.imshow("EyeFrame", im_with_keypoints_eye)
    with open(f"points_T265_{side}.csv", 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(pointsT265)
    with open(f"points_Eye_{side}.csv", 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(pointsEye)