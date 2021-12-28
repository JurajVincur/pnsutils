import numpy as np
import cv2
import uvc

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001)

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

objectPoints = np.zeros((44, 3), np.float32)  # In this asymmetric circle grid, 44 circles are adopted
dist = 2
for i in range(len(objectPoints)):
    objectPoints[i] = ((i>>2)*dist*0.5, (i%4)*dist + (((i%8)>>2)*dist*0.5), 0)
objectPoints += [-2.5*dist, -1.75*dist, 0]
print(objectPoints)

objpoints = []
imgpoints = []

dev_list = uvc.device_list()
print(dev_list)
cap = uvc.Capture(dev_list[3]["uid"])
cap.frame_mode = (400, 400, 120)
for cont in cap.controls:
    if cont.display_name == "Absolute Exposure Time":
        cont.value = 15
print(cap.get_frame_robust().img.shape)

found = 0
while(True):
    
    waitedKey = cv2.waitKey(100) & 0xFF
    if(waitedKey == ord('q')):
        break
    
    frame = cap.get_frame_robust() # Capture frame-by-frame
    img = frame.img #cv2.flip(frame.img, 1) # NO!
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints = blobDetector.detect(gray) # Detect blobs.

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    if ret == True:
        #objpoints.append(objectPoints)  # Certainly, every loop objp is the same, in 3D.

        corners2 = corners#cv2.cornerSubPix(im_with_keypoints_gray, corners, (3,3), (-1,-1), criteria)    # Refines the corner locations.
        #imgpoints.append(corners2)

        # Draw and display the corners.
        im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
        
        if(waitedKey == ord('c')):
            found += 1
            print("Registered pattern number:", found)
            objpoints.append(objectPoints)
            imgpoints.append(corners2)
        
    cv2.imshow("img", im_with_keypoints) # display

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()

if(found >= 10):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("RMS error:", ret)
    print("CM:", mtx)
    print("DC:", dist)
    np.savez("cameraCalibrationET_left.npz",
        cameraMatrix=mtx,
        distCoeffs=dist
    )
else:
    print("Not enough patterns registered. Please capture at least 10 patterns.")
