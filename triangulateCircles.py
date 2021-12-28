import numpy as np
import cv2
from cameras import T265Camera, LeapMotion, FakeCamera
import csv
import sys
import math

def closestAlphaOnSegmentToLine(segA, segB, lineA, lineB):
    lineBA = lineB - lineA
    lineDirSqrMag = np.dot(lineBA, lineBA)
    inPlaneA = segA - ((np.dot(segA - lineA, lineBA) / lineDirSqrMag) * lineBA)
    inPlaneB = segB - ((np.dot(segB - lineA, lineBA) / lineDirSqrMag) * lineBA)
    inPlaneBA = inPlaneB - inPlaneA
    return np.dot(lineA - inPlaneA, inPlaneBA) / np.dot(inPlaneBA, inPlaneBA) if (lineDirSqrMag != 0.0 and np.array_equal(inPlaneA, inPlaneB) is False) else 0.0

def rayRayIntersection(rayAOrigin, rayADirection, rayBOrigin, rayBDirection):
    alpha = closestAlphaOnSegmentToLine(rayAOrigin, rayAOrigin + rayADirection, rayBOrigin, rayBOrigin + rayBDirection)

    if (alpha > 0.0):
        return rayAOrigin + rayADirection * alpha
    else:
        return None
        
        
if __name__=="__main__":
    
    #blob detector
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.minThreshold = 50
    blobParams.maxThreshold = 255
    blobParams.filterByArea = True
    blobParams.minArea = 8
    blobParams.maxArea = 128
    #blobParams.minArea = 8     # minArea may be adjusted to suit for your experiment
    #blobParams.maxArea = 64   # maxArea may be adjusted to suit for your experiment
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.5
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.95
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    
    postRotation = None
    if(len(sys.argv) < 2):
        print("triangulateChecker.py camname [index]")
        print("triangulateChecker.py fake imgPath calPath fisheye [camera name]")
        sys.exit(0)
    camName = sys.argv[1]
    camera = None
    fname = None
    if(camName.upper() == "T265"):
        camera = T265Camera(autoExposure=True)
    elif(camName.upper() == "LMC"):
        camera = LeapMotion(int(sys.argv[2]) if len(sys.argv) > 2 else 0, 480, 1280, 60)
        camera._cap.setExposure(4500)
        camera._cap.setCenterLED(False)
        postRotation = np.array([
            [math.cos(math.pi), -math.sin(math.pi), 0],
            [math.sin(math.pi), math.cos(math.pi), 0],
            [0, 0, 1]
        ]) #need to rotate LMC results by 180deg bcs of LM reverse
    elif(camName.upper() == "FAKE"):
        camera = FakeCamera(sys.argv[2], sys.argv[3], sys.argv[4] == "1")
        if len(sys.argv) > 4:
            fname = sys.argv[5]
    if fname is None:
        fname = type(camera).__name__
    
    file = open(f"points_{fname}.csv", 'w', newline='')
    writer = csv.writer(file)
    
    calib = camera.calibration
    sides = ["left", "right"]
    
    uf = cv2.fisheye.initUndistortRectifyMap if camera.fisheye else cv2.initUndistortRectifyMap
    
    maps = [uf(calib[f"{side}CameraMatrix"], calib[f"{side}DistCoeffs"], None, calib[f"{side}CameraMatrix"], (camera.resolution[1] >> 1, camera.resolution[0]), cv2.CV_32FC1) for sideId, side in enumerate(sides)]
    baseline = calib["baseline"]
    rois = [(0,0,0,0) for _ in sides]
    
    savedImageId = 0
    toProcess = []
    while(True):
        allCorners = []
        newFrame, frame = camera.readCapture()
        
        waitedKey = cv2.waitKey(100) & 0xFF
        if(waitedKey == ord('q')):
            camera.release()
            break
        elif(waitedKey == ord('p')):
            mean = np.median(toProcess, axis=0)
            for p in mean:
                strVals = [str(x) for x in p]
                writer.writerow(strVals)
            print(f"Median of {len(toProcess)} samples stored")
            toProcess.clear()
        if(newFrame):
            for sideId, imgslice in enumerate(np.split(frame, 2, 1)):
                imgslice = cv2.remap(imgslice, maps[sideId][0], maps[sideId][1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                if(waitedKey == ord('s')):
                    cv2.imwrite(f"frame_{fname}_{savedImageId}_{sideId}.png", frame)
                    savedImageId += 1
                elif(waitedKey == ord('r')):
                    rois[sideId] = cv2.selectROI("ROI selector", imgslice)
                    print("Selected ROI:", rois[sideId])
                    cv2.destroyWindow("ROI selector")
                
                gray = imgslice
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                
                roi = rois[sideId]
                if sum(roi) != 0:
                    gray = np.zeros(imgslice.shape, imgslice.dtype)
                    npslice = np.s_[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                    gray[npslice] = imgslice[npslice]
                    img = cv2.rectangle(img, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0,255,0), 1)
                keypoints = blobDetector.detect(gray)
                im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                ret, corners = cv2.findCirclesGrid(gray, (4,11), None, cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector)
                if ret == True:
                    rectified = cv2.undistortPoints(corners, calib[f"{sides[sideId]}CameraMatrix"], np.zeros(5), R = calib[f"R{sideId+1}"])
                    allCorners.append(rectified)
                    im_with_keypoints = cv2.drawChessboardCorners(im_with_keypoints, (4,11), corners, ret)
                cv2.imshow(f"{sides[sideId]} frame", im_with_keypoints)
        if(len(allCorners)==2 and waitedKey == ord('c')):
            leftCorners, rightCorners = allCorners
            triangulated = []
            for i in range(len(leftCorners)):
                leftCorner = leftCorners[i]
                rightCorner = rightCorners[i]
                
                #ray ray
                a = np.array([-baseline*0.5,0,0]) #baseline 0.64
                aDir = leftCorner.ravel()
                aDir = np.append(aDir, 1)
                b = np.array([baseline*0.5,0,0])
                bDir = rightCorner.ravel()
                bDir = np.append(bDir, 1)
                
                ab = rayRayIntersection(a, aDir, b, bDir)
                ba = rayRayIntersection(b, bDir, a, aDir)
                merged = (ab + ba) * 0.5
                if postRotation is not None:
                    merged = np.dot(postRotation, merged)
                triangulated.append(merged)
                
            toProcess.append(triangulated)
            print(f"Pattern number {len(toProcess)} captured")
    file.close()