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
        
def triangulate(leftCorners, rightCorners, baseline, postRotation=None):
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
    return triangulated
    
if __name__=="__main__":
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    checkerSizes = ((6, 5), (9, 4), (10, 3), (8, 3))
    postRotation = None
    checkerId = 0
    masks = [[], []]
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
            checkerId = (checkerId + 1) % len(checkerSizes)
            _allTriangulated = []
            for _allCorners in toProcess:
                _allRectified = []
                for sideId in range(len(_allCorners)):
                    masks[sideId].append(cv2.convexHull(_allCorners[sideId])[:, 0].astype(int))
                    _allRectified.append(cv2.undistortPoints(_allCorners[sideId], calib[f"{sides[sideId]}CameraMatrix"], np.zeros(5), R = calib[f"R{sideId+1}"]))
                _allTriangulated.append(triangulate(_allRectified[0], _allRectified[1], baseline, postRotation))
            mean = np.median(_allTriangulated, axis=0) #maybe do this before rectify and triangulate
            for p in mean:
                strVals = [str(x) for x in p]
                writer.writerow(strVals)
            print(f"Median of {len(_allTriangulated)} samples stored")
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
                
                if checkerId == 0:
                    masks[sideId].clear()
                for mask in masks[sideId]:
                    cv2.fillConvexPoly(gray, mask, (0, 0, 0))
                
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                
                roi = rois[sideId]
                if sum(roi) != 0:
                    gray = np.zeros(imgslice.shape, imgslice.dtype)
                    npslice = np.s_[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                    gray[npslice] = imgslice[npslice]
                    img = cv2.rectangle(img, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0,255,0), 1)
                
                ret, corners = cv2.findChessboardCorners(gray, checkerSizes[checkerId], None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
                im_with_keypoints = img
                
                if ret == True:
                    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                    allCorners.append(corners)
                    im_with_keypoints = cv2.drawChessboardCorners(im_with_keypoints, checkerSizes[checkerId], corners, ret)
                cv2.imshow(f"{sides[sideId]} frame", im_with_keypoints)
        if(len(allCorners)==2 and waitedKey == ord('c')):
            toProcess.append(allCorners)
            print(f"Pattern number {len(toProcess)} captured")
    file.close()