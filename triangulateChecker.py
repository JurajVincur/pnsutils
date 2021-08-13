import numpy as np
import cv2
from cameras import T265Camera, LeapMotion
import csv

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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    
    camera = T265Camera()
    """
    camera = LeapMotion(0, 480, 1280)
    camera._cap.setExposure(7500)
    camera._cap.setCenterLED(False)
    """
    
    file = open(f"points_{type(camera).__name__}.csv", 'w', newline='')
    writer = csv.writer(file)
    
    calib = camera.calibration
    sides = ["left", "right"]
    
    uf = cv2.fisheye.initUndistortRectifyMap if camera.fisheye else cv2.initUndistortRectifyMap
    
    maps = [uf(calib[f"{side}CameraMatrix"], calib[f"{side}DistCoeffs"], None, calib[f"{side}CameraMatrix"], (camera.resolution[1] >> 1, camera.resolution[0]), cv2.CV_32FC1) for sideId, side in enumerate(sides)]
    baseline = calib["baseline"]
    
    while(True):
        allCorners = []
        newFrame, frame = camera.readCapture()
        
        waitedKey = cv2.waitKey(100) & 0xFF
        if(waitedKey == ord('q')):
            camera.release()
            break
            
        if(newFrame):
            for sideId, imgslice in enumerate(np.split(frame, 2, 1)):
                imgslice = cv2.remap(imgslice, maps[sideId][0], maps[sideId][1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                gray = imgslice
                
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                ret, corners = cv2.findChessboardCorners(gray, (9,6), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
                im_with_keypoints = img
                
                if ret == True:
                    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    rectified = cv2.undistortPoints(corners, calib[f"{sides[sideId]}CameraMatrix"], np.zeros(5), R = calib[f"R{sideId+1}"])
                    allCorners.append(rectified)
                    im_with_keypoints = cv2.drawChessboardCorners(im_with_keypoints, (9,6), corners, ret)
                cv2.imshow(f"{sides[sideId]} frame", im_with_keypoints)
        if(len(allCorners)==2 and waitedKey == ord('c')):
            leftCorners, rightCorners = allCorners
            for i in range(len(leftCorners)):
                leftCorner = leftCorners[i]
                rightCorner = rightCorners[i]
                
                a = np.array([-baseline*0.5,0,0]) #baseline 0.64
                aDir = leftCorner.ravel()
                aDir = np.append(aDir, 1)
                b = np.array([baseline*0.5,0,0])
                bDir = rightCorner.ravel()
                bDir = np.append(bDir, 1)
                
                ab = rayRayIntersection(a, aDir, b, bDir)
                ba = rayRayIntersection(b, bDir, a, aDir)
                merged = (ab + ba) * 0.5
                strVals = [str(x) for x in merged]
                print(",".join(strVals))
                writer.writerow(strVals)
            print("")
    file.close()