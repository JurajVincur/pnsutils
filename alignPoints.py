import csv
import cv2
import math
import numpy as np

def align(canonical_points, predicted_points):
    _, transform, _ = cv2.estimateAffine3D(canonical_points, predicted_points, confidence=1)
    rotation = transform[:,0:3]
    translation = transform[:,3:4].ravel()
    return rotation, translation

def rot2euler(R):
    p = np.hstack((R, np.zeros((3,1))))
    return cv2.decomposeProjectionMatrix(p)[-1]

if __name__=="__main__":

    side = "left"
    scale = 1

    data1 = None
    data2 = None
    
    with open(f"points_T265Camera.csv", newline='') as f1, open(f"points_LeapMotion.csv") as f2:
    #with open(f"points_T265_{side}.csv", newline='') as f1, open(f"points_Eye_{side}.csv") as f2:
        reader = csv.reader(f1, quoting=csv.QUOTE_NONNUMERIC)
        data1 = list(reader)
        reader = csv.reader(f2, quoting=csv.QUOTE_NONNUMERIC)
        data2 = list(reader)
    
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    r, t = align(data1,data2)
    
    print("OpenCV")
    print("r =",np.array2string(r, separator=','))
    print("t =",np.array2string(t, separator=','))
    
    r = rot2euler(r).ravel()
    t[1] *= -1
    print("Unity 1")
    print("r:",np.array2string(r, separator=','))
    print("t:",np.array2string(t * scale, separator=','))
    
    #what the hell? I probably need rotation from one to another and translation in reversed order
    _, t = align(data2,data1)
    r, _ = align(data1,data2)
    
    r = rot2euler(r).ravel()
    t[1] *= -1
    print("Unity 2")
    print("r:",np.array2string(r, separator=','))
    print("t:",np.array2string(t * scale, separator=','))