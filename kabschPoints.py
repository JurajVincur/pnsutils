import csv
import cv2
import math
import numpy as np

def kabsch(canonical_points, predicted_points):
    """
    rotation from preditcted to canonical
    translation from predicted to canonical
    """
    canonical_mean = np.mean(canonical_points, axis=0)
    predicted_mean = np.mean(predicted_points, axis=0)

    canonical_centered = canonical_points - np.expand_dims(canonical_mean, axis=0)
    predicted_centered = predicted_points - np.expand_dims(predicted_mean, axis=0)

    cross_correlation = predicted_centered.T @ canonical_centered

    u, s, vt = np.linalg.svd(cross_correlation)

    rotation = u @ vt

    det = np.linalg.det(rotation)

    if det < 0.0:
        vt[-1, :] *= -1.0
        rotation = np.dot(u, vt)

    translation = predicted_mean - canonical_mean
    translation = np.dot(rotation, translation) - np.dot(rotation, predicted_mean) + predicted_mean

    return rotation, translation

def rot2euler(R):
    p = np.hstack((R, np.zeros((3,1))))
    return cv2.decomposeProjectionMatrix(p)[-1]

if __name__=="__main__":

    side = "right"
    scale = 1

    data1 = None
    data2 = None
    
    #with open(f"points_FakeT265.csv", newline='') as f1, open(f"points_FakeLeapMotion.csv") as f2:
    #with open(f"points_T265Camera.csv", newline='') as f1, open(f"points_LeapMotion.csv") as f2:
    with open(f"points_T265_{side}.csv", newline='') as f1, open(f"points_Eye_{side}.csv") as f2:
        reader = csv.reader(f1, quoting=csv.QUOTE_NONNUMERIC)
        data1 = list(reader)
        reader = csv.reader(f2, quoting=csv.QUOTE_NONNUMERIC)
        data2 = list(reader)
    
    r, t = kabsch(data1,data2)

    print("OpenCV")
    print("r =",np.array2string(r, separator=','))
    print("t =",np.array2string(t, separator=','))

    r = rot2euler(r).ravel()
    t[1] *= -1
    print("Unity 1")
    print("r:",np.array2string(r, separator=','))
    print("t:",np.array2string(t * scale, separator=','))

    #what the hell? I probably need rotation from one to another and translation in reversed order
    _, t = kabsch(data2,data1)
    r, _ = kabsch(data1,data2)

    r = rot2euler(r).ravel()
    t[1] *= -1
    print("Unity 2")
    print("r:",np.array2string(r, separator=','))
    print("t:",np.array2string(t * scale, separator=','))