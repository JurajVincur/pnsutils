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

def rot2euler(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

if __name__=="__main__":

    data1 = None
    data2 = None
    
    with open('points_T265Camera.csv', newline='') as f1, open('points_LeapMotion.csv') as f2:
        reader = csv.reader(f1, quoting=csv.QUOTE_NONNUMERIC)
        data1 = list(reader)
        reader = csv.reader(f2, quoting=csv.QUOTE_NONNUMERIC)
        data2 = list(reader)
    
    r, t = kabsch(data2,data1)
    r = 180*rot2euler(r)/math.pi
    #r = 180*cv2.Rodrigues(r)[0].ravel()/math.pi #angleaxis
    t[1] *= -1
    #r[0] *= -1
    #r[2] *= -1
    print("Final")
    print("r:",",".join([str(x) for x in r]))
    print("t:",",".join([str(x) for x in t]))