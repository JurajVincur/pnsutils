import cv2
import leapuvc
import numpy as np

capResolution = (640, 480)
cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, capResolution[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, capResolution[1])
calibration = leapuvc.retrieveLeapCalibration(cam, capResolution)
ret, frame = cam.read()
if ret is True:
    cv2.imshow("F", frame)
    cv2.waitKey(0)
else:
    print("Something went wrong!")
cam.release()

print(calibration)

cal = {}
for i, cam in enumerate(("left", "right")):
    cal[cam + "CameraMatrix"] = calibration[cam]["extrinsics"]["cameraMatrix"]
    cal[f"R{i+1}"] = calibration[cam]["extrinsics"]["r"]
    cal[cam + "DistCoeffs"] = calibration[cam]["intrinsics"]["distCoeffs"]

#baseline correction
loff = -np.array(calibration["left"]["intrinsics"]["offset"])
roff = -np.array(calibration["right"]["intrinsics"]["offset"])
loff += np.array([-calibration["baseline"]*0.5, 0])
roff += np.array([calibration["baseline"]*0.5, 0])
cal["baseline"] = calibration["baseline"]#np.linalg.norm(loff - roff)
print(cal)

np.savez("cameraCalibration_leap.npz",
    leftCameraMatrix=cal["leftCameraMatrix"],
    rightCameraMatrix=cal["rightCameraMatrix"],
    leftDistCoeffs=cal["leftDistCoeffs"],
    rightDistCoeffs=cal["rightDistCoeffs"],
    R1=cal["R1"],
    R2=cal["R2"],
    baseline=cal["baseline"]*0.001
)
