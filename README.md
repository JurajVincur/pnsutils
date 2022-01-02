# pnsutils

- *alignPoints.py* - alternative implementation of *kabschPoints.py*

- *board.jpg* - dummy image showed to t265 during its initialization (if camera frame is too dark t265 won't start streaming)

- *cameras.py* - implementation of all cameras should be eventually provided here

- *eyeCameraAligner.py* - capturing 3d points for ET camera to t265 camera alignment

- *eyeCameraCalibration.py* - pyuvc ET camera calibration

- *graycodeCalibration.py* - creating screen to camera mappings (gradients) that are being used to construct virtual cameras in 3D-Calibration-VCam

- *intelutils.py* - core implementation for t26x

- *kabschPoints.py* - script that computes relative pose between sensors based on set of 3d points captured by *eyeCameraAligner.py* or *triangulateX.py*

- *leapuvc.py* - core implementation for LMC

- *lmcCalibration.py* - script that reads calibration parameters from LMC and store them to calibration file

- *plotTriangulatedPoints.py* - script for debug purposes that shows captured 3d points and their final alignment given by relative poses from *kabschPoints.py* or *alignPoints.py*

- *pupilLabsTracking.py* - eye tracking based on pye3d which send ET data to Unity app via zmq

- *rect.py* - script for debug purposes that show rectified t265 frames

- *calibrationRig/controller.py* - app to control motorized rig

- *calibrationRig/driver.ino* - arduino code for motorized rig

- *triangulateChecker.py*, *triangulateChecker2.py*, *triangulateCircles.py* - 3 alternative implementations that capture 3d points with stereo cameras that will be used to derive relative pose between them
