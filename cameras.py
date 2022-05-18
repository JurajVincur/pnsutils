import cv2
import numpy as np
import math
import abc
import typing
import time

class Camera(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def readCapture(self) -> typing.Tuple[bool, np.ndarray]:
        return

    @abc.abstractmethod
    def release(self):
        return

    @property
    @abc.abstractmethod
    def resolution(self) -> typing.Tuple[int, int]:
        return

    @property
    @abc.abstractmethod
    def calibration(self) -> dict:
        return

    @property
    def fisheye(self):
        return False

    def readNewFrame(self, waitTime=0.1, tryCount=10):
        for _ in range(tryCount):
            ret, frame = self.readCapture()
            if(ret is True):
                return frame
            time.sleep(waitTime)
        self.release()
        raise Exception('cannot read frame')
        return None

class FakeCamera(Camera):

    def __init__(self, imgPath, calPath, fisheye):
        self._image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        self._calPath = calPath
        self._fisheye = fisheye
        return

    def readCapture(self):
        return True, self._image

    def release(self):
        return

    @property
    def resolution(self):
        return self._image.shape[:2]

    @property
    def fisheye(self):
        return self._fisheye

    @property
    def calibration(self):
        return np.load(self._calPath) #TODO consider read always vs init

class CV2Camera(Camera):

    def __init__(self, index, frameHeight, frameWidth):
        self._frameHeight = frameHeight
        self._frameWidth = frameWidth
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH , self._frameWidth)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frameHeight)
        return

    def readCapture(self):
        return self._cap.read()

    def release(self):
        self._cap.release()
        return

    @property
    def resolution(self):
        return (self._frameHeight, self._frameWidth)

    @property
    def calibration(self):
        return np.load("./cameraCalibration_cv2.npz", ) #TODO consider read always vs init

    def readNewFrame(self, waitTime=0.1, tryCount=10):
        return cv2.cvtColor(Camera.readNewFrame(self, waitTime, tryCount), cv2.COLOR_BGR2GRAY)

class ZedCamera(CV2Camera):

    @property
    def calibration(self):
        return np.load("./cameraCalibration_cv2_ZED.npz", )

class T265Camera(Camera): #can be potentially refactored to generic realsense camera however everything in intelutils is probably hardcoded for t265

    def __init__(self, autoExposure=False):
        import pyrealsense2 as rs2
        import intelutils
        self._cap = intelutils.intelCamThread(frame_callback = lambda frame: None, autoExposure=autoExposure)
        self._cap.start()
        self._frameWidth = 848 * 2
        self._frameHeight = 800
        return

    def readCapture(self):
        return self._cap.read()

    def release(self):
        self._cap.kill()
        return

    @property
    def calibration(self):
        return np.load("./cameraCalibration_rs.npz", ) #TODO consider read always vs init

    def readNewFrame(self, waitTime=0.1, tryCount=10):
        self.readCapture() #reseting newFrame flag in intelutils
        return Camera.readNewFrame(self, waitTime, tryCount)

    @property
    def resolution(self):
        return (self._frameHeight, self._frameWidth)

    @property
    def fisheye(self):
        return True

class LeapMotion(Camera):

    def __init__(self, index, frameHeight, frameWidth, timeout=3.0):
        import leapuvc
        self._frameHeight = frameHeight
        self._frameWidth = frameWidth
        self._cap = leapuvc.leapImageThread(index, resolution=(self._frameWidth >> 1, self._frameHeight), timeout=timeout)
        self._cap.start()
        return

    def readCapture(self):
        ret, frame = self._cap.read()
        if frame is not None:
            frame = np.hstack(frame[:2])
        return ret, frame

    def release(self):
        self._cap.timeout = -1
        return

    @property
    def calibration(self):
        return np.load("./cameraCalibration_leap.npz", ) #TODO consider read always vs init

    def readNewFrame(self, waitTime=0.1, tryCount=10):
        self.readCapture() #reseting newFrame flag in intelutils
        return Camera.readNewFrame(self, waitTime, tryCount)

    @property
    def resolution(self):
        return (self._frameHeight, self._frameWidth)