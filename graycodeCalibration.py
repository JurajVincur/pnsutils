import os
import numpy as np
import cv2
import math
import time
import abc
import typing
from wand.image import Image
import sys
from polyHelpers import calcCoeffs

def wait(timeInMs):
    timeInNs = timeInMs * 1000000
    t = time.time_ns()
    while(time.time_ns() - t < timeInNs):
        cv2.waitKey(10)
    return

def cachedArray(func):
    def inner(self, height, width, *args, **kwargs):
        if(hasattr(self, "_cache") is not True):
            self._cache = {}
        cachedValue = self._cache.get(func.__name__)
        if(cachedValue is None or cachedValue.shape != (height, width)):
            self._cache[func.__name__] = func(self, height, width, *args, **kwargs)
        return self._cache[func.__name__]
    return inner

class Borg:
    
    _shared_state = {}
    
    def __init__(self):
        self.__dict__ = self._shared_state
        if(getattr(self, "_initialized", False) is False):
            self.initialize()
            self._initialized = True
        return
    
    def initialize(self):
        return

class CalibrationHelpers(Borg):
    
    _shared_state = {}
    
    def __init__(self):
        Borg.__init__(self)
        self.continuum = self._continuum()
        return
    
    @cachedArray
    def allWhite(self, height, width):
        return np.ones((height, width), dtype=np.uint8) * 255
    
    @cachedArray
    def allDark(self, height, width):
        return np.zeros((height, width), dtype=np.uint8)
    
    def _continuum(self):
        c = np.arange(0, 256, dtype=np.uint8)
        c = np.bitwise_xor(c, c//2) # Binary to Gray
        return c
    
    @cachedArray
    def widthContinuum(self, height, width, splitscreen):
        wc = self.allDark(height, width)
        c = self.continuum
        if splitscreen is False:
            wc = cv2.resize(c[None, :], (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            wc[:, : int(width / 2)] = cv2.resize(c[None, :], (int(width / 2), height), interpolation=cv2.INTER_NEAREST)
            wc[:, int(width / 2) :] = wc[:, : int(width / 2)]
        return wc
        
    @cachedArray
    def heightContinuum(self, height, width):
        return cv2.resize(self.continuum[:, None], (width, height), interpolation=cv2.INTER_NEAREST)
        
    @cachedArray
    def widthBits(self, height, width, splitscreen=False):
        wc = self.widthContinuum(height, width, splitscreen)
        return np.unpackbits(wc[: , :, None].astype(np.uint8), axis=-1)
        
    @cachedArray
    def heightBits(self, height, width):
        hc = self.heightContinuum(height, width)
        return np.unpackbits(hc[:, :, None].astype(np.uint8), axis=-1)
        
    @staticmethod
    def calibration2GLSL(cal):
        glslStrs = []
        for key, coeffs in cal.items():
            glslStr = f"float[] {key} = float[{len(coeffs)}] ("
            glslStr += ", ".join(map(str, coeffs))
            glslStr += ");"
            glslStrs.append(glslStr)
        return "\n".join(glslStrs)

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
    @abc.abstractmethod
    def fisheye(self) -> bool:
        return
    
    def readNewFrame(self, undistort=False, waitTime=0.1, tryCount=50):
        for _ in range(tryCount):
            ret, frame = self.readCapture()
            if(ret is True):
                if(undistort is True):
                    if(hasattr(self, "maps") is False):
                        self.maps = []
                        iurmFunc = cv2.fisheye.initUndistortRectifyMap if self.fisheye is True else cv2.initUndistortRectifyMap
                        for ii, side in enumerate(("left", "right")):
                            camera_matrix = self.calibration[side + "CameraMatrix"]
                            dist_coeffs = self.calibration[side + "DistCoeffs"]
                            r = None
                            self.maps.append(iurmFunc(camera_matrix, dist_coeffs, r, camera_matrix, (self.resolution[1] >> 1, self.resolution[0]), cv2.CV_32FC1))
                    leftRightImage = np.split(frame, 2, axis=1)
                    for i in range(2):
                        leftRightImage[i] = cv2.remap(leftRightImage[i], self.maps[i][0], self.maps[i][1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                    frame = np.hstack(leftRightImage)
                #store frames
                #if(hasattr(self, "currentFrameId") is False):
                #    self.currentFrameId = 0
                #cv2.imwrite(f"./frames/image{self.currentFrameId}.png", frame)
                #self.currentFrameId += 1
                return frame
            time.sleep(waitTime)
        self.release()
        raise Exception('cannot read frame')
        return None

class T265Camera(Camera): #can be potentially refactored to generic realsense camera however everything in intelutils is probably hardcoded for t265
    
    def __init__(self):
        import pyrealsense2 as rs2
        import intelutils
        self._cap = intelutils.intelCamThread(frame_callback = lambda frame: None, exposure = 10000)
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
    
    @property
    def resolution(self):
        return (self._frameHeight, self._frameWidth)
        
    @property
    def fisheye(self):
        return True

class CV2Camera(Camera):
    
    def __init__(self, index, frameHeight, frameWidth):
        self._frameHeight = frameHeight
        self._frameWidth = frameWidth
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH , self._frameWidth)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frameHeight)
        self.readNewFrame() #first frame not reliable
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
    
    def readNewFrame(self, undistort=False, waitTime=0.1, tryCount=10):
        return cv2.cvtColor(Camera.readNewFrame(self, undistort, waitTime, tryCount), cv2.COLOR_BGR2GRAY)
        
    @property
    def fisheye(self):
        return False
        
class FakeCapture:
    
    def __init__(self, folder, basename):
        self.currentFrameId = 0
        self.folder = folder
        self.basename = basename
        return
        
    def read(self):
        frame = cv2.imread(os.path.join(self.folder, self.basename.format(self.currentFrameId)), cv2.IMREAD_GRAYSCALE)
        self.currentFrameId += 1
        return True, frame
        
class FakeCamera(Camera):
    
    def __init__(self, folder="./frames", basename="image{}.png", fisheye=False, calibrationFile="./cameraCalibration_cv2.npz"):
        self._cap = FakeCapture(folder, basename)
        _, frame = self._cap.read()
        self._frameHeight, self._frameWidth = frame.shape
        self._cap.currentFrameId = 0
        self._fisheye = fisheye
        self.calibrationFile = calibrationFile
        return
        
    def release(self):
        return
        
    @property
    def fisheye(self):
        return self._fisheye
        
    @property
    def calibration(self):
        return np.load(self.calibrationFile, )
        
    def readCapture(self):
        return self._cap.read()
        
    @property
    def resolution(self):
        return (self._frameHeight, self._frameWidth)
    
class CalibrationManager(Borg):

    _shared_state = {}

    def __init__(self):
        Borg.__init__(self)
        return
        
    def initialize(self):
        self.camera = None
        self.windowName = None
        self.displayResolution = None
        self.helpers = CalibrationHelpers()
        return
        
    def createFullscreenWindow(self, offsetX=1920, offsetY=0, name="Viewport", destroyPrev=True):
        if(destroyPrev is True and self.windowName is not None):
            cv2.destroyWindow(self.windowName)
        self.windowName = name
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.windowName, offsetX, offsetY)
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return
        
    def setActiveCamera(self, camera):
        if(self.camera is not None):
            self.camera.release()
        self.camera = camera
        return
        
    def setDisplayResolution(self, pVertical, pHorizontal):
        self.displayResolution = (pVertical, pHorizontal)
        return
        
    def createMonitorMaskRoutine(self, threshold=53, displayTimeMs=200):
        aw = self.helpers.allWhite(*self.displayResolution)
        ad = self.helpers.allDark(*self.displayResolution)
        cv2.imshow(self.windowName, aw)
        wait(displayTimeMs)
        frame = self.camera.readNewFrame(undistort=True)
        cv2.imshow(self.windowName, ad)
        wait(displayTimeMs)
        darkFrame = self.camera.readNewFrame(undistort=True)
        return self.createMask(frame, darkFrame, threshold)
        
    def erodeMask(self, mask, ksize=(10,10)):
        maskcopy = mask.copy()
        for img in np.hsplit(maskcopy, 2):
            cv2.rectangle(img, (0,0), img.shape[::-1], (0, 0, 0), 1)
        kernel = np.ones(ksize, np.uint8)
        return cv2.erode(maskcopy, kernel)
        
    def measureBitsRoutine(self, bits, mask, invert=False, brightness=127, threshold=1, displayTimeMs=250, colorOverride = None):
        displayedBuffer = bits[:, :, 0] * brightness
        darkFrameBuffer = None
        measuredBits = np.ones((self.camera.resolution) + (8, ), dtype=np.uint8)
        lastResult = np.full(self.camera.resolution, invert, dtype=np.uint8)
        for i in range(15):
            bitIndex = (i + 1) // 2
            cv2.imshow(self.windowName, displayedBuffer if colorOverride is None else cv2.merge((displayedBuffer * colorOverride[0], displayedBuffer * colorOverride[1], displayedBuffer * colorOverride[2])))
            wait(displayTimeMs)
            frame = self.camera.readNewFrame(undistort=True)
            if i % 2 is 0:
                darkFrameBuffer = frame.copy()
                displayedBuffer = (1 - bits[:, :, bitIndex]) * brightness
            else:
                bitmask = self.createMask(frame, darkFrameBuffer, 1)
                lastResult = bitmask == lastResult # xor with last bitmask - Grey -> binary
                measuredBits[:, :, bitIndex - 1] = lastResult
                displayedBuffer = bits[:, :, bitIndex] * brightness
        return np.packbits(measuredBits, axis=-1)[:, :, 0] * mask
        
    def createMask(self, frame, darkFrame, threshold):
        mask = cv2.threshold(cv2.subtract(frame, darkFrame), thresh=threshold, maxval=1, type=cv2.THRESH_BINARY)[1]
        return mask
        
    def polyMask(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        maxContour = 0
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContour:
                maxContour     = contourSize
                maxContourData = contour
        # Create a mask from the largest contour
        mask = np.zeros_like(mask)
        mask = cv2.fillPoly(mask, [maxContourData], 1)
        return mask
        
    def calibrateGreycodes(self, widthData, heightData, calibration=None):
        rawData    = np.zeros((widthData.shape[0], widthData.shape[1], 3), dtype=np.uint8)
        rawData[..., 2] = widthData; rawData[..., 1] = heightData
        leftData   = rawData [:, : int(rawData.shape[1] / 2)  ]
        rightData  = rawData [:,   int(rawData.shape[1] / 2) :]

        if(calibration is None):
            calibration = self.camera.calibration
        leftCoeffs = calcCoeffs(leftData , calibration['leftCameraMatrix' ], np.zeros(4), calibration['R1'])
        rightCoeffs = calcCoeffs(rightData, calibration['rightCameraMatrix'], np.zeros(4), calibration['R2'])
        print(f"DDDDDD: {leftCoeffs[0]}")
        return {
            'left_uv_to_rect_x' : leftCoeffs[0].flatten().tolist(),  'left_uv_to_rect_y': leftCoeffs[1].flatten().tolist(),
            'right_uv_to_rect_x': rightCoeffs[0].flatten().tolist(), 'right_uv_to_rect_y': rightCoeffs[1].flatten().tolist()
        }

if __name__ == '__main__':
    """OFFLINE MODE - from files, no devices
    import sys
    cm = CalibrationManager()
    mbw = cv2.imread("./WidthCalibration.png", cv2.IMREAD_GRAYSCALE)
    mbh = cv2.imread("./HeightCalibration.png", cv2.IMREAD_GRAYSCALE)
    cal = cm.calibrateGreycodes(mbw, mbh, np.load("./cameraCalibration_cv2.npz", ), False)
    print(cal)
    print(CalibrationHelpers.calibration2GLSL(cal))
    sys.exit(0)
    """
    
    """
    ch = CalibrationHelpers()
    cm = CalibrationManager()
    cm.setDisplayResolution(1920, 1080)
    #cm.createFullscreenWindow(-1920, 0)
    
    camera = CV2Camera(0, 1080, 3840)
    import sys
    frame = camera.readNewFrame(undistort=True)
    cv2.imwrite("./dst2.png", frame)
    camera.release()
    sys.exit(0)
    """
    
    ch = CalibrationHelpers()
    cm = CalibrationManager()
    cm.setDisplayResolution(1920, 1080)
    splitscreen = sys.argv[1] == "1"
    cm.createFullscreenWindow(0, 1080)
    cv2.imshow(cm.windowName, cv2.imread("board.jpg"))
    wait(200)
    #camera = CV2Camera(0, 1080, 3840)

    #fisheye = False
    camera = T265Camera()
    #camera = FakeCamera(calibrationFile="./cameraCalibration_rs.npz", fisheye=True)
    #camera.readNewFrame()
    #fisheye = True #move to camera properties? need it also for offline (no camera)
    cm.setActiveCamera(camera)
    mask = cm.createMonitorMaskRoutine(100)
    erodedMask = cm.erodeMask(mask)
    erodedMaskWand = Image.from_array(erodedMask*255)
    cv2.imshow("res", mask * 100) #binary mask
    wait(200)
    widthBits = ch.widthBits(*cm.displayResolution, splitscreen)
    mbw = cm.measureBitsRoutine(widthBits, mask)
    cv2.imshow("res", cv2.applyColorMap(mbw, cv2.COLORMAP_JET))
    cv2.imwrite("./WidthCalibration.png", mbw)
    
    mbw = (mbw*257).astype(np.uint16)
    with Image.from_array(mbw) as img:
        img.selective_blur(radius=0, sigma=7, threshold=0.05 * img.quantum_range)
        img.composite_channel("gray", erodedMaskWand, "multiply", 0, 0)
        img.save(filename="./WidthCalibration_blur.png")
    """ SUBPIXEL TEST
    widthBits = ch.widthBits(*cm.displayResolution)
    mbg = cm.measureBitsRoutine(widthBits, mask, brightness=127, threshold=1, displayTimeMs=200, colorOverride = (0,1,0))
    cv2.imshow("res", cv2.applyColorMap(mbg, cv2.COLORMAP_JET))
    #wait(0)
    mbb = cm.measureBitsRoutine(widthBits, mask, brightness=127, threshold=1, displayTimeMs=200, colorOverride = (1,0,0))
    cv2.imshow("res", cv2.applyColorMap(mbb, cv2.COLORMAP_JET))
    #wait(0)
    mbr = cm.measureBitsRoutine(widthBits, mask, brightness=127, threshold=1, displayTimeMs=200, colorOverride = (0,0,1))
    cv2.imshow("res", cv2.applyColorMap(mbr, cv2.COLORMAP_JET))
    #wait(0)
    merged = cv2.merge((mbb, mbg, mbr))
    cv2.imshow("res", merged)
    wait(0)
    """
    heightBits = ch.heightBits(*cm.displayResolution)
    mbh = cm.measureBitsRoutine(heightBits, mask, True)
    cv2.imshow("res", cv2.applyColorMap(mbh, cv2.COLORMAP_JET))
    cv2.imwrite("./HeightCalibration.png", mbh)
    camera.release()
    
    #undistort one time
    #cv2.imwrite(f"./frames/image{camera._cap.currentFrameId}.png", mbw)
    #frame = camera.readNewFrame(undistort=True)
    #cv2.imwrite("./frames/test1.png", frame)
    #mbw = frame
    #cv2.imwrite(f"./frames/image{camera._cap.currentFrameId}.png", mbh)
    #frame = camera.readNewFrame(undistort=True)
    #cv2.imwrite("./frames/test2.png", frame)
    #mbh = frame
    mbh = (mbh*257).astype(np.uint16)
    with Image.from_array(mbh) as img:
        img.selective_blur(radius=0, sigma=7, threshold=0.05 * img.quantum_range)
        img.composite_channel("gray", erodedMaskWand, "multiply", 0, 0)
        img.save(filename="./HeightCalibration_blur.png")
    
    cv2.waitKey(2000)
    cal = cm.calibrateGreycodes(mbw, mbh)
    print(cal)
    #print(CalibrationHelpers.calibration2GLSL(cal))