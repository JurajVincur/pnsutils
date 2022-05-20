from polyHelpers import polyval2dExpanded
import cv2
import json
import numpy as np

class LookupTable:

    def __init__(self, resolution=(180, 200)):
        self.lut = np.zeros((2, resolution[1], resolution[0], 3), dtype=np.uint16)
        self.projectionMatrix = None
        return
        
    def viewToUV(self, x, y, z=-1.0):
        rx = self.projectionMatrix[0] * x + self.projectionMatrix[1] * y + self.projectionMatrix[2] * z + self.projectionMatrix[3]
        ry = self.projectionMatrix[4] * x + self.projectionMatrix[5] * y + self.projectionMatrix[6] * z + self.projectionMatrix[7]
        rz = self.projectionMatrix[8] * x + self.projectionMatrix[9] * y + self.projectionMatrix[10] * z + self.projectionMatrix[11]
        w = self.projectionMatrix[12] * x + self.projectionMatrix[13] * y + self.projectionMatrix[14] * z + self.projectionMatrix[15]
        rx /= w
        ry /= w
        rx = (rx * 0.5 + 0.5)
        ry = (ry * 0.5 + 0.5)
        return np.array((rx, ry))
        
    def loadCameraProperties(self, path):
        data = None
        with open(path, "r") as f:
            data = json.load(f)
        self.projectionMatrix = data["projectionMatrix"]
        return
        
    def loadV2Calibration(self, path):
        data = None
        with open(path, "r") as f:
            data = json.load(f)
        _, height, width, _ = self.lut.shape
        for x in range(width):
            for y in range(height):
                for i, side in enumerate(("left", "right")):
                    u = x / (width - 1)
                    v = (height - y - 1) / (height - 1)
                    rx = polyval2dExpanded(u, v, np.array(data[f"{side}_uv_to_rect_x"]))
                    ry = -polyval2dExpanded(u, v, np.array(data[f"{side}_uv_to_rect_y"]))
                    rg = self.viewToUV(rx, ry) * 65535
                    self.lut[i, y, x, 2] = rg[0] #BGR
                    self.lut[i, y, x, 1] = rg[1]
        return
        
    def save(self):
        return
        
if __name__=="__main__":
    lut = LookupTable()
    lut.loadCameraProperties(r"sampleData\CameraProperties.json")
    lut.loadV2Calibration(r"sampleData\V2Out.json")
    cv2.imshow('lut_left', lut.lut[0])
    cv2.imshow('lut_right', lut.lut[1])
    stacked = np.hstack(lut.lut)
    cv2.imshow('lut_stacked', stacked)
    cv2.imwrite('lut_left.png', lut.lut[0])
    cv2.imwrite('lut_right.png', lut.lut[1])
    cv2.imwrite('lut.png', stacked)
    cv2.waitKey(0)
    