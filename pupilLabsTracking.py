import cv2
import numpy as np
from pupil_detectors import Detector2D
from pye3d.camera import CameraModel
from pye3d.detector_3d import Detector3D
import time
import uvc
import zmq
import msgpack
from threading import Thread

def portService(subPort):
    ctx = zmq.Context()
    s = ctx.socket(zmq.REP)
    s.bind("tcp://127.0.0.1:50020")
    while True:
        message = s.recv_multipart()
        if message[0] in [b'SUB_PORT', b'PUB_PORT']:
            s.send(subPort.encode('utf-8'))
        elif message[0] == b'v':
            s.send("pye3d 0.0.6".encode('utf-8'))
        elif message[0] == b't':
            s.send(str(time.time()).encode('utf-8'))
        else: #ignore some notifications
            s.send_multipart([b'ok'])

camIds = [3, 4]
reqConf = (400,400,120)

#start port announce service
subPort = "5021"
t = Thread(target = portService, args = (subPort, ), daemon = True)
t.start()

ctx = zmq.Context()
s = ctx.socket(zmq.PUB)
s.bind(f"tcp://127.0.0.1:{subPort}")

dev_list = uvc.device_list()
captures = [uvc.Capture(dev_list[camId]["uid"]) for camId in camIds]

for cap in captures:
    cap.frame_mode = reqConf
    for cont in cap.controls:
        if cont.display_name == "Absolute Exposure Time":
            cont.value = 30

calibrations = [np.load("./cameraCalibrationET_right.npz"), np.load("./cameraCalibrationET_left.npz")]
cameras = [
    CameraModel(
        calibrations[ic]["cameraMatrix"][0,0] + calibrations[ic]["cameraMatrix"][1,1] / 2,
        (reqConf[0], reqConf[1])
    ) for ic in range(len(calibrations))
]
detectors2d = [Detector2D() for _ in range(len(camIds))]
detectors3d = [Detector3D(cameras[ic]) for ic in range(len(camIds))]
maps = [cv2.initUndistortRectifyMap(c["cameraMatrix"], c["distCoeffs"], None, c["cameraMatrix"], reqConf[:2], cv2.CV_32FC1) for c in calibrations]

while(True):
    waitedKey = cv2.waitKey(1) & 0xFF
    if(waitedKey == ord('q')):
        break
    elif(waitedKey == ord('s')):
        captures.reverse()
        for d in detectors3d:
            d.reset()
    elif(waitedKey == ord('r')):
        for d in detectors3d:
            d.reset()
    for ic, cap in enumerate(captures):
        frame = cap.get_frame_robust().img
        frame = cv2.remap(frame, maps[ic][0], maps[ic][1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result2d = detectors2d[ic].detect(gray)
        ellipse = result2d['ellipse']
        frame = cv2.ellipse(frame, [int(e) for e in ellipse['center']], [int(e/2) for e in ellipse['axes']], ellipse['angle'], 0, 360, (255,0,0), 2)
        result2d["timestamp"] = time.time() * 1000
        result3d = detectors3d[ic].update_and_detect(result2d, gray)
        ellipse = result3d['projected_sphere']
        if ellipse["axes"][0] > 0 and ellipse["axes"][1] > 0:
            frame = cv2.ellipse(frame, [int(e) for e in ellipse['center']], [int(e/2) for e in ellipse['axes']], ellipse['angle'], 0, 360, (255,0,0), 2)
            cv2.imshow(str(ic), frame)
        resultPupil = {
           "id":ic,
           "topic":f"pupil.{ic}.3d",
           "method":"pye3d 0.0.6 real-time",
           "norm_pos":[
              0.0,
              0.0
           ],
           "diameter":0.0,
           "confidence":0.0,
           "timestamp":0.0,
           "sphere":{
              "center":[
                 0.0,
                 0.0,
                 0.0
              ],
              "radius":0
           },
           "projected_sphere":{
              "center":[
                 0.0,
                 0.0
              ],
              "axes":[
                 0.0,
                 0.0
              ],
              "angle":0.0
           },
           "circle_3d":{
              "center":[
                 0.0,
                 0.0,
                 0.0
              ],
              "normal":[
                 0.0,
                 0.0,
                 0.0
              ],
              "radius":0.0
           },
           "diameter_3d":0.0,
           "ellipse":{
              "center":[
                 0.0,
                 0.0
              ],
              "axes":[
                 0.0,
                 0.0
              ],
              "angle":0.0
           },
           "location":[
              0.0,
              0.0
           ],
           "model_confidence":0.0,
           "theta":0.0,
           "phi":0.0
        }
        for key in result3d:
            if key in resultPupil:
                resultPupil[key] = result3d[key]
        s.send_multipart(["pupil.".encode('utf-8'), msgpack.packb(resultPupil, use_bin_type=True)])

print("Waiting for message queues to flush...")
time.sleep(0.5)
print("Done.")

