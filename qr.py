import numpy as np
import cv2
from agentspace import Agent, space
import time

class QrAgent(Agent):

    def __init__(self,imageName,qrName):
        self.imageName = imageName
        self.qrName = qrName
        super().__init__()

    def init(self):
        self.decoder = cv2.wechat_qrcode_WeChatQRCode("qr/detect.prototxt", "qr/detect.caffemodel", "qr/sr.prototxt", "qr/sr.caffemodel")
        self.attach_timer(1.0)
        
    def senseSelectAct(self):
        rgb = space[self.imageName]
        if not rgb is None:
            qr, points = self.decoder.detectAndDecode(rgb)
            qr = None if len(qr) == 0 else qr[0]
            if qr is not None:
                print(qr) # 'geo:48.8016394,16.8011145'
                values = qr[4:].split(',')
                goal = (float(values[0]),float(values[1]))
                space[self.qrName] = goal # (48.8016394,16.8011145)
                time.sleep(3)
