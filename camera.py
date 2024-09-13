import numpy as np
import cv2 as cv
from agentspace import Agent, space

class CameraAgent(Agent):

    def __init__(self, nameImage, id=0):
        self.nameImage = nameImage
        self.id = id
        super().__init__()
        
    def init(self):
        self.camera = cv.VideoCapture(self.id,cv.CAP_DSHOW)
        while not self.stopped:
            # Grab a frame
            ret, img = self.camera.read()
            if not ret:
                #self.stop()
                #return
                continue
            
            # sample it onto blackboard
            space(validity=0.15)[self.nameImage] = img
            
        self.camera.release()
 
if __name__ == "__main__":
    import time

    camera_agent = CameraAgent('bgr',0)

    class ViewerAgent(Agent):
    
        def init(self):
            space.attach_trigger('bgr',self)
            self.t0 = int(time.time())
            self.fs = 0
            self.fps = 0
            
        def senseSelectAct(self):
            frame = space["bgr"]
            if frame is None:
                self.stopped = True
                return
                
            self.fs += 1
            self.t1 = int(time.time())
            if self.t1 > self.t0:
                self.fps = self.fs / (self.t1-self.t0)
                self.fs = 0
                self.t0 = self.t1

            result = np.copy(frame)
            cv.putText(result, f"{self.fps:1.0f}", (8,25), 0, 1.0, (0, 255, 0), 2)

            cv.imshow('right eye',result)
            key = cv.waitKey(1) & 0xff
            if key == 27:
                self.stopped = True
                return
    
    viewer_agent = ViewerAgent()
    time.sleep(20)
    viewer_agent.stop()
    cv.destroyAllWindows()
    camera_agent.stop()
   