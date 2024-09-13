import time
from agentspace import Agent, space, Trigger
import cv2
import numpy as np

model = 'segformer' # 'glee'
if model == 'glee':
    from run_glee import text2mask 
else:
    from run_segformer import text2mask 

def heading(binary_mask):
    binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.rectangle(binary_mask,(0,binary_mask.shape[0]-15,binary_mask.shape[1]-1,binary_mask.shape[0]-1),255,cv2.FILLED)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, (7,7))
    
    binary_mask_flipped = cv2.flip(binary_mask,0)
    data = np.argmin(binary_mask_flipped, axis=0)
    
    midy = binary_mask.shape[0]-1
    mid = binary_mask.shape[1]//2
    y = np.concatenate([np.minimum.accumulate(data[:mid][::-1])[::-1],np.minimum.accumulate(data[mid:])])
    y[y<=50]=0
    x = np.arange(binary_mask.shape[1]) - mid
    threshold = 160
    li = mid
    while li > 0 and y[li] > threshold:
        li -= 1
    ri = mid
    while ri-1 < binary_mask.shape[1]-1 and y[ri] > threshold:
        ri += 1
    #import matplotlib.pyplot as plt
    #plt.plot(x,y)
    #plt.plot([x[li],x[li],x[ri],x[ri]],[0,threshold,threshold,0])
    #plt.show()
    A = np.trapz(y, x)
    Mx = np.trapz(x * y, x)
    My = np.trapz(y**2 / 2, x)
    if A != 0:
        cx = Mx / A
        cy = My / A
        obstacle = False
    else:
        cx, cy = 0, 0
        obstacle = True
    
    result = cv2.cvtColor(binary_mask,cv2.COLOR_GRAY2BGR)
    if obstacle:
        cv2.putText(result,"!!!!!",(mid-60,midy-20),0,1.5,(0,0,255),2)
    else:
        cv2.arrowedLine(result,(mid,midy),(mid+int(cx),midy-int(cy)),(0,0,255),2)
        cv2.arrowedLine(result,(mid,midy),(mid+int(x[li]),midy-int(threshold)),(0,255,0),1)
        cv2.arrowedLine(result,(mid,midy),(mid+int(x[ri]),midy-int(threshold)),(0,255,0),1)
        dg = space['turn']
        if dg:
            xact, yact = np.sin(np.radians(dg))*320, np.cos(np.radians(dg))*240
            cv2.arrowedLine(result,(mid,midy),(mid+int(xact),midy-int(yact)),(255,0,255),3)
    goal = space['goal']
    if goal is not None:
        cv2.arrowedLine(result,(mid,midy),(mid+int(80*np.sin(np.radians(goal))),midy-int(80*np.cos(np.radians(goal)))),(255,0,255),3)    

    degrees = [ np.degrees(np.arctan2(cx,cy)), np.degrees(np.arctan2(x[li],threshold)), np.degrees(np.arctan2(x[ri],threshold)) ] 
    return degrees, obstacle, result

class HeadingAgent(Agent):

    def __init__(self,rgbName,headingNane,obstacleName):
        self.rgbName = rgbName
        self.headingNane = headingNane
        self.obstacleName = obstacleName
        self.t0 = int(time.time())
        self.n = 0
        self.fps = 1
        super().__init__()

    def init(self):
        space.attach_trigger(self.rgbName,self)
        
    def senseSelectAct(self):
        frame = space[self.rgbName]
        if frame is None:
            return
        
        clock0 = time.time()
        
        self.t1 = int(clock0)
        if self.t1 == self.t0:
            self.n += 1
        else:
            self.fps = self.n
            self.n = 1
            
        if self.t1-self.t0 >= 2:
            self.t0 = self.t1
            cv2.imwrite(f'logs/{self.t1}.png',frame)
        
        mask, visual = text2mask(frame)
        
        angles, obstacle, result = heading(mask)
        angle, anglefrom, angleto = angles
        clock1 = time.time()
        validity = 2.5*(clock1-clock0) # (model's fps depend on powering)
        
        angle = min(max(angle,-50),50)
        space(validity=validity)[self.headingNane] = [angle, anglefrom, angleto]
        
        space(validity=validity)[self.obstacleName] = obstacle

        disp = cv2.flip(cv2.vconcat([visual,result]),1)
        cv2.putText(disp,f'{self.fps} {clock1-clock0:.3f}s',(10,25),0,1,(0,255,255),1)
        cv2.imshow("detection",disp)
        cv2.waitKey(1)

if __name__ == '__main__':
    video = True
    if video:
        from camera import CameraAgent
        camera_agent = CameraAgent('rgb',0)
        heading_agent = HeadingAgent('rgb','heading','obstacle')
        space['turn'] = -30.0
        input()
        heading_agent.stop()
        camera_agent.stop()
        time.sleep(2)
        cv2.destroyAllWindows()
    else:
        path = '../images/'
        name = path + '1722614048' # '1111111111' # '1722614037'
        binary_mask = cv2.imread(name+'-mask.png',cv2.IMREAD_GRAYSCALE)
        space['goal'] = -20.0
        angles, obstacle, result = heading(binary_mask)
        cv2.imwrite(name+'-heading.png',result)
        if obstacle:
            print("STOP!")
        else:
            print('angles:',angles[0],',',angles[1],'-',angles[2])
        
