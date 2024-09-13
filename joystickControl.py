from agentspace import Agent,space
import time

class JoystickControlAgent(Agent):

    def __init__(self,name,nameForward,nameTurn):
        self.name = name
        self.nameForward = nameForward
        self.nameTurn = nameTurn
        super().__init__()

    def init(self):
        self.heading = 0.0
        self.active = False
        space.attach_trigger(self.name,self)
        
    def valueOf(self,value):
        if value < -25000:
            return -1
        elif value > 25000:
            return 1
        else:
            return 0

    def senseSelectAct(self):
        axisXYZ = space(default=[[],[0,0,0],[]])[self.name][1]
        turn = self.valueOf(axisXYZ[0])
        forward = -self.valueOf(axisXYZ[1])
        if forward != 0 or turn != 0 or self.active:
            self.active = True
            self.heading += 5*turn
            self.heading = min(max(self.heading,-50),50)
            if turn == 0:
                self.heading = 0
            #print('joystick',axisXYZ[0],axisXYZ[1],'turn',turn,'forward',forward,'heading',self.heading)
            space(validity=1.0,priority=4)[self.nameForward] = forward
            space(validity=1.0,priority=4)[self.nameTurn] = self.heading
            if forward == 0 and turn == 0:
                self.active = False
