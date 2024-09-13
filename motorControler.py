import time
import serial # pyserial
from agentspace import Agent, space, Trigger
import cv2

class MotorControl:
    
    def __init__(self,line='COM12'):
        '''Creates an object that you can call to control the robot
        '''
        self.ser = serial.Serial(
            port=line,
            baudrate=9600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1 #nonblock
        )
        self.speed = 65 # minimal speed
        if not self.ser.isOpen():
            print('cannot open line '+line)
        else:
            print(line+' opened')
            self.last_forward = 0
            self.last_turn = 0
            self.reset()
            self.command(0.0,0.0)
            time.sleep(0.5)

    def command(self, forward, turn):
        if self.ser.isOpen():
            forward = int(forward)
            turn = int(turn)
            if turn != self.last_turn:
                self.ser.write(bytes(f'm0 {-turn} 255\r', encoding='ascii'))
                self.last_turn = turn
            if forward != self.last_forward:
                self.ser.write(bytes(f'm1 {forward} {self.speed}\r', encoding='ascii'))
                self.last_forward = forward

    def reset(self):
        if self.ser.isOpen():
            self.ser.write(bytes(f'm2\r', encoding='ascii'))

    def left(self):
        self.command(1.0,-30.0)

    def right(self):
        self.command(1.0,30.0)
        
    def forward(self):
        self.command(1.0,0.0)
        
    def backward(self):
        self.command(-1.0,0.0)
        
    def stop(self):
        self.command(0.0,self.last_turn)
        
    def setSpeed(self, speed):
        self.speed = speed

class MotorAgent(Agent):

    def __init__(self,forwardName,turnName,line='COM12'):
        self.line = line
        self.forwardName = forwardName
        self.turnName = turnName
        super().__init__()

    def init(self):
        self.control = MotorControl(self.line)
        space.attach_trigger(self.forwardName,self)
        space.attach_trigger(self.turnName,self)
        
    def senseSelectAct(self):
        speed = space(default=100)['speed'] # 100 is optimal speed
        self.control.setSpeed(speed)
        forward = space(default=0)[self.forwardName]
        turn = space(default=0)[self.turnName]
        #print('turn',turn,'forward',forward)
        self.control.command(forward, turn)
        time.sleep(0.1)
        
    def stop(self):
        self.control.command(0.0,0.0)
        time.sleep(0.4)
        self.stopped = True

# Test    
if __name__ == "__main__":
    robot = MotorControl('COM4')
    print('right')
    robot.right()
    time.sleep(2)
    print('left')
    robot.left()
    time.sleep(2)
    print('forward')
    robot.forward()
    time.sleep(2)
    print('backward')
    robot.backward()
    time.sleep(2)
    print('stop')
    robot.stop()
    #agent = MotorAgent('forward','turn',line='COM12')
    #space['turn']  = 1
    #space['forward']  = 1
    #time.sleep(2)
    #agent.stop()


"""
robot.forward()
time.sleep(2)
robot.stop()
"""

"""
agent = MotorAgent('forward','turn',line='COM12')
space['turn']  = 1
space['forward']  = 1
agent.stop()
"""
