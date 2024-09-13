import time
import serial
import re
from agentspace import Agent, space
from dgmath import dgnorm

class GpsReceiver:
    
    def __init__(self,line='COM10'):
        '''Creates an object that you can call to control the robot
        '''
        self.ser = serial.Serial(
            port=line,
            baudrate=9600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=0 #block
        )
        if not self.ser.isOpen():
            print('cannot open line '+line)
        else:
            self.buffer = ''

    def receive(self):
        if self.ser.isOpen():
            while self.buffer.find('\n')==-1:
                self.buffer += self.ser.read(1024).decode()
            result = re.sub('[\r\n].*','',self.buffer)
            self.buffer = self.buffer[self.buffer.find('\n')+1:]
        else:
            result = ''
        #print(result)
        return result
        
    def recalc(self,bulg):
        dg = int(bulg/100)
        min = bulg-100*dg
        return dg + min/60

    def getPosition(self):
        cog = None
        while True:
            line = self.receive()
            #print(line)
            if line.find('$GPGLL') != -1 or line.find('$GNGLL') != -1:
                if ',,' in line:
                    return [-1,-1,None]
                else:
                    values = re.findall(r'[\d\.]+(?:,[\d\.]+)?',line)
                    return [self.recalc(float(v)) for v in values[:2]] + [ cog ]
                    # azimuth is 0=north, 90=east, 180=south, -90=west
            elif line.find('$GNRMC') != -1:
                    fields = line.split(',')
                    try:
                        cog = dgnorm(float(fields[8]))
                    except ValueError:
                        pass
                        

class GpsAgent(Agent):

    def __init__(self,gpsName,line='COM11'):
        self.line = line
        self.gpsName = gpsName
        super().__init__()

    def init(self):
        gps = GpsReceiver(self.line)
        while True:
            position = gps.getPosition()
            if position[0] > 0 and position[1] > 0:
                space(validity=1.5)[self.gpsName+'-azimuth'] = position[2]
                space(validity=1.5)[self.gpsName] = position[:2]
                with open('logs/gps.txt','at') as f:
                    timestamp = time.time()
                    f.write(f'{timestamp},{position[0]},{position[1]},{position[2]}\n')
        
    def senseSelectAct(self):
        pass

# Test    
if __name__ == "__main__":
    gps = GpsReceiver(line='COM3') 
    while True:
        pos = gps.getPosition()
        print(f"{pos[0]:1.6f},{pos[1]:1.6f}  azimut: {pos[2]}dg")