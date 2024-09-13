import os

from agentspace import space
from gpsReceiver import GpsAgent
from camera import CameraAgent
from qr import QrAgent
from heading import HeadingAgent
from osm import OsmAgent
from control import ControlAgent
from motorControler import MotorAgent
from joystickApi import JoystickAgent
from joystickControl import JoystickControlAgent

import signal
import time

import warnings
warnings.filterwarnings("ignore")

def quit():
    try:
        motors.stop()
    except:
        pass
    os._exit(0)
    
# exit on ctrl-c
def signal_handler(signal, frame):
    quit()
    
signal.signal(signal.SIGINT, signal_handler)

GpsAgent('gps',line='COM3')
time.sleep(1)

motors = MotorAgent('forward','turn',line='COM4')
time.sleep(1)
JoystickAgent('joystick')
time.sleep(1)
JoystickControlAgent('joystick','forward','turn')
time.sleep(1)

CameraAgent('rgb',1)
time.sleep(1)

QrAgent('rgb','qr')
time.sleep(1)

OsmAgent('gps','qr','goal')
time.sleep(1)

HeadingAgent('rgb','heading','obstacle')
time.sleep(1)
ControlAgent('heading','obstacle','goal','forward','turn')
time.sleep(1)

time.sleep(4)
space['qr'] = (48.5934894,17.8365257) # fontanka
