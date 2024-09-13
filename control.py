import time
from agentspace import Agent, space
import cv2
import numpy as np
from dgmath import dgin, dgdiff, dgaverage

class ControlAgent(Agent):

    def __init__(self,headingName,obstacleName,goalName,forwardName,turnName):
        self.headingName = headingName
        self.obstacleName = obstacleName
        self.goalName = goalName
        self.forwardName = forwardName
        self.turnName = turnName
        self.freeToMove = time.time()
        super().__init__()

    def init(self):
        self.attach_timer(0.25)
        
    def senseSelectAct(self):
        
        obstacle = space(default=True)[self.obstacleName]
        heading = space(default=[0.0,0.0,0.0])[self.headingName] # what we see: [angle, anglefrom, angleto]
        goal = space[self.goalName] # how we navigate
        
        if goal is None or obstacle: 
            space[self.forwardName] = 0
            space[self.turnName] = 0
            
            if goal is not None:
                if time.time() - self.freeToMove > 3.0:
                    space(validity=1.5,priority=5)[self.forwardName] = -1.0
                    space(validity=1.5,priority=5)[self.turnName] = 0.0
            
            return
        
        # combine goal and heading
        if dgin(goal,heading[1],heading[2]):
            action = goal
        elif abs(dgdiff(goal,heading[1])) < abs(dgdiff(goal,heading[2])):
            action = heading[1]
        else:
            action = heading[2]
            
        action = dgaverage([action,heading[0]]) #?
        
        space(validity=1.5)[self.forwardName] = 1.0
        space(validity=1.5)[self.turnName] = action
        self.freeToMove = time.time()
        