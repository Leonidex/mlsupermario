import pygame
import numpy as np
import tensorflow.compat.v1 as tf
from classes.Dashboard import Dashboard
from classes.Level import Level
from classes.Menu import Menu
from classes.Sound import Sound
from entities.Mario import Mario
from ml.MLPlayer import MLPlayer
import os

class MainWindow:

    def __init__(self):
        self.techniqueIdentifier = 0
        self.maxLearningRate = 0.2   # learning rate
        minimalLearningRate = 0.01
        self.param1 = 0.9
        self.param2 = 0.99

        self.gamma = 0.97

        self.experimentsCount = 100

        self.windowSize = (640,480)
        self.mario = None
        
        self.episodesPerStep = 10
        self.globalStepMax = int(self.experimentsCount/self.episodesPerStep)
        self.globalStep = 1  #(self.maxLearningRate-minimalLearningRate)/self.globalStepMax #tf.Variable(0, trainable=False)
        self.decayRate = 0.95
        self.learningRate = tf.train.exponential_decay(self.maxLearningRate, self.globalStep, 1, self.decayRate, staircase=True)

        self.player = MLPlayer(self.techniqueIdentifier, self.learningRate, self.globalStep, self.param1, self.param2)
        # for i in range(0, agentsCount):
        #     self.player.addAgents(1, self.techniqueIdentifier, self.learningRate, self.param1, self.param2)
        #     self.learningRate -= (self.maxLearningRate-minimalLearningRate)/agentsCount

        self.totalrewards = np.empty(self.experimentsCount)
        self.running_avg = np.empty(self.experimentsCount)

        self.maxFrames = 1200
    
        #(lambda: os.system('cls'))()    # Clear console

    def decreaseLearningRate(self):
        self.globalStep += 1
    def calculateLearningRate(self):
        return self.maxLearningRate * (self.decayRate ** (self.globalStep))
    def printLearningRate(self):
        print("Learning rate is:{0}".format(self.calculateLearningRate()))

    def main(self):
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        screen = pygame.display.set_mode(self.windowSize)
        max_frame_rate = 9999
        dashboard = Dashboard("./img/font.png", 8, screen)
        sound = Sound()
        
        maxReward = 0
        timesMaxedTheReward = 0

        # maxvelX = 0
        # minvelX = 0
        # maxvelY = 0
        # minvelY = 0
        for episodeCounter in range(0, self.experimentsCount):
            print("=============================================")
            print("============WELCOME TO EPISODE {0}============".format(episodeCounter))
            print("=============================================")
            framesCounter = 0

            if episodeCounter/self.experimentsCount >= self.globalStep/self.globalStepMax:
                self.decreaseLearningRate()
                print("Learning rate is:{0}".format(self.calculateLearningRate()))
            
            self.level = Level(screen, sound, dashboard)
            
            self.level.loadLevel('Level1-2')
            
            self.mario = Mario(0, 0, self.level, screen, dashboard, sound)
            self.mario.setPos(0,384)
            clock = pygame.time.Clock()

            max_x = 0
            obs = self.mario.getObservation()

            eps = 1.0 / np.sqrt(episodeCounter + 1)
            
            while not self.mario.restart and framesCounter <= self.maxFrames:
                # if maxvelX < self.mario.vel.x:
                #     maxvelX = self.mario.vel.x
                #     print("vel_x max: ", self.mario.vel.x)
                # elif minvelX > self.mario.vel.x:
                #     minvelX = self.mario.vel.x
                #     print("vel_x min: ", self.mario.vel.x)
                # if maxvelY < self.mario.vel.y:
                #     maxvelY = self.mario.vel.y
                #     print("vel_y max: ", self.mario.vel.y)
                # elif minvelY > self.mario.vel.y:
                #     minvelY = self.mario.vel.y
                #     print("vel_y min: ", self.mario.vel.y)

                #print(framesCounter)
                framesCounter += 1
                pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
                if self.mario.pause:
                    self.mario.pauseObj.update()
                else:
                    self.level.drawLevel(self.mario.camera)
                    dashboard.update()
                    self.mario.update()
                    #print(self.mario.rect.x, self.mario.rect.y)
                    prevObs = obs
                    obs = self.mario.getObservation()

                    actionIdentifier = self.player.agent.sample_action(obs, eps)
                    self.doAction(actionIdentifier)

                    if max_x<=self.mario.rect.x:
                        max_x = self.mario.rect.x
                    
                    reward = max_x/framesCounter

                    next = self.player.agent.predict(obs)
                    assert(len(next.shape) == 1)
                    G = reward + self.gamma*np.max(next)
                    self.player.agent.update(prevObs, actionIdentifier, G)

                pygame.display.update()
                clock.tick(max_frame_rate)

            totalreward = max_x
            if totalreward > maxReward:
                maxReward = totalreward
            self.totalrewards[episodeCounter - 1] = totalreward
            self.running_avg[episodeCounter] = self.totalrewards[max(0, episodeCounter - self.episodesPerStep):(episodeCounter + 1)].mean()
            #if episodeCount % 1 == 0:
            print("episode: {0}, totalReward: {1} eps: {2} avg reward (last 100): {3} maxReward {4}".format(episodeCounter, totalreward, eps, float((self.running_avg[episodeCounter]+self.maxFrames)/(self.maxFrames+self.level.max_X)),  maxReward))

            if max_x >= self.level.max_X:
                timesMaxedTheReward += 1
            if timesMaxedTheReward > 0:
                print("I've maxed the reward {0} times!".format(timesMaxedTheReward))
        for reward in self.totalrewards:
            print(reward)
            

    # Controls
    def doAction(self, actionIdentifier):
        if (actionIdentifier == 0):
            # Left
            self.mario.traits['jumpTrait'].jump(False)
            self.mario.traits["goTrait"].direction = -1
        elif (actionIdentifier == 1):
            # Right
            self.mario.traits['jumpTrait'].jump(False)
            self.mario.traits["goTrait"].direction = 1
        elif (actionIdentifier == 2):
            # Stand
            self.mario.traits['jumpTrait'].jump(False)
            self.mario.traits['goTrait'].direction = 0
        elif (actionIdentifier == 3):
            # Jump+Stand
            self.mario.traits['jumpTrait'].jump(True)
            self.mario.traits['goTrait'].direction = 0
        elif (actionIdentifier == 4):
            # Jump+Left
            self.mario.traits['jumpTrait'].jump(True)
            self.mario.traits["goTrait"].direction = -1
        elif (actionIdentifier == 5):
            # Jump+Right
            self.mario.traits['jumpTrait'].jump(True)
            self.mario.traits["goTrait"].direction = 1
        if (actionIdentifier < 5):
            self.mario.traits["goTrait"].boost = 0
        if (actionIdentifier > 5):
            self.mario.traits["goTrait"].boost = 1

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution() 
    window = MainWindow()
    window.main()
