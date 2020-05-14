import pygame
import numpy as np
import tensorflow.compat.v1 as tf
from classes.Dashboard import Dashboard
from classes.Level import Level
from classes.Menu import Menu
from classes.Sound import Sound
from entities.Mario import Mario
from ml.MLPlayer import MLPlayer
import pickle as pckl
import os

class MainWindow:

    def __init__(self):
        # Initializing ML parameters
        self.techniqueIdentifier = 0
        self.startingLearningRate = 0.1
        self.param1 = 0.9
        self.param2 = 0.99
        self.learningPeriod = 300

        self.gamma = 0.97
        self.eps = 1.0

        self.experimentsCount = 20000
        self.maxFrames = 3000

        self.windowSize = (640,480)
        self.mario = None
        
        self.episodesPerStep = 10
        self.globalStepMax = int(self.experimentsCount/self.episodesPerStep)
        self.globalStep = 1
        self.decayRate = 0.99
        self.learningRate = tf.train.exponential_decay(self.startingLearningRate, self.globalStep, 1, self.decayRate, staircase=True)

        self.player = MLPlayer(self.techniqueIdentifier, self.maxFrames, 1888, self.learningRate, self.globalStep, self.param1, self.param2)

        self.totalrewards = np.empty(self.experimentsCount)
        self.running_avg = np.empty(self.experimentsCount)
    
        #(lambda: os.system('cls'))()    # Clear console

    def decreaseLearningRate(self):
        self.globalStep += 1
    def calculateLearningRate(self):
        return self.startingLearningRate * (self.decayRate ** (self.globalStep/1))
    def printLearningRate(self):
        print("Learning rate is:{0}".format(self.calculateLearningRate()))

    def main(self):
        # Initializing the game, graphics and sound
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        screen = pygame.display.set_mode(self.windowSize)
        max_frame_rate = 9999
        dashboard = Dashboard("./img/font.png", 8, screen)
        sound = Sound()
        
        maxReward = 0
        timesMaxedTheReward = 0
        winStreak = 0
        maxWinStreak = 0

        differentialFramesCount = 10
        dxStartFrame = 0
        diffXOverFrames = 0
        dxConstant = 0
        lastX = 0

        epsCoef = 0

        for episodeCounter in range(0, self.experimentsCount):
            print("=============================================")
            print("============WELCOME TO EPISODE {0}============".format(episodeCounter))
            print("=============================================")
            framesCounter = 0

            stillLearning = episodeCounter <= self.learningPeriod

            if stillLearning and episodeCounter/self.experimentsCount >= self.globalStep/self.globalStepMax:
                self.decreaseLearningRate()
                self.printLearningRate()
            
            self.level = Level(screen, sound, dashboard)
            
            self.level.loadLevel('Level1-1')
            dashboard.coins = 0
            dashboard.points = 0
            
            self.mario = Mario(0, 0, self.level, screen, dashboard, sound)
            self.mario.setPos(0,384)
            clock = pygame.time.Clock()

            max_x = 0
            obs = np.append(np.append(self.mario.getObservation(),framesCounter),self.level.getClosestEntityDistance(self.mario))

            finishedLevel = False
            
            while framesCounter < self.maxFrames and not finishedLevel:
                #print(framesCounter)
                pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
                if self.mario.pause:
                    self.mario.pauseObj.update()
                else:
                    self.level.drawLevel(self.mario.camera)
                    dashboard.update()
                    self.mario.update()
                    #print(self.mario.rect.x, self.mario.rect.y)
                    prevObs = obs
                    obs = np.append(np.append(self.mario.getObservation(),framesCounter),self.level.getClosestEntityDistance(self.mario))

                    reward = 0
                    actionIdentifier = self.player.agent.sample_action(obs, self.eps)

                    if max_x < obs[0]:
                        max_x = obs[0]
                        reward += 1
                    
                    # diffXOverFrames = self.mario.rect.x - dxConstant

                    if self.mario.restart:
                        print("I DIED!")
                        finishedLevel = True
                        reward -= 400
                        winStreak = 0
                    else:
                        if max_x >= self.level.max_X and not finishedLevel:
                            finishedLevel = True
                            timesMaxedTheReward += 1
                            winStreak += 1
                            reward += 200
                        else:
                            if self.mario.rect.x > max_x:
                                reward += 5
                            if framesCounter - dxStartFrame >= differentialFramesCount:
                                dxStartFrame = framesCounter
                                dxConstant = self.mario.rect.x
                    if stillLearning:
                        next = self.player.agent.predict(obs)
                        assert(len(next.shape) == 1)
                        G = reward + self.gamma*np.max(next)
                        self.player.agent.update(prevObs, actionIdentifier, G)

                    self.doAction(actionIdentifier)

                pygame.display.update()
                clock.tick(max_frame_rate)
                
                framesCounter += 1

            if timesMaxedTheReward > 0:
                print("I've maxed the reward {0} times!".format(timesMaxedTheReward))
                if maxWinStreak < winStreak:
                    maxWinStreak = winStreak
                print("winStreak {0}, maxWinStreak: {1}".format(winStreak, maxWinStreak))

            if stillLearning:
                print("Still learning!")
                self.running_avg[episodeCounter] = self.totalrewards[0:episodeCounter+1].mean()


                self.eps -= 1 / ((epsCoef+2)*(epsCoef+1)) # eps=1/(n+1) for n=epsCoef ~~ episodeCounter
                epsCoef += 1

                if self.running_avg[episodeCounter]<self.running_avg[int(np.ceil(np.sqrt(episodeCounter)))]:
                    self.eps = 1 / (np.ceil(np.sqrt(episodeCounter)) + 1)
                    epsCoef = np.ceil(np.sqrt(episodeCounter))
                    print("Reverting epsilon**: {0}".format(self.eps))
                elif episodeCounter>=self.episodesPerStep and self.running_avg[episodeCounter]<self.running_avg[episodeCounter-self.episodesPerStep]:
                    self.eps = 1 / ((episodeCounter-self.episodesPerStep) + 1)
                    epsCoef = episodeCounter
                    print("Reverting epsilon: {0}".format(self.eps))
            else:
                self.running_avg[episodeCounter] = self.totalrewards[learningPeriod:episodeCounter+1].mean()

            totalreward = max_x
            if totalreward > maxReward:
                maxReward = totalreward
            self.totalrewards[episodeCounter] = totalreward

            print("Episode: {0}, maxReward: {1} eps: {2} totalReward: {3}".format(episodeCounter, maxReward, self.eps, totalreward))
            print("Avg: {0}".format(self.running_avg[episodeCounter]/self.level.max_X))

        for reward in self.totalrewards:
            print(reward)

    # Controls
    def doAction(self, actionIdentifier):
        if (actionIdentifier == 0):
            # Left
            self.mario.traits['jumpTrait'].jump(True)
            self.mario.traits["goTrait"].direction = -1
        elif (actionIdentifier == 1):
            # Right
            self.mario.traits['jumpTrait'].jump(True)
            self.mario.traits["goTrait"].direction = 1
        if (actionIdentifier == 2):
            # Left
            self.mario.traits['jumpTrait'].jump(False)
            self.mario.traits["goTrait"].direction = -1
        elif (actionIdentifier == 3):
            # Right
            self.mario.traits['jumpTrait'].jump(False)
            self.mario.traits["goTrait"].direction = 1

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution() 
    window = MainWindow()
    window.main()
