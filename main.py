import pygame
import numpy as np
from classes.Dashboard import Dashboard
from classes.Level import Level
from classes.Menu import Menu
from classes.Sound import Sound
from entities.Mario import Mario
from ml.MLPlayer import MLPlayer

class MainWindow:

    def __init__(self):
        self.windowSize = (640,480)
        self.mario = None
        self.player = MLPlayer()
        self.gamma = 0.97

        self.experimentsCount = 5000
        self.totalrewards = np.empty(self.experimentsCount)
        self.running_avg = np.empty(self.experimentsCount)

        self.maxFrames = 300
    
    def main(self):
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        screen = pygame.display.set_mode(self.windowSize)
        max_frame_rate = 600
        dashboard = Dashboard("./img/font.png", 8, screen)
        sound = Sound()
        
        for episodeCount in range(0, self.experimentsCount):
            print("=============================================")
            print("======WELCOME TO EPISODE {}======".format(episodeCount))
            print("=============================================")
            framesCounter = 0

            level = Level(screen, sound, dashboard)
            
            level.loadLevel('Level1-2')
            
            self.mario = Mario(0, 0, level, screen, dashboard, sound)
            self.mario.setPos(0,384)
            clock = pygame.time.Clock()

            max_x = 0
            obs = self.mario.getObservation()

            eps = 1.0 / np.sqrt(episodeCount + 1)
            # play_one()
            while not self.mario.restart and framesCounter <= self.maxFrames:
                #print(framesCounter)
                framesCounter += 1
                pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
                if self.mario.pause:
                    self.mario.pauseObj.update()
                else:
                    level.drawLevel(self.mario.camera)
                    dashboard.update()
                    self.mario.update()
                    #print(self.mario.rect.x, self.mario.rect.y)
                    prevObs = obs
                    obs = self.mario.getObservation()
                    actionIdentifier = self.player.agent.sample_action(obs, eps)
                    self.doAction(actionIdentifier)

                    reward = self.mario.rect.x
                    if max_x<=reward:
                        max_x = reward
                    next = self.player.agent.predict(obs)
                    assert(len(next.shape) == 1)
                    G = reward + self.gamma*np.max(next)
                    self.player.agent.update(prevObs, actionIdentifier, G)

                pygame.display.update()
                clock.tick(max_frame_rate)

            totalreward = max_x
            self.totalrewards[episodeCount - 1] = totalreward
            self.running_avg[episodeCount] = self.totalrewards[max(0, episodeCount - 100):(episodeCount + 1)].mean()
            #if episodeCount % 1 == 0:
            print("episode: {0}, total reward: {1} eps: {2} avg reward (last 100): {3}".format(episodeCount, totalreward, eps,
                                                                                                self.running_avg[episodeCount]), )

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

if __name__ == "__main__":
    window = MainWindow()
    window.main()
