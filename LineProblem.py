import Agent
import Network
import random

Inputs = 4
Outputs = 1

class LineProblem:
    def __init__(self, rewardSize, shareSize):
        self.RewardStart = .5 - (rewardSize / 2)
        self.RewardEnd = .5 + (rewardSize / 2)
        self.ShareStart = self.RewardEnd - shareSize

    def GetFitness(self, agent, stepsToTake, uncertainty):
        currentLocation = random.uniform(0, 1)
        stepsInReward = 0
        stepsInShare = 0

        for i in range(stepsToTake):
            input = [currentLocation, self.RewardStart, self.RewardEnd, self.ShareStart]
            output = agent.Network.Feedforwad(input)

            distanceToMove = (output[0] - .5) * 2
            potentialError = distanceToMove * distanceToMove * uncertainty
            distanceMoved = distanceToMove + random.uniform(-potentialError, potentialError)

            currentLocation = currentLocation + distanceMoved
            if(currentLocation < 0):
                currentLocation += 1

            if(currentLocation < self.RewardEnd):
                if(currentLocation > self.RewardStart):
                    stepsInReward += 1
                if(currentLocation > self.ShareStart):
                    stepsInShare += 1

        return [stepsInReward, stepsInShare]
