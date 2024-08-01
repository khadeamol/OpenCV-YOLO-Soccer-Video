import sys
sys.path.append("../")

from utils import getCenterofBBox, measureDistance

class PlayerBallAssigner():
    def __init__(self):
        self.maxDistance = 70

    def AssignBalltoPlayer(self, players, ballBbox):
        ballPosition = getCenterofBBox(ballBbox)
        minDistance = 99999
        assignedPlayer = -1
        for playerId, player in players.items():
            playerBbox = player["bbox"]

            distanceLeft = measureDistance((playerBbox[0], playerBbox[-1]), ballPosition)
            distanceRight = measureDistance((playerBbox[2], playerBbox[-1]), ballPosition)
            distance = min(distanceLeft, distanceRight)
            if distance < self.maxDistance:
                if distance < minDistance:
                    minDistance = distance
                    assignedPlayer = playerId
        return assignedPlayer
    