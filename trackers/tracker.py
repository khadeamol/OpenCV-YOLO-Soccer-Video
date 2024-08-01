from ultralytics import YOLO
import pandas as pd
import supervision as sv
import pickle, os
import sys
import cv2
import numpy as np
sys.path.append('../')
from utils import getCenterofBBox, get_bbox_width, getFootPosition



class Tracker():

    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def addPositionToTracks(self, tracks):
        for object, objectTracks in tracks.items():
            for frameNum, track in enumerate(objectTracks):
                for trackId, trackInfo in track.items():
                    bbox = trackInfo['bbox']
                    if object == "ball":
                        position = getCenterofBBox(bbox)
                    else:
                        position = getFootPosition(bbox)
                    tracks[object][frameNum][trackId]['position'] = position



    def interpolateBallPosition(self, ballPosition):
        ballPosition = [x.get(1,{}).get('bbox',[]) for x in ballPosition]
        dfBallPosition = pd.DataFrame(ballPosition, columns=['x1','y1','x2','y2'])
        dfBallPosition = dfBallPosition.interpolate()
        dfBallPosition = dfBallPosition.bfill()
        ballPosition = [{1: {"bbox":x}}for x in dfBallPosition.to_numpy().tolist()]

        return ballPosition

    
    def detectFrames(self, frames):
        batchSize = 20
        detections = []
        for i in range(0, len(frames), batchSize):
            detectionsBatch = self.model.predict(frames[i:i+batchSize], conf = 0.1, device = 'mps')
            detections += detectionsBatch
        return detections 
    
    def getObjectTracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detectFrames(frames)

        tracks = {
            "player":[],
            "referee":[], 
            "ball":[]
        }


        for frameNum, detection in enumerate(detections):
            clsNames = detection.names
            clsNamesInv = {v:k for k,v in clsNames.items()}
            # convert to supervision detection format
            detectionSupervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to player object
            for objectInd, class_id in enumerate(detectionSupervision.class_id):
                if clsNames[class_id] == "goalkeeper":
                    detectionSupervision.class_id[objectInd] = clsNamesInv["player"]

            # track objects
            detectionWithTracks = self.tracker.update_with_detections(detectionSupervision)

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frameDetection in detectionWithTracks:
                bbox = frameDetection[0].tolist()
                clsId = frameDetection[3]
                trackId = frameDetection[4]

                if clsId == clsNamesInv['player']:
                    tracks["player"][frameNum][trackId] = {"bbox":bbox}
                if clsId == clsNamesInv['referee']:
                    tracks["referee"][frameNum][trackId] = {"bbox":bbox}

            for frameDetection in detectionSupervision:
                bbox = frameDetection[0].tolist()
                clsId = frameDetection[3]
                if clsId == clsNamesInv['ball']:
                    tracks["ball"][frameNum][1] = {"bbox":bbox}

            print(detectionWithTracks)
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def drawEllipse(self, frame, bbox, color, trackId=None):

        # Draw Ellipse
        y2 = int(bbox[3])
        xCenter, _ = getCenterofBBox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(           
            frame,
            center = (xCenter, y2),
            axes = (30, 7),
            angle=0.0,
            startAngle=30,
            endAngle=300,
            color = color,
            thickness=4,
            lineType=cv2.LINE_AA
            )        
        
        # Draw Rectangle

        rectangleWidth = 30
        rectangleHeight = 20
        x1Rect = xCenter - rectangleWidth//2
        x2Rect = xCenter + rectangleWidth//2
        y1Rect = (y2 - rectangleHeight//2) + 18
        y2Rect = (y2 + rectangleHeight//2) + 18

        if trackId is not None:
            cv2.rectangle(frame, (int(x1Rect), int(y1Rect)),(int(x2Rect),int(y2Rect)), color, cv2.FILLED)

        x1Text = x1Rect+8
        if trackId > 99:
            x1Text -= 10

        cv2.putText(frame, f"{trackId}", (int(x1Text),int(y1Rect+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        
        return frame


    def drawTriangle(self, frame, bbox, color):
        y = int(bbox[1])

        x, _ = getCenterofBBox(bbox)
        trianglePoints = np.array([[x,y],[x-10, y-20], [x+10, y-20]])
        cv2.drawContours(frame, [trianglePoints], 0, color,cv2.FILLED)
        cv2.drawContours(frame, [trianglePoints], 0, (0,0,0),2)
        return frame 
    

    def drawTeamBallControl(self, frame, frameNum, teamBallControl):
        # draw rectangle

        overlay = frame.copy()

        cv2.rectangle(overlay, (1350,850), (1900, 970), (255,255,255), -1)
        alpha = 0.4

        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        teamBallControlTillFrame = teamBallControl[:frameNum+1]
        # get the number of frames for which each team has the ball

        teamOneNumFrames = teamBallControlTillFrame[teamBallControlTillFrame==1].shape[0]
        teamTwoNumFrames = teamBallControlTillFrame[teamBallControlTillFrame==2].shape[0]

        teamOne = teamOneNumFrames/(teamOneNumFrames+teamTwoNumFrames)
        teamTwo = teamTwoNumFrames/(teamOneNumFrames+teamTwoNumFrames)
        cv2.putText(frame, f"Team 1 Ball Control: {round(teamOne*100,2)}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),3)
        cv2.putText(frame, f"Team 2 Ball Control: {round(teamTwo*100,2)}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),3)

        return frame
    def drawAnnotations(self, videoFrames, tracks, teamBallControl):

        outputVideoFrames = []
        for frameNum, frame in enumerate(videoFrames):
            frame = frame.copy()

            playerDict = tracks["player"][frameNum]
            ballDict = tracks["ball"][frameNum]
            refereeDict = tracks["referee"][frameNum]
            
            
            # draw players
            for trackId, player in playerDict.items():
                color = player.get("teamColor")
                frame = self.drawEllipse(frame, player['bbox'], color, trackId=trackId)

                if player.get("hasBall", False):
                    frame =  self.drawTriangle(frame, player["bbox"], (0,0,255))

            for trackId, referee in refereeDict.items():
                frame = self.drawEllipse(frame, referee['bbox'], color=(255,255,0), trackId=trackId)
            
            for trackId, ball in ballDict.items():
                frame = self.drawTriangle(frame, ball['bbox'], color=(255,0,0))


            # draw team ball control %
            frame = self.drawTeamBallControl(frame, frameNum, teamBallControl)

            outputVideoFrames.append(frame)

        return outputVideoFrames
    

            