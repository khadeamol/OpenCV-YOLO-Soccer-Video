
import cv2
import pickle
import numpy as np
import os
import sys

sys.path.append('../')
from utils import measureDistance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        firstFrameGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.min_distance = 5
        mask_features = np.zeros_like(firstFrameGrayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners = 100, 
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )
        
        self.lkparams = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03)
        )
        
        pass

    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, objectTracks in tracks.items():
            for frameNum, track in enumerate(objectTracks):
                for trackId, trackInfo in track.items():
                    position = trackInfo["position"]
                    cameraMovement = camera_movement_per_frame[frameNum]
                    positionAdjusted = (position[0]-cameraMovement[0],position[1]-cameraMovement[1])
                    tracks[object][frameNum][trackId]['positionAdjusted'] = positionAdjusted

    def getCameraMovement(self, frames, read_from_stub = False, stub_path = None):
        # read stub if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
        # camera movement per frame

        camera_movement = [[0,0]]*len(frames)

        oldGray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        oldFeatures = cv2.goodFeaturesToTrack(oldGray, **self.features)

        for frameNum in range(1,len(frames)):
            frameGray = cv2.cvtColor(frames[frameNum],cv2.COLOR_BGR2GRAY)
            newFeatures, _, _ = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, oldFeatures, None, **self.lkparams)
            
            max_distance = 0
            cameraMovementX, cameraMovementY = 0,0
            for i, (new,old) in enumerate(zip(newFeatures, oldFeatures)):
                newFeatures_point = new.ravel()
                oldFeatures_point = old.ravel()

                distance = measureDistance(newFeatures_point, oldFeatures_point)
                if distance > max_distance:
                    max_distance = distance 
                    cameraMovementX, cameraMovementY = measure_xy_distance(oldFeatures_point, newFeatures_point)

            if max_distance > self.min_distance:
                camera_movement[frameNum] = [cameraMovementX, cameraMovementY]
                oldFeatures = cv2.goodFeaturesToTrack(frameGray, **self.features)

            oldGray = frameGray.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement 

    def drawCameraMovement(self, frames, camera_movement_per_frame):
        outputFrames = []
        for frameNum, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            xMovement, yMovement = camera_movement_per_frame[frameNum]
            frame = cv2.putText(frame, f"Camera Movement X: {xMovement:.2f}", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),3)
            frame = cv2.putText(frame, f"Camera Movement Y: {yMovement:.2f}", (10,90),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),3)
            outputFrames.append(frame)

        return outputFrames


