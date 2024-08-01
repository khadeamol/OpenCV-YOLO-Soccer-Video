import cv2
import sys
sys.path.append("../")

from utils import measureDistance, getFootPosition


class SpeedAndDistance_Estimator():
    def __init__(self) -> None:
        
        self.frameWindow = 5
        self.frameRate = 24
    
    def add_speed_and_distance_to_tracks(self, tracks):

        total_distance = {}

        for object, objectTracks in tracks.items():
            if object == "ball" or object == "referee":
                continue

            number_of_frames = len(objectTracks)
            for frameNum in range(0, number_of_frames, self.frameWindow):
                lastFrame = min(frameNum+ self.frameWindow, number_of_frames-1)
                for trackId, _ in objectTracks[frameNum].items():
                    if trackId not in objectTracks[lastFrame]:
                        continue
                
                    startPosition = objectTracks[frameNum][trackId]["positionAdjusted"]
                    endPosition = objectTracks[lastFrame][trackId]["positionAdjusted"]

                    if startPosition is None or endPosition is None:
                        continue

                    distanceCovered = measureDistance(startPosition, endPosition)
                    timeElapsed = (lastFrame-frameNum)/self.frameRate
                    speed_mps = distanceCovered/timeElapsed
                    speed_kmph = speed_mps*3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if trackId not in total_distance[object]:
                        total_distance[object][trackId] = 0
                    
                    total_distance[object][trackId] += distanceCovered

                    for frame_num_batch in range(frameNum, lastFrame):
                        if trackId not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][trackId]['speed'] = speed_kmph
                        tracks[object][frame_num_batch][trackId]['distance'] = distanceCovered

    def draw_speed_and_distance(self, frames, tracks):
        outputFrames = []

        for frameNum, frame in enumerate(frames):
            for object, objectTrack in tracks.items():
                if object == "ball" or object == "referee":
                    continue
                    
                for _, trackInfo  in objectTrack[frameNum].items():
                    if "speed" in trackInfo:
                        speed = trackInfo.get('speed', None)
                        distance = trackInfo.get('distance', None)

                        if speed is None or distance is None:
                            continue
                    bbox = trackInfo["bbox"]
                    position = getFootPosition(bbox)
                    position = list(position)
                    position[1]+=40 

                    position = tuple(map(int, position))
                    cv2.putText(frame, f"{speed:.2f}kmph", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
                    cv2.putText(frame, f"{distance:.2f}m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)

            outputFrames.append(frame)
        return outputFrames