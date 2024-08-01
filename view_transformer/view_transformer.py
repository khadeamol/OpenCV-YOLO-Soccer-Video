import numpy as np
import cv2
class ViewTransformer():
    def __init__(self) -> None:
        fieldWidth = 68
        fieldLength = 40

        self.pixel_vertices = np.array(
            [
                [110,1035],
                [265,275],
                [910,260],
                [1640,950]]
            
        )


        self.target_vertices = np.array(

            [[0,fieldWidth],
            [0,0],
            [fieldLength,0],
            [fieldLength,fieldWidth]]
        )

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        
        self.perspective_transfomer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]),int(point[1]))
        isInside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0
        if not isInside:
            return None
        else:
            reshapedPoint = point.reshape(-1,1,2).astype(np.float32)
            transformedPoint = cv2.perspectiveTransform(reshapedPoint, self.perspective_transfomer)

        return transformedPoint.reshape(-1,2)


    def add_transformed_position_to_tracks(self, tracks):
        for object, objectTracks in tracks.items():
            for frameNum, track in enumerate(objectTracks):
                for trackId, trackInfo in track.items():
                    position = trackInfo["positionAdjusted"]
                    position = np.array(position)
                    positionTransformed = self.transform_point(position)
                    if positionTransformed is not None:
                        positionTransformed = positionTransformed.squeeze().tolist()
                    tracks[object][frameNum][trackId]["positionAdjusted"] = positionTransformed

