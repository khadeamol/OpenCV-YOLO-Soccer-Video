from utils import read_video, save_video
import cv2
import numpy as np
from trackers import Tracker
from team_assigner import TeamAssigner
from playerBallAssignment import PlayerBallAssigner

from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_transformer import SpeedAndDistance_Estimator
def main():
    # read video
    videoFrames = read_video('input_videos/08fd33_4.mp4')

    # initialize tracker
    tracker = Tracker('models/best-3.pt')

    tracks = tracker.getObjectTracks(videoFrames,
                                     read_from_stub=True,
                                     stub_path="stubs/tracks_stubs.pkl")
    
    # get object position
    tracker.addPositionToTracks(tracks)

    # camera movement estimator
    getCameraMovement = CameraMovementEstimator(videoFrames[0])
    camera_movement_per_frame = getCameraMovement.getCameraMovement(videoFrames, True, "stubs/camera_movement_stub.pkl")
    getCameraMovement.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # view transformer 
    viewTransformer = ViewTransformer()
    viewTransformer.add_transformed_position_to_tracks(tracks)


    # interpolate ball positions
    tracks["ball"] = tracker.interpolateBallPosition(tracks["ball"])


    # speed and distance estimator
    speedDistanceEstimator = SpeedAndDistance_Estimator()
    speedDistanceEstimator.add_speed_and_distance_to_tracks(tracks)
    

    # assign teams
    teamAssigner = TeamAssigner()
    teamAssigner.assignTeamColor(videoFrames[0], tracks['player'][0])


    for frameNum, playerTrack in enumerate(tracks['player']):
        for playerId, track in playerTrack.items():
            team = teamAssigner.getPlayerTeam(videoFrames[frameNum], track['bbox'], playerId)   

            tracks['player'][frameNum][playerId]['team'] = team
            tracks['player'][frameNum][playerId]['teamColor'] = teamAssigner.teamColors[team]

    # Assign Ball 
    playerAssigner = PlayerBallAssigner()
    teamBallControl = []

    for frameNum, playerTrack in enumerate(tracks['player']):
        ballBbox = tracks["ball"][frameNum][1]["bbox"]
        assignedPlayer = playerAssigner.AssignBalltoPlayer(playerTrack, ballBbox)

        if assignedPlayer != -1:
            tracks["player"][frameNum][assignedPlayer]['hasBall'] = True
            teamBallControl.append(tracks['player'][frameNum][assignedPlayer]['team'])
        else:
            teamBallControl.append(teamBallControl[-1])
    teamBallControl = np.array(teamBallControl)
    
    outputVideoFrames = tracker.drawAnnotations(videoFrames, tracks, teamBallControl)
    print("Annotations Drawn")
    
    # draw speed and distance

    speedDistanceEstimator.draw_speed_and_distance(outputVideoFrames, tracks)


    # save video
    outputVideoFrames = getCameraMovement.drawCameraMovement(frames=outputVideoFrames, camera_movement_per_frame= camera_movement_per_frame)
    save_video(outputVideoFrames, 'output_videos/output_video.mp4')

if __name__=="__main__":
    main()

