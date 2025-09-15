from utils import (read_video, save_video, get_distance,
                   convert_pixels_to_meters_distance, draw_player_stats, TennisPlayerExtractor)
from trackers import (PlayerTracker, BallTracker)
from court_line_detector import CourtLineDetector
import constants
import cv2
from graphical_court import GraphicalCourt
from copy import deepcopy
import pandas as pd
import constants


def main():
    # Read video frames
    input_path = r"input_videos\input_video.mp4"
    video_frames = read_video(input_path)

    # Identify the players
    name_extractor = TennisPlayerExtractor()
    player_info = name_extractor.extract_with_bounding_boxes(
        image_path=r"input_videos/image_captured.jpg")
    player_names = [info['name'] for info in player_info]

    # Initialize player tracker and detect players in frames
    player_tracker = PlayerTracker(model_path="yolov8x")
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stubs=True, stub_path="tracker_stubs/player_detections.pkl")

    # Initialize ball tracker and detect the ball in frames
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stubs=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect the frames when the ball was hit
    ball_hit_frames = ball_tracker.get_ball_shots_frames(ball_detections)
    print(f"Ball hit frames: {ball_hit_frames}")

    # Detect court lines
    court_line_detector = CourtLineDetector(
        model_path="models/keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Draw only the players and not the rest of the people
    player_detections = player_tracker.filter_players(
        court_keypoints, player_detections)

    # Initialize graphical court
    graphical_court = GraphicalCourt(video_frames[0])

    # Convert positions to graphical court coordinates
    player_graph_court_detections, ball_graph_court_detections = graphical_court.convert_bounding_boxes_to_graphical_court_coord(
        player_detections, ball_detections, court_keypoints)

    player_stats_data = [{
        'frame_num': 0,

        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0
    }]

    for ball_shot_idx in range(len(ball_hit_frames) - 1):
        start_frame = ball_hit_frames[ball_shot_idx]
        end_frame = ball_hit_frames[ball_shot_idx + 1]

        # Calculate the time of one shot to cover the court
        # divide by 24 because the input was reduced at 24 fps
        ball_shot_time_seconds = (end_frame - start_frame) / 24

        # Calculate the distance covered by the ball
        distance_px = get_distance(
            ball_graph_court_detections[start_frame][1], ball_graph_court_detections[end_frame][1])
        distance_meters = convert_pixels_to_meters_distance(distance_px,
                                                            constants.DOUBLE_LINE_WIDTH,
                                                            graphical_court.get_width())

        # Calculate the speed in km/h
        ball_speed = (distance_meters / ball_shot_time_seconds) * \
            3.6  # * 3600 meters / 1000 seconds

        # Speed of player who's closer to the ball
        player_pos = player_graph_court_detections[start_frame]
        shot_player = min(player_pos.keys(), key=lambda pid: get_distance(
            player_pos[pid], ball_graph_court_detections[start_frame][1]))

        # Speed of the opponent player
        opp_player_id = 1 if shot_player == 2 else 2

        opp_dist_px = get_distance(player_graph_court_detections[start_frame][opp_player_id],
                                   player_graph_court_detections[end_frame][opp_player_id])
        opp_dist_meters = convert_pixels_to_meters_distance(opp_dist_px,
                                                            constants.DOUBLE_LINE_WIDTH,
                                                            graphical_court.get_width())

        opp_speed = (opp_dist_meters / ball_shot_time_seconds) * 3.6

        # Track the stats
        current_player_stats = deepcopy(player_stats_data[-1])

        current_player_stats['frame_num'] = start_frame

        current_player_stats[f'player_{shot_player}_number_of_shots'] += 1
        current_player_stats[f'player_{shot_player}_total_shot_speed'] += ball_speed
        current_player_stats[f'player_{shot_player}_last_shot_speed'] = ball_speed

        current_player_stats[f'player_{opp_player_id}_total_player_speed'] += opp_speed
        current_player_stats[f'player_{opp_player_id}_last_player_speed'] = opp_speed

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(
        frames_df, player_stats_data_df, on='frame_num', how='left').ffill()

    # Calculate the avgs
    player_stats_data_df['player_1_avg_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / \
        player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_avg_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / \
        player_stats_data_df['player_2_number_of_shots']

    player_stats_data_df['player_1_avg_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / \
        player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_avg_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / \
        player_stats_data_df['player_1_number_of_shots']

    # Draw the bounding boxes on the frames
    output_video_frames = player_tracker.draw_boxes(
        video_frames, player_detections, player_names)
    output_video_frames = ball_tracker.draw_boxes(
        output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_frames(
        output_video_frames, court_keypoints)

    # Draw the graphical court
    output_video_frames = graphical_court.draw_court_bg(output_video_frames)
    output_video_frames = graphical_court.draw_points_on_graphical_court(
        output_video_frames, player_graph_court_detections, color=(0, 255, 0))
    output_video_frames = graphical_court.draw_points_on_graphical_court(
        output_video_frames, ball_graph_court_detections, color=(0, 255, 255))

    # Draw player stats
    output_video_frames = draw_player_stats(
        output_video_frames, player_stats_data_df, player_names)

    # We will display the frame number on each frame
    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Save the processed video frames
    save_video(output_video_frames, r"output_videos\output_video.avi")


if __name__ == "__main__":
    main()
