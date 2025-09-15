import constants
import sys
import cv2
import numpy as np
import constants
from utils import (convert_meters_to_pixels_distance, convert_pixels_to_meters_distance,
                   get_foot_position, get_closest_kp_index, get_height_of_box, measure_xy_dist,
                   get_center_of_box, get_distance)
sys.path.append('../')


class GraphicalCourt:
    def __init__(self, frame):
        self.drawing_rect_w = 250
        self.drawing_rect_h = 500
        self.buffer = 50
        self.padding = 20

        self.set_canvas_position(frame)
        self.set_court_position()
        self.set_keypoints()
        self.set_court_lines()

    def set_canvas_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.start_x = self.end_x - self.drawing_rect_w

        self.end_y = self.drawing_rect_h + self.buffer
        self.start_y = self.end_y - self.drawing_rect_h

    def set_court_position(self):
        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y + self.padding
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y - self.padding

        self.court_width = self.court_end_x - self.court_start_x
        self.court_height = self.court_end_y - self.court_start_y

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixels_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_width
        )

    def set_keypoints(self):

        kp = []  # store (x, y) tuples for clarity

        # Point 0
        kp.append((int(self.court_start_x), int(self.court_start_y)))
        # Point 1
        kp.append((int(self.court_end_x), int(self.court_start_y)))
        # Point 2
        kp.append((
            int(self.court_start_x),
            self.court_start_y +
            self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        ))
        # Point 3
        kp.append((
            kp[0][0] + self.court_width,
            kp[2][1]
        ))
        # Point 4
        kp.append((
            kp[0][0] +
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE),
            kp[0][1]
        ))
        # Point 5
        kp.append((
            kp[2][0] +
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE),
            kp[2][1]
        ))
        # Point 6
        kp.append((
            kp[1][0] -
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE),
            kp[1][1]
        ))
        # Point 7
        kp.append((
            kp[3][0] -
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE),
            kp[3][1]
        ))
        # Point 8
        kp.append((
            kp[4][0],
            kp[4][1] +
            self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        ))
        # Point 9
        kp.append((
            kp[8][0] +
            self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH),
            kp[8][1]
        ))
        # Point 10
        kp.append((
            kp[5][0],
            kp[5][1] -
            self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        ))
        # Point 11
        kp.append((
            kp[10][0] +
            self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH),
            kp[10][1]
        ))
        # Point 12
        kp.append((
            int((kp[8][0] + kp[9][0]) / 2),
            kp[8][1]
        ))
        # Point 13
        kp.append((
            int((kp[10][0] + kp[11][0]) / 2),
            kp[10][1]
        ))

        # Flatten back to list [x0,y0,x1,y1,...] if needed
        self.drawing_key_points = [c for point in kp for c in point]

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    def draw_bg_rect(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y),
                      (self.end_x, self.end_y), (255, 255, 255), -1)
        output_frame = frame.copy()

        a = 0.5  # set the transparency factor
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(frame, a, shapes, 1 - a, 0)[mask]
        # output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

        return output_frame

    def draw_court_bg(self, frames):
        output_frames = []

        for frame in frames:
            frame = self.draw_bg_rect(frame)
            frame = self.draw_mini_court(frame)
            output_frames.append(frame)

        return output_frames

    def draw_mini_court(self, frame):
        # draw the keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            cv2.circle(frame, (int(self.drawing_key_points[i]), int(self.drawing_key_points[i+1])),
                       5, (0, 0, 255), -1)

        # draw the lines
        for line in self.lines:
            pt1 = (int(self.drawing_key_points[line[0]*2]),
                   int(self.drawing_key_points[line[0]*2+1]))
            pt2 = (int(self.drawing_key_points[line[1]*2]),
                   int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, pt1, pt2, (0, 0, 0), 2)

        # draw the net
        net_y = int(
            (self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        net_p1 = (int(self.drawing_key_points[0]), net_y)
        net_p2 = (int(self.drawing_key_points[2]), net_y)
        cv2.line(frame, net_p1, net_p2, (255, 0, 255), 2)

        return frame

    def get_start_points(self):
        return (self.start_x, self.start_y)

    def get_width(self):
        return self.drawing_rect_w

    def get_keypoints(self):
        return self.drawing_key_points

    def get_graphical_court_coord(self, object_pos, closest_kp, closest_kp_idx, player_h_px, player_h_m):
        # get distance in pixels
        dist_from_kp_x_px, dist_from_kp_y_px = measure_xy_dist(
            object_pos, closest_kp)

        # convert the distance to meters
        dist_from_kp_x_m = convert_pixels_to_meters_distance(
            dist_from_kp_x_px,
            player_h_m,
            player_h_px
        )

        dist_from_kp_y_m = convert_pixels_to_meters_distance(
            dist_from_kp_y_px,
            player_h_m,
            player_h_px
        )

        # convert the distance back to pixels in the graphical court
        graph_court_dist_x_px = self.convert_meters_to_pixels(
            dist_from_kp_x_m)
        graph_court_dist_y_px = self.convert_meters_to_pixels(
            dist_from_kp_y_m)

        closest_kp_graph_court = (self.drawing_key_points[closest_kp_idx * 2],
                                  self.drawing_key_points[closest_kp_idx * 2 + 1])

        graph_court_player_pos = (closest_kp_graph_court[0] + graph_court_dist_x_px,
                                  closest_kp_graph_court[1] + graph_court_dist_y_px)

        return graph_court_player_pos

    def convert_bounding_boxes_to_graphical_court_coord(self, player_bb, ball_bb, original_kp):

        player_h = {
            1: constants.PLAYER1_H,
            2: constants.PLAYER2_H
        }

        output_player_bb = []
        output_ball_bb = []

        for frame_number, player_bbox in enumerate(player_bb):
            output_player_bb_dict = {}

            ball_box = ball_bb[frame_number][1]
            ball_pos = get_center_of_box(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda pid: get_distance(
                ball_pos, get_center_of_box(player_bbox[pid])))

            for player_id, bbox in player_bbox.items():
                foot_pos = get_foot_position(bbox)

                # Get the closest kp in the original frame
                closest_kp_index = get_closest_kp_index(
                    foot_pos, original_kp, [0, 2, 12, 13])
                closest_kp = (original_kp[closest_kp_index * 2],
                              original_kp[closest_kp_index * 2 + 1])

                # Get the player height in pixels
                frame_idx_min = max(0, frame_number - 20)
                frame_idx_max = min(len(player_bb), frame_number + 50)
                bbox_heights_in_px = [get_height_of_box(
                    player_bb[i][player_id]) for i in range(frame_idx_min, frame_idx_max)]
                max_player_height_in_px = max(bbox_heights_in_px)

                # get the player position in the graphical court
                graph_court_player_pos = self.get_graphical_court_coord(
                    foot_pos,
                    closest_kp,
                    closest_kp_index,
                    max_player_height_in_px,
                    player_h[player_id]
                )

                output_player_bb_dict[player_id] = graph_court_player_pos

                if closest_player_id_to_ball == player_id:
                    # Get the closest kp in the original frame
                    closest_kp_index = get_closest_kp_index(
                        ball_pos, original_kp, [0, 2, 12, 13])
                    closest_kp = (original_kp[closest_kp_index * 2],
                                  original_kp[closest_kp_index * 2 + 1])

                    graph_court_ball_pos = self.get_graphical_court_coord(
                        ball_pos,
                        closest_kp,
                        closest_kp_index,
                        max_player_height_in_px,
                        player_h[player_id]
                    )

                    output_ball_bb.append({1: graph_court_ball_pos})

            output_player_bb.append(output_player_bb_dict)

        return output_player_bb, output_ball_bb

    def draw_points_on_graphical_court(self, frames, positions, color=(255, 0, 0)):
        for frame_id, frame in enumerate(frames):
            for _, pos in positions[frame_id].items():
                x, y = pos
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), 5, color, -1)

        return frames
