import constants
import sys
import cv2
import numpy as np
import constants
from utils import convert_meters_to_pixels_distance, convert_pixels_to_meters_distance
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
