def get_center(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) // 2)
    center_y = int((y1 + y2) // 2)
    return (center_x, center_y)


def get_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_foot_position(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) // 2), y2)


def get_closest_kp_index(point, kp, indices):
    min_distance = float('inf')
    closest_index = -1
    for idx in indices:
        kp_point = (kp[idx * 2], kp[idx * 2 + 1])
        distance = abs(get_distance(point, kp_point))
        if distance < min_distance:
            min_distance = distance
            closest_index = idx
    return closest_index


def get_height_of_box(box):
    return box[3] - box[1]


def measure_xy_dist(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def get_center_of_box(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) // 2)
    center_y = int((y1 + y2) // 2)
    return (center_x, center_y)
