def convert_pixels_to_meters_distance(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    """
    Convert a distance in pixels to meters using a reference height.

    Args:
        pixel_distance (float): The distance in pixels to convert.
        reference_height_in_meters (float): The known height in meters.
        reference_height_in_pixels (float): The known height in pixels.

    Returns:
        float: The converted distance in meters.
    """
    if reference_height_in_pixels == 0:
        raise ValueError("Reference height in pixels cannot be zero.")
    meters_per_pixel = reference_height_in_meters / reference_height_in_pixels
    return pixel_distance * meters_per_pixel


def convert_meters_to_pixels_distance(meter_distance, reference_height_in_meters, reference_height_in_pixels):
    """
    Convert a distance in meters to pixels using a reference height.

    Args:
        meter_distance (float): The distance in meters to convert.
        reference_height_in_meters (float): The known height in meters.
        reference_height_in_pixels (float): The known height in pixels.

    Returns:
        float: The converted distance in pixels.
    """
    if reference_height_in_meters == 0:
        raise ValueError("Reference height in meters cannot be zero.")
    pixels_per_meter = reference_height_in_pixels / reference_height_in_meters
    return meter_distance * pixels_per_meter
