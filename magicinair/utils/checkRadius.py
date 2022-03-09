def check_radius(center, coord, radius) -> bool:
    x_coord, y_coord = coord
    center_x, center_y = center
    distance = (y_coord - center_y) ** 2 + (x_coord - center_x) ** 2
    if distance < (radius ** 2):
        return True
    return False
