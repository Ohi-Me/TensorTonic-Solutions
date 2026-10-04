import math

def rotate_image(image: list, angle_degrees: float) -> list:
    """
    Returns the counterclockwise nearest-neighbor rotation.
    """
    # Write code here
    angle = math.radians(angle_degrees)
    c = math.cos(angle)
    s = math.sin(angle)

    h = len(image)
    w = len(image[0])

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    ans = [[0] * w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            dy = i - cy
            dx = j - cx

            sy = round(cy + dy * c + dx * s)
            sx = round(cx - dy * s + dx * c)

            if 0 <= sy < h and 0 <= sx < w:
                ans[i][j] = image[sy][sx]

    return ans