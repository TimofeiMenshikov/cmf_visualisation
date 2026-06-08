class Point:
    def __init__(self, xy):
        self.x = float(xy[0])
        self.y = float(xy[1])


class Polygon:
    def __init__(self, points):
        self.points = [(float(x), float(y)) for x, y in points]

    def contains(self, point):
        points = self.points
        if len(points) < 3:
            return False

        inside = False
        j = len(points) - 1
        for i, (xi, yi) in enumerate(points):
            xj, yj = points[j]
            if (yi > point.y) != (yj > point.y):
                x_intersection = (xj - xi) * (point.y - yi) / (yj - yi) + xi
                if point.x < x_intersection:
                    inside = not inside
            j = i
        return inside

