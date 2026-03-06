def get_center_of_bbox(bbox):
    """
    计算边界框的中心点坐标
    :param bbox: 边界框，格式为 (x1, y1, x2, y2)
    :return: 中心点坐标 (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """
    计算边界框的宽度
    :param bbox: 边界框，格式为 (x1, y1, x2, y2)
    :return: 边界框宽度
    """
    x1, y1, x2, y2 = bbox
    return int(x2 - x1)

def measure_distance(point1, point2):
    """
    计算两点之间的欧几里得距离
    :param point1: 第一个点坐标 (x1, y1)
    :param point2: 第二个点坐标 (x2, y2)
    :return: 两点之间的距离
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def measure_xy_distance(point1, point2):
    """
    计算两点在 x 和 y 方向上的距离差
    :param point1: 第一个点坐标 (x1, y1)
    :param point2: 第二个点坐标 (x2, y2)
    :return: x 方向距离差, y 方向距离差
    """
    x1, y1 = point1
    x2, y2 = point2
    return x2 - x1, y2 - y1

def get_foot_position(bbox):
    """
    获取边界框底部中心点坐标（脚的位置）
    :param bbox: 边界框，格式为 (x1, y1, x2, y2)
    :return: 底部中心点坐标 (center_x, bottom_y)
    """
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2),int(y2)