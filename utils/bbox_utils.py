from typing import Tuple, List


def get_bbox_center(bbox: List[int]) -> Tuple[int, int]:
    """
    Calculates the center of a bounding box.

    Args:
        bbox (List[int]): A list of four integers representing the bounding box coordinates [x1, y1, x2, y2].

    Returns:
        Tuple[int, int]: The (x, y) coordinates of the bounding box center.
    """
    if len(bbox) != 4:
        raise ValueError(
            "Bounding box must contain exactly four elements [x1, y1, x2, y2]."
        )

    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: List[int]) -> int:
    """
    Calculates the width of a bounding box.

    Args:
        bbox (List[int]): A list of four integers representing the bounding box coordinates [x1, y1, x2, y2].

    Returns:
        int: The width of the bounding box.
    """
    if len(bbox) != 4:
        raise ValueError(
            "Bounding box must contain exactly four elements [x1, y1, x2, y2]."
        )

    return bbox[2] - bbox[0]
