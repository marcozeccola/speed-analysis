import re
import numpy as np

"""

"""

_N_GRIPS = 21
_GRIP_DECL = re.compile(r"(?P<id>\d+)]\s+@(?P<relx>[A-Z][1-2])-SN(?P<sn>\d+)#(?P<rely>\d+)")
# According to https://images.ifsc-climbing.org/ifsc/image/private/t_q_good/prd/urwl7n2hnnyvhiwiq0xg.pdf
_GRIP_LOC = """ Tournament grips by specification: 
                1] @F2-SN2#1    2] @G2-SN2#3    3] @A2-SN2#9
                4] @G1-SN3#4    5] @L1-SN3#10   6] @C2-SN4#2
                7] @L1-SN4#8    8] @C2-SN5#3    9] @E2-SN5#9   
                10] @H1-SN6#2   11] @L1-SN6#7   12] @F1-SN6#9
                13] @M1-SN7#4   14] @G1-SN7#9   15] @L1-SN8#1
                16] @I1-SN8#3   17] @C1-SN8#8   18] @A2-SN9#2
                19] @E2-SN9#7   20] @M1-SN9#10  21] @A2-SN10#10 """

# For the 2D plane embedded in 3d real space, assume y-axis = height, x-axis = width, z axis = 0.
GRIP_VALUES_LIST = [
    # See the documentation and the explanation above
    [(ord(rel_x[0]) - ord('A')) * 0.1363 + (int(rel_x[1]) - 1) * 1.5, (int(sec) - 1) * 1.5 + int(rel_y) * 0.1363]
    for g_id, rel_x, sec, rel_y in (grip.groups() for grip in _GRIP_DECL.finditer(_GRIP_LOC))
]
GRIP_VALUES = np.array(GRIP_VALUES_LIST)
GRIP_VALUES_Z_EQ_ZERO = np.array([val + [0.0] for val in GRIP_VALUES_LIST])

# By standard, each grip is 0.35meters ( 35 centimeters )
GRIP_SIZE = 0.35


_N_GRIPS = 20

_GRIP_DECL = re.compile(
    r"(?P<id>\d+)]\s+@(?P<relx>[A-Z][1-2])-SN(?P<sn>\d+)#(?P<rely>\d+)"
)

_GRIP_LOC = """
1]  @F2-SN2#1
2]  @G2-SN2#3
3]  @A2-SN2#9
4]  @G1-SN3#4
5]  @M1-SN3#10
6]  @B2-SN4#2
7]  @M1-SN4#8
8]  @C2-SN5#3
9]  @E2-SN5#9
10] @H1-SN6#2
11] @L1-SN6#7
12] @F1-SN6#9
13] @M1-SN7#4
14] @G1-SN7#9
15] @L1-SN8#1
16] @I1-SN8#3
17] @C1-SN8#8
18] @A2-SN9#2
19] @E2-SN9#7
20] @M1-SN9#10
"""

# Coordinate system:
#   X: horizontal position across ONE 3.0 m lane
#   Y: vertical height above ground
#   Z: ignored here; wall is represented as a 2D plane
#
# IFSC standard:
#   panel size          = 1.5 m x 1.5 m
#   hole pitch          = 0.125 m
#   vertical edge offset= 0.1875 m
#   wall bottom         = 0.20 m above ground
#
# Horizontal columns are:
# A B C D E F G H I L M
# NOTE: J and K do not exist.

HOLE_PITCH = 0.125
PANEL_SIZE = 1.5
VERTICAL_EDGE_OFFSET = 0.1875
WALL_BOTTOM_HEIGHT = 0.20

COLUMN_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "L": 9,
    "M": 10,
}


def grip_to_wall_coordinates(rel_x, panel, row):
    """
    rel_x:
        e.g. 'F2'
        letter = insert column
        1 = left 1.5 m half of lane (SN)
        2 = right 1.5 m half of lane (DX)

    panel:
        vertical panel number, 1..10

    row:
        vertical insert row, 1..10

    Returns:
        [X, Y] in metres, with Y=0 at ground.
    """
    column = rel_x[0]
    half = int(rel_x[1])

    col_idx = COLUMN_INDEX[column]

    # Horizontal:
    # First hole is 0.125 m from the outer edge.
    x = (
        (half - 1) * PANEL_SIZE
        + (col_idx + 1) * HOLE_PITCH
    )

    # Vertical:
    # wall starts 0.20 m above ground
    # first row is 0.1875 m above panel bottom
    y = (
        WALL_BOTTOM_HEIGHT
        + (int(panel) - 1) * PANEL_SIZE
        + VERTICAL_EDGE_OFFSET
        + (int(row) - 1) * HOLE_PITCH
    )

    return [x, y]


GRIP_VALUES_LIST = [
    grip_to_wall_coordinates(rel_x, sec, rel_y)
    for g_id, rel_x, sec, rel_y
    in (grip.groups() for grip in _GRIP_DECL.finditer(_GRIP_LOC))
]

GRIP_VALUES = np.asarray(GRIP_VALUES_LIST, dtype=np.float64)

GRIP_VALUES_Z_EQ_ZERO = np.column_stack(
    (GRIP_VALUES, np.zeros(len(GRIP_VALUES)))
)

# ------------------------------------------------------------------
# 21st YOLO class: FINISH / STOP DEVICE
# ------------------------------------------------------------------
#
# Reference point = centre of the finishing pad sensitive area.
#
# Horizontal position:
# finish pad is on DX10 around columns A/B.
# A = 1.625 m
# B = 1.750 m
# centre = 1.6875 m
#
# Vertical position:
# lowest starting hand hold centre = 1.8875 m above ground
# finish-pad centre is 13.140 m above that reference
#
# 1.8875 + 13.140 = 15.0275 m
#
FINISH_PAD_CENTER = [1.6875, 15.0275]

GRIP_VALUES_LIST.append(FINISH_PAD_CENTER)

assert len(GRIP_VALUES_LIST) == 21

GRIP_VALUES = np.asarray(GRIP_VALUES_LIST, dtype=np.float64)

GRIP_VALUES_Z_EQ_ZERO = np.column_stack([
    GRIP_VALUES,
    np.zeros(len(GRIP_VALUES))
])