import numpy as np
import cv2 as cv
import math
import logging
from typing import List
from board import Board

## VERY IMPORTANT!!
## Y(top(0), bottom(max))  X(left(0), right(max))
## Images are NOT in RGB, and are in BGR
## JUST CHANGE IT TO HSV ANYWAY

BOARD_DIMS = [26, 9]
TILE_WIDTH = 40
BOARD_TL_PX = [13, 977]
BLUE = 115.5
CYAN = 90
YELLOW = 30
ORANGE = 26

# STATE_TO_COLOUR = {
#     -2: (CYAN, 60, 250),
#     -1: (BLUE, 100, 250),
#     0: (0, 0, 15),
#     1: (YELLOW, 100, 250),
#     2: (ORANGE, 60, 250),
# }

STATE_TO_CHAR = {-2: "C", -1: "B", 0: "", 1: "Y", 2: "O"}

#STATE_TO_INVERTED_COLOUR = {x:((255/2+y[0]) % 255, 255-y[1], y[2]) for x, y in STATE_TO_COLOUR.items()}

def nonlinear_adjust_vert(y_px_ind):
    return math.floor(10 * (-0.5 + 2 * np.exp(-((y_px_ind - 540) / 600) ** 2)))

def showhsv(frame):
    cv.imshow("Display window", cv.cvtColor(frame, cv.COLOR_HSV2BGR))

def sample_sorter(mean_hsv) -> int:
    # first chech black
    if mean_hsv[2] < 40:
        return 0
    # elif mean_hsv[2] < 200:
    #     return 0
    elif abs(mean_hsv[0] - CYAN) < 2:
        return -2
    elif abs(mean_hsv[0] - BLUE) < 2:
        return -1
    elif abs(mean_hsv[0] - YELLOW) < 2:
        # print(mean_hsv)
        return 1
    elif abs(mean_hsv[0] - ORANGE) < 2:
        # print(mean_hsv)
        return 2
    logging.warning(f"Colour (HSV) {mean_hsv} not expected")
    return None

# TODO:_Check if yellow tiles were orange last turn, coerce to orange

def get_board_sample(frame): #  -> list[list[tuple(in, int, int)]]
    """
    Gets a sample from the topp of each tile on the board.
    Draws samples if asked
    Returns colours in HSV
    """
    board_sample = []
    draw = False
    for y_board_ind, y_px_ind in enumerate(np.arange(
            BOARD_TL_PX[0] + TILE_WIDTH//4,
            BOARD_TL_PX[0] + TILE_WIDTH//4 + BOARD_DIMS[0]*TILE_WIDTH,
            TILE_WIDTH
    )):
        board_sample.append([])
        for x_board_ind, x_px_ind in enumerate(np.arange(
                BOARD_TL_PX[1] + TILE_WIDTH // 2,
                BOARD_TL_PX[1] + TILE_WIDTH // 2 + BOARD_DIMS[1] * TILE_WIDTH,
                TILE_WIDTH
        )):
            y_px_start = y_px_ind - nonlinear_adjust_vert(y_px_ind)
            test_px = (slice(y_px_start, (y_px_start + 8)), slice(x_px_ind - 16, (x_px_ind + 12)))

            board_sample[-1].append(
                np.mean(
                    frame[test_px]
                    , axis= (0,1)
                )
            )
            # Debug tile positions by drawing a gray box in each tile.
            # frame[test_px] = [0, 0, 199] #board_sample[-1][-1]

    if draw == True:
        showhsv(frame)
        #k = cv.waitKey(0)
        #print(board_sample)
    return board_sample

def visual_test(frame, board_state): #  -> None
    """
    Draws samples on board
    """
    for y_board_ind, y_px_ind in enumerate(np.arange(
            BOARD_TL_PX[0] + TILE_WIDTH//4,
            BOARD_TL_PX[0] + TILE_WIDTH//4 + BOARD_DIMS[0]*TILE_WIDTH,
            TILE_WIDTH
    )):
        for x_board_ind, x_px_ind in enumerate(np.arange(
                BOARD_TL_PX[1] + TILE_WIDTH // 2,
                BOARD_TL_PX[1] + TILE_WIDTH // 2 + BOARD_DIMS[1] * TILE_WIDTH,
                TILE_WIDTH
        )):
            y_px_start = y_px_ind - nonlinear_adjust_vert(y_px_ind)
            test_px = (slice(y_px_start, (y_px_start + 8)), slice(x_px_ind - 16, (x_px_ind + 12)))
            text_pos = (x_px_ind-TILE_WIDTH//3, y_px_start+TILE_WIDTH//2)
            # frame[test_px] = STATE_TO_INVERTED_COLOUR[board_state[y_board_ind][x_board_ind]] #board_sample[-1][-1]
            # print(frame[text_pos])
            magenta = (150,256,256)
            cv.putText(frame, STATE_TO_CHAR[board_state[y_board_ind][x_board_ind]], text_pos, cv.FONT_HERSHEY_SIMPLEX, 1, magenta, 2)
    showhsv(frame)
    k = cv.waitKey(0)
    return

def get_state_from_sample(board_sample_mean):
    board_state = [[sample_sorter(sample) for sample in roworcolidk] for roworcolidk in board_sample_mean]
    return board_state

def get_state_from_frame(frame, draw_debug: bool = False) -> List[List[int]]:
    """
    This is the function you want probably
    Takes the frame (BGR)
    returns board state
    optionally shows visual debug info and pauses
    """
    board_sample_mean_HSV = get_board_sample(frame)
    board_state = get_state_from_sample(board_sample_mean_HSV)
    if draw_debug:
        visual_test(frame, board_state)
    return np.asarray(board_state)

# load a frame from video file:
def main():
    cap = cv.VideoCapture('game_examples/output.avi')
    cap.set(cv.CAP_PROP_POS_FRAMES, 6480)

    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    state = get_state_from_frame(frame)
    board = Board(state)
    print(board)
    visual_test(frame, state)
    print("Done! 😊")

if __name__ == "__main__":
    main()
