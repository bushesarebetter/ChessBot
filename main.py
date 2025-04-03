
import chess
import chess.pgn
from board_ratings import evaluate_board
from minimax import get_best_move
import time

import minimax
board = chess.Board()

turn = True
while not board.is_checkmate():
    print(board)

    if turn:
        best_move = get_best_move(board)
        board.push(best_move)
    else:
        move = input("What move do you want to play?: ")
        board.push(board.parse_uci(move))
    turn = not turn