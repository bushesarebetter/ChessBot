import math
from board_ratings import evaluate_board
import chess


def order_moves(board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: (
        board.is_capture(move) * 100, 
        board.gives_check(move) * 10,  
        (move.promotion is not None) * 50  
    ), reverse=True)
    return moves

def minimax(board: chess.Board, depth, alpha, beta, maximizing_player=True):

    if depth == 0 or board.is_game_over():
        eval = evaluate_board(board)

        if board.is_checkmate():
            eval = -math.inf if maximizing_player else math.inf
        elif board.is_stalemate():
            eval = 0
        print(eval)
        return eval, None
    
    best_move = None
    ordered_moves = order_moves(board)

    if maximizing_player:
        max_eval = -math.inf
        if not ordered_moves:
            return evaluate_board(board), None 
        for move in ordered_moves:
            board.push(move)

            eval, _ = minimax(board, depth-1, alpha, beta, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        if not ordered_moves:
            return evaluate_board(board), None 
        for move in ordered_moves:
            board.push(move)
            if depth-1 == 0:
                print(move)
            eval, _ = minimax(board, depth-1, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval

                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                return min_eval, best_move
        return min_eval, best_move
        
def get_best_move(board, depth=4):
    _, best_move = minimax(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
    return  best_move or list(board.legal_moves)[0]

