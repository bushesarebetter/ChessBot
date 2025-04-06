import chess
import numpy as np
import torch

def convert_pgn_to_fen(game):
    board = chess.Board()
    white_elo = game.headers.get('WhiteElo', None)
    black_elo = game.headers.get('BlackElo', None)
    result = game.headers.get('Result', None)

    data = []
    for move in game.mainline_moves():

        fen = board.fen()
        move_san = board.uci(move)
        data.append({
            "WhiteElo": white_elo,
            "BlackElo": black_elo,
            "Result": result,
            "FEN": fen,
            "Move": move_san
        })
        board.push(move)
    return data
def FEN_to_TENSOR(fen_info):
    board = chess.Board(fen_info)

    piece_plane = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        'P': 0,
        'N': 1,
        'B': 2,
        'R': 3,
        'Q': 4,
        'K': 5,
        'p': 6,
        'n': 7,
        'b': 8,
        'r': 9,
        'q': 10,
        'k': 11,
    }
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece_map[piece.symbol()] if piece.symbol() in piece_map else piece_plane.shape[0]
        piece_plane[idx, row, col] = 1.0

    castling_white = np.zeros((1, 8, 8), dtype=np.float32)
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_white[0, 0, 4] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_white[0, 0, 0] = 1

    castling_black = np.zeros((1, 8, 8), dtype=np.float32)
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_black[0, 7, 4] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_black[0, 7, 0] = 1

    en_passant = np.zeros((1, 8, 8), dtype=np.float32)
    if board.ep_square:
        ep_square = board.ep_square
        row = 7 - (ep_square // 8)
        col = ep_square % 8
        en_passant[0, row, col] = 1.0

    move_count = np.zeros((1, 8, 8), dtype=np.float32)
    moves = min(board.halfmove_clock / 50.0, 1.0)
    move_count.fill(moves)


    turn = np.ones((1, 8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((1, 8, 8), dtype=np.float32)

    tensor = np.concatenate([piece_plane, castling_white, castling_black, en_passant, move_count, turn], axis=0)
    return tensor
def move_to_tensor(move: chess.Move):
    start_tensor = torch.zeros(64, dtype=torch.float32)
    end_tensor = torch.zeros(64, dtype=torch.float32)

    start_tensor[move.from_square] = 1.0
    end_tensor[move.to_square] = 1.0

    return start_tensor, end_tensor
def parse_winner(result):
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
    
def mask_policy_outputs(ps, pe, boards):
    masked_ps = []
    masked_pe = []
    device = ps.device

    for i, board in enumerate(boards):
        legal_start_mask = torch.zeros(64, dtype=torch.float32, device=device)
        legal_end_mask = torch.zeros(64, dtype=torch.float32, device=device)
        for move in board.legal_moves:
            legal_start_mask[move.from_square] = 1.0
            legal_end_mask[move.to_square] = 1.0
        lneg = -1e9
        masked_ps.append(ps[i] + (1 - legal_start_mask) * lneg)
        masked_pe.append(pe[i] + (1 - legal_end_mask) * lneg)
    return torch.stack(masked_ps), torch.stack(masked_pe)
def one_hot_moves(move: chess.Move):
    start = torch.zeros((1, 64), dtype=torch.float32)
    end = torch.zeros((1, 64), dtype=torch.float32)

    start[move.from_square] = 1.0
    end[move.from_square] = 1.0

    return start, end
