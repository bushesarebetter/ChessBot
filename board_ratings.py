import chess

center_weights = [
-0.37, -0.25, -0.12, 0.00, 0.00, -0.12, -0.25, -0.37,
-0.25, -0.12, 0.00, 0.25, 0.25, 0.00, -0.12, -0.25,
-0.12, 0.00, 0.25, 0.37, 0.37, 0.25, 0.00, -0.12,
0.00, 0.25, 0.37, 0.50, 0.50, 0.37, 0.25, 0.00,
0.00, 0.25, 0.37, 0.50, 0.50, 0.37, 0.25, 0.00,
-0.12, 0.00, 0.25, 0.37, 0.37, 0.25, 0.00, -0.12,
-0.25, -0.12, 0.00, 0.25, 0.25, 0.00, -0.12, -0.25,
-0.37, -0.25, -0.12, 0.00, 0.00, -0.12, -0.25, -0.37,
]
pieces = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100,
}

def check_isolated_pawn(board: chess.Board, square: int, color: chess.WHITE | chess.BLACK):
    file = square % 8

    has_left_pawn = (file - 1) >= 0 and any(p % 8 == (file - 1) for p in board.pieces(chess.PAWN, color))
    has_right_pawn = (file + 1) <= 7 and any(p % 8 == (file + 1) for p in board.pieces(chess.PAWN, color))

    return not has_left_pawn or has_right_pawn
def rook_open_file(board: chess.Board, color):
    bonus = 0
    for rook_sq in board.pieces(chess.ROOK, color):
        file = chess.square_file(rook_sq)
        if not any(board.pieces(chess.PAWN, chess.WHITE, chess.File(file))) and \
           not any(board.pieces(chess.PAWN, chess.BLACK, chess.File(file))):
            bonus += 0.4
    return bonus
def bishop_pair_bonus(board: chess.Board, color):
    bishops = board.pieces(chess.BISHOP, color)
    if len(bishops) >= 2:
        return 0.5
    return 0
def calculate_mobility(board: chess.Board, color):
    mobility = 0
    for move in board.legal_moves:
        if board.color_at(move.from_square) == color:
            mobility += 0.1
    return mobility
def pawn_structure_check(board: chess.Board, color: chess.WHITE | chess.BLACK):
    penalty = 0
    pawn_squares = board.pieces(chess.PAWN, color)
    file_counts = [0] * 8

    
    for square in pawn_squares:
        file = square % 8 
        file_counts[file] += 1
        if check_isolated_pawn(board, square, color):
            penalty += 0.5

    for count in file_counts:
        if count > 1:
            penalty += (count-1) * 0.3
    
    return penalty

def hanging_pieces_check(board: chess.Board, color):
    penalty = 0
    for square in board.occupied_co[color]:
        piece = board.piece_at(square)

        attackers = board.attackers(not color, square)
        defenders = board.attackers(color, square)

        if len(attackers) > len(defenders):
            piece_value = pieces[piece.piece_type] if piece.piece_type != chess.KING else 0
            penalty += piece_value

    return penalty
def king_safety_check(board: chess.Board, color):
    king_square = board.king(color)
    penalty = 0

    pawn_shield = 0
    king_rank = chess.square_rank(king_square)
    king_file = chess.square_file(king_square)

    for file_offset in [-1, 0, 1]:
        file = king_file + file_offset
        if 0 <= king_file < 8:
            for rank_offset in [1, 2]:
                shield_square = chess.square(file, king_rank + rank_offset * (1 if color == chess.WHITE else -1))
                if board.piece_at(shield_square) == chess.Piece(chess.PAWN, color):
                    pawn_shield += 0.5
    penalty += (3-pawn_shield) * 0.4

    open_file_penalty = 0
    for file_offset in [-1, 0, 1]:
        file = king_file + file_offset
        if 0 <= file < 8:
            if not any(board.pieces(chess.PAWN, color, chess.File(file))):
                if any(board.pieces(pt, not color, chess.File(file)) for pt in [chess.ROOK, chess.QUEEN]):
                    open_file_penalty += 0.8
    penalty += open_file_penalty

    center_dist = max(3.5 - abs(king_file - 3.5), 3.5 - abs(king_rank - 3.5))
    penalty += center_dist * (0.8 if 6 < detect_game_phase(board) < 16 else 0.3) 
 

def detect_game_phase(board: chess.Board):
    phase = 0
    piece_values = {chess.QUEEN: 4, chess.ROOK: 2, chess.BISHOP: 1, chess.KNIGHT: 1}

    for piece, value in piece_values:
        phase += len(board.pieces(piece, chess.WHITE)) * value
        phase += len(board.pieces(piece, chess.BLACK)) * value
    
    return min(phase, 24)
                 
def evaluate_board(board: chess.Board):

    if board.is_checkmate():
        return 9999 if board.turn else -999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    evaluation = 0

    for piece in pieces:
        white = board.pieces(piece, chess.WHITE)
        black = board.pieces(piece, chess.BLACK)
        evaluation += white * pieces[piece] 
        evaluation -= black * pieces[piece]
        evaluation += sum(center_weights[p] for p in white)
        evalaution -= sum(center_weights[p] for p in black)

    evaluation += calculate_mobility(board, chess.WHITE) - calculate_mobility(board, chess.BLACK)
    evaluation += bishop_pair_bonus(board, chess.WHITE) - bishop_pair_bonus(board, chess.BLACK)
    evaluation += rook_open_file(board, chess.WHITE) - rook_open_file(board, chess.BLACK)

    # criticisms
    evaluation -= pawn_structure_check(board, chess.WHITE)
    evaluation += pawn_structure_check(board, chess.BLACK)
    evaluation -= king_safety_check(board, chess.WHITE)
    evaluation += king_safety_check(board, chess.BLACK)
    evaluation -= hanging_pieces_check(board, chess.WHITE)
    evaluation += hanging_pieces_check(board, chess.BLACK)

    return evaluation
