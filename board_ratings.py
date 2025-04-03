import chess
center_weights = [
-0.37, -0.25, -0.12, 0.00, 0.00, -0.12, -0.25, -0.37,
-0.25, -0.12, 0.00, 0.25, 0.25, 0.00, -0.12, -0.25,
-0.12, 0.00, 0.25, 0.37, 0.37, 0.25, 0.00, -0.12,
0.00, 0.25, 0.37, 0.75, 0.75, 0.37, 0.25, 0.00,
0.00, 0.25, 0.37, 0.75, 0.75, 0.37, 0.25, 0.00,
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
    file = chess.square_file(square)
    pawn_files = {chess.square_file(p) for p in board.pieces(chess.PAWN, color)}
    return not ((file - 1 in pawn_files) or (file + 1 in pawn_files))

def rook_open_file(board, color):
    bonus = 0
    for rook_sq in board.pieces(chess.ROOK, color):
        file_idx = chess.square_file(rook_sq)
        no_white_pawns = not any(
            p for p in board.pieces(chess.PAWN, chess.WHITE) 
            if chess.square_file(p) == file_idx
        )
        no_black_pawns = not any(
            p for p in board.pieces(chess.PAWN, chess.BLACK) 
            if chess.square_file(p) == file_idx
        )
        
        if no_white_pawns and no_black_pawns:
            bonus += 0.4 
        elif (color == chess.WHITE and no_white_pawns) or (color == chess.BLACK and no_black_pawns):
            bonus += 0.2
            
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

def hanging_pieces_check(board, color):
    penalty = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != color:
            continue
            
        attackers = board.attackers(not color, square)
        defenders = board.attackers(color, square)
        
        if len(attackers) > len(defenders):
            piece_value = pieces.get(piece.piece_type, 0)
            if piece.piece_type != chess.KING:
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
        if 0 <= file < 8:
            for rank_offset in [1, 2]:
                shield_square = chess.square(file, king_rank + rank_offset * (1 if color == chess.WHITE else -1))
                if board.piece_at(shield_square) == chess.Piece(chess.PAWN, color):
                    pawn_shield += 0.5
    penalty += (3-pawn_shield) * 0.4

    open_file_penalty = 0
    white_pawn_files = {chess.square_file(p) for p in board.pieces(chess.PAWN, chess.WHITE)}
    black_pawn_files = {chess.square_file(p) for p in board.pieces(chess.PAWN, chess.BLACK)}
    for file_offset in [-1, 0, 1]:
        file = king_file + file_offset
        if 0 <= file < 8:
            # Check if file is open (no pawns of either color)
            if file not in white_pawn_files and file not in black_pawn_files:
                # Check if opponent has major pieces on this file
                for square in chess.SQUARES:
                    if chess.square_file(square) == file:
                        piece = board.piece_at(square)
                        if piece and piece.color != color and piece.piece_type in [chess.ROOK, chess.QUEEN]:
                            open_file_penalty += 0.8
                            break
    penalty += open_file_penalty

    game_phase = detect_game_phase(board)
    center_dist = max(3.5 - abs(king_file - 3.5), 3.5 - abs(king_rank - 3.5))
    
    penalty += center_dist * (0.8 if 6 < game_phase < 16 else 0.1) 
    
    return penalty * (1 if 6 < game_phase < 16 else 0.5)

def detect_game_phase(board: chess.Board):
    phase = 0
    phase = sum(len(board.pieces(p, chess.WHITE)) + len(board.pieces(p, chess.BLACK)) * v 
                for p, v in {chess.QUEEN: 4, chess.ROOK: 2, chess.BISHOP: 1, chess.KNIGHT: 1}.items())
    return min(phase, 24)

                 
def evaluate_board(board: chess.Board):

    if board.is_checkmate():
        return -9999 if board.turn else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    evaluation = 0

    for piece in pieces:
        white = board.pieces(piece, chess.WHITE)
        black = board.pieces(piece, chess.BLACK)
        evaluation += len(white) * pieces[piece] 
        evaluation -= len(black) * pieces[piece]
        evaluation += sum(center_weights[p] for p in white)
        evaluation -= sum(center_weights[p] for p in black)

    #evaluation += calculate_mobility(board, chess.WHITE) - calculate_mobility(board, chess.BLACK)
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
