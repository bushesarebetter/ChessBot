from math import sqrt
import torch
import torch.nn.functional as F

import chess
from bot import ChessBot
from processing_tools import FEN_to_TENSOR


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TreeNode:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior_prob

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def get_move_probs(start_logits, end_logits, board: chess.Board):
    start_probs = F.softmax(start_logits, dim=1).squeeze(0)  
    end_probs = F.softmax(end_logits, dim=1).squeeze(0)

    move_probs = {}
    for move in board.legal_moves:
        start_square = move.from_square
        end_square = move.to_square
        prob = start_probs[start_square].item() * end_probs[end_square].item()
        move_probs[move] = prob

    # Normalize
    total = sum(move_probs.values())
    if total > 0:
        for move in move_probs:
            move_probs[move] /= total

    return move_probs

def ucb_score(parent: TreeNode, child: TreeNode, c_puct = 2.0):
    prior = child.prior
    value = child.value()
    return value + c_puct * prior * sqrt(parent.visit_count) / (1 + child.visit_count)
def expand(node: TreeNode, board: chess.Board, model: ChessBot):
    fen = board.fen()
    board_tensor = FEN_to_TENSOR(fen)
    board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    start_logits, end_logits, value = model(board_tensor)

    probs = get_move_probs(start_logits, end_logits, board)
    value = value.item()


    for move, prob in probs.items():
        if move not in node.children:
            node.children[move] = TreeNode(parent=node, prior_prob=prob)
    
    return value
def select_child(node: TreeNode):
    best_score = -float('inf')
    best_move = None
    best_child = None
    for move, child in node.children.items():
        score = ucb_score(node, child)
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child

def select_move(root: TreeNode):
    best_visit = -1
    best_move = None
    for move, child in root.children.items():
        if child.visit_count > best_visit:
            best_visit = child.visit_count
            best_move = move
    return best_move


def run_mcts(board: chess.Board, model: ChessBot, simulations=50, c_puct=1.0):
    root = TreeNode(parent=None, prior_prob=1.0)
    expand(root, board, model)

    for _ in range(simulations):
        node = root
        scratch_board = board.copy()
        path = [node]

        while node.children:
            move, node = select_child(node)
            scratch_board.push(move)
            path.append(node)

        if not scratch_board.is_game_over():
            value = expand(node, scratch_board, model)
        else:
            outcome = scratch_board.result()
            if outcome == '1-0':
                value = 1
            elif outcome == '0-1':
                value = -1
            else:
                value = 0

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value  

    move_visits = {move: child.visit_count for move, child in root.children.items()}
    total_visits = sum(move_visits.values())
    pi = {move.uci(): count / total_visits for move, count in move_visits.items()}

    # Select the best move as the one with the highest visit count.
    best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
    return best_move, pi
