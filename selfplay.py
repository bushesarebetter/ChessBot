import time
from MCTS import run_mcts, expand, TreeNode

import chess
import chess.pgn
import torch
from processing_tools import FEN_to_TENSOR, parse_winner
import torch.optim as optim
from bot import ChessBot
from collections import deque
import random
import torch.nn.functional as F



REPLAY_BUFFER_SIZE = 100000
SELF_PLAY_GAMES = 10
SIMS = 20
LR = 5e-4
EPOCHS = 1000
BATCH_SIZE = 64

def self_play_game(model, n_simulations=50):
    board = chess.Board()
    root = TreeNode(parent=None, prior_prob=1.0)
    expand(root, board, model)


    game_data = []
    
    while not board.is_game_over():
        fen = board.fen()
        board_tensor = FEN_to_TENSOR(fen)
        
        best_move, pi = run_mcts(board, model, simulations=n_simulations)
        

        start_target, end_target = best_move.from_square, best_move.to_square
        
        game_data.append((board_tensor, (start_target, end_target)))
        
        board.push(best_move)
    
    outcome = parse_winner(board.result())
    
    game_data = [(tensor, targets, outcome) for (tensor, targets) in game_data]

    
    return game_data

def train_on_selfplay(model: ChessBot, optimizer: optim.Adam, batch):
    model.train()

    board_tensors, move_targets, value_targets = zip(*batch)

    states = torch.stack([torch.tensor(t, dtype=torch.float32) for t in board_tensors]).to(device)

    start_targets = torch.tensor([m[0] for m in move_targets], dtype=torch.long).to(device)
    end_targets = torch.tensor([m[1] for m in move_targets], dtype=torch.long).to(device)

    value_targets = torch.tensor(value_targets, dtype=torch.float32).to(device)
    pred_start, pred_end, pred_value = model(states)

    loss_start = F.cross_entropy(pred_start, start_targets)
    loss_end = F.cross_entropy(pred_end, end_targets)
    loss_value = F.mse_loss(pred_value.squeeze(), value_targets)

    loss = loss_start + loss_end + 0.2 * loss_value
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_loop(model, optimizer, replay_buffer, n_epochs=1000, batch_size=64):    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        for _ in range(SELF_PLAY_GAMES):
            game_data = self_play_game(model, n_simulations=SIMS)
            print('Generated game!')
            replay_buffer.extend(game_data)
        
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            
            loss = train_on_selfplay(model, optimizer, batch)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
        
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"chess_model_epoch_{epoch + 1}.pth")

        if (epoch + 1) % 100 == 0:
            print(f"Replay buffer size: {len(replay_buffer)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessBot().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load('chessbot_epoch_85.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optim_state_dict'])


for param_group in optimizer.param_groups:
    param_group['lr'] = LR

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

train_loop(model, optimizer, replay_buffer, n_epochs=EPOCHS, batch_size=BATCH_SIZE)






replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
