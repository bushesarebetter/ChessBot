
import pandas as pd
from processing_tools import FEN_to_TENSOR, convert_pgn_to_fen, move_to_tensor, parse_winner, mask_policy_outputs
from torch.utils.data import DataLoader, random_split
from dataloader import ChessDataset
from bot import ChessBot
import chess.pgn
import torch
import torch.optim as optim
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessBot().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataset = ChessDataset('output.csv')

checkpoint = torch.load('chessbot_epoch_20.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optim_state_dict'])


for param_group in optimizer.param_groups:
    param_group['lr'] = 0.01
batch_size = 64
total_size = len(dataset)
val_size = 262703
train_size = total_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(val_dataset, batch_size, True)
epochs = 100

policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

for epoch in range(31, epochs):
    model.train()
    total_loss = 0.0
    l = 0
    for board_fens, board_tensor, (start_target, end_target), value_target in train_loader:
        
        board_list = [chess.Board(fen) for fen in board_fens]
        board_tensor = board_tensor.to(device)
        start_target = torch.argmax(start_target.to(device).long(), dim=1)
        end_target = torch.argmax(end_target.to(device).long(), dim=1)

        value_target = value_target.to(device).float()

        pred_start, pred_end, pred_value = model(board_tensor)

        pred_start, pred_end = mask_policy_outputs(pred_start, pred_end, board_list)

        loss_start = policy_loss_fn(pred_start, start_target)
        loss_end = policy_loss_fn(pred_end, end_target)

        loss_val = value_loss_fn(pred_value.squeeze(), value_target)

        loss = loss_start + loss_end + 0.2 * loss_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        l += 1
        if l % 1000 == 0: print("1000 training cycles complete")

    if epoch % 10 == 0:
        torch.save({
                'model_state_dict': model.state_dict(), 
                'optim_state_dict': optimizer.state_dict(),
            }, f'chessbot_epoch_{epoch}.pth')
        print(f'Saved model at chessbot_epoch_{epoch}.pth')
    print(f"EPOCH: {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


        