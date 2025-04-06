import torch
from torch.utils.data import Dataset
from processing_tools import FEN_to_TENSOR, parse_winner, move_to_tensor
import pandas as pd
import chess

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        fen = row['FEN']
        move_uci = row['Move']
        result = row['Result']
        result = parse_winner(result)

        state_np = FEN_to_TENSOR(fen)
        state_tensor = torch.tensor(state_np, dtype=torch.float32)

        move = chess.Move.from_uci(move_uci)
        start_target, end_target = move_to_tensor(move)
        
        result_tensor = torch.tensor(result, dtype=torch.float32)

        return fen, state_tensor, (start_target, end_target), result_tensor