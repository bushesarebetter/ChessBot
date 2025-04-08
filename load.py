import pandas as pd
from processing_tools import convert_pgn_to_fen
import chess.pgn


all_data = []
with open("tatamast25.pgn") as pgn:
    while game := chess.pgn.read_game(pgn):
        game_data = convert_pgn_to_fen(game)
        all_data.extend(game_data)

df = pd.DataFrame(all_data)
df.to_csv('new.csv')