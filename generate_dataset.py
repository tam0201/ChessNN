import os
import chess.pgn
import numpy as np
from state import State


def get_datastet(n_samples=None):
    X, Y = [], []
    gn = 0
    values = {"1/2-1/2": 0, "0-1": -1, "1-0": 1}
    for fn in os.listdir("data"):
        pgn = open(os.path.join("data", fn))
        while 1:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            res = game.headers["Result"]
            if res not in values:
                continue
            value = values[res]
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                ser = State(board).serialize()
                X.append(ser)
                Y.append(value)
            print(f"parsed {gn} games, with {len(X)} examples")
            if n_samples is not None and len(X) > n_samples:
                return X, Y
            gn += 1
        X, Y = np.array(X), np.array(Y)
        return X, Y


if __name__ == "__main__":
    X, Y = get_datastet(250000)
    print(X.shape)
    print(Y.shape)
    np.savez("dataset.npz", X, Y)
