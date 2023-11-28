from torch import nn
import torch
import numpy as np
import chess

class EvalConvMetaModel(nn.Module):

    def __init__(self):
        super(EvalConvMetaModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(12, 128, 5, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(2048+5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, meta):
        x = self.conv(x)
        #print(x.shape, meta.shape)
        x = torch.cat((x, meta), 1)
        x = self.linear(x)
        return x


def load_model(model_path="model.pt"):
    model = EvalConvMetaModel()
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    return model

def make_matrix(fen):
    res = [] 
    rows = fen.split('/')
    for row in rows:
        row_list = []
        pieces = row.split(" ", 1)[0]
        for thing in pieces:
            if thing.isdigit():
                row_list += '.' * int(thing)
            else:
                row_list += thing
        res.append(row_list)
    return res

def extract_metadata(fen):
    res = [] 
    data = fen.split(' ')
    
    if data[1][0] == 'w': res.append(1)
    else: res.append(0)

    if "K" in data[2]: res.append(1)
    else: res.append(0)

    if "Q" in data[2]: res.append(1)
    else: res.append(0)

    if "k" in data[2]: res.append(1)
    else: res.append(0)

    if "q" in data[2]: res.append(1)
    else: res.append(0)
        
    return res

def vectorize(fen):
    
    table = {
        '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        'P': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'B': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'R': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'Q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    
        'p': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'b': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'n': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    
    res = []
    for i in make_matrix(fen):
        res.append(list(map(table.get, i)))
    return np.array(res)

def evaluate(fen, model):

    data = vectorize(fen)
    data = torch.tensor(data, dtype=torch.float).permute(2, 0, 1).cpu()
    data = data[None, :]

    meta = extract_metadata(fen)
    meta = torch.tensor(meta, dtype=torch.float).cpu()
    meta = meta[None, :]

    model = model.cpu()

    with torch.no_grad():
        model.eval()
        res = model(data, meta).item()
    return res

def white_to_move(fen):
    return fen.split(' ')[1][0] == 'w'

def make_move(fen, model, debug=False):
    
    best_move = None
    best_score = None

    board = chess.Board(fen)
    is_white = white_to_move(fen)
    
    for move in board.legal_moves:
        board.push(move)
        score = evaluate(board.fen(), model)
        board.pop()

        if debug:
            print(f'{str(move)} - {score}')
        
        if best_score is None:
            best_score = score
            best_move = move
            continue
        
        if is_white and score > best_score:
            best_score = score
            best_move = move
            continue

        if not is_white and score < best_score:
            best_score = score
            best_move = move
            continue

    return best_move
        