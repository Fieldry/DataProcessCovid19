import os

import torch
from torch.nn.utils.rnn import unpad_sequence
import pandas as pd


def get_los_info(dataset_dir):
    path = os.path.join(dataset_dir, 'los_info.pkl')
    los_info = pd.read_pickle(path)
    return los_info


def unpad_y(y_pred, y_true, lens):
    raw_device = y_pred.device
    device = torch.device("cpu")
    y_pred, y_true, lens = y_pred.to(device), y_true.to(device), lens.to(device)
    y_pred_unpad = unpad_sequence(y_pred, batch_first=True, lengths=lens)
    y_pred_stack = torch.vstack(y_pred_unpad).squeeze(dim=-1)
    y_true_unpad = unpad_sequence(y_true, batch_first=True, lengths=lens)
    y_true_stack = torch.vstack(y_true_unpad).squeeze(dim=-1)
    return y_pred_stack.to(raw_device), y_true_stack.to(raw_device)


def unpad_batch(x, y, lens):
    x = x.detach().cpu()
    y = y.detach().cpu()
    lens = lens.detach().cpu()
    x_unpad = unpad_sequence(x, batch_first=True, lengths=lens)
    x_stack = torch.vstack(x_unpad).squeeze(dim=-1)
    y_unpad = unpad_sequence(y, batch_first=True, lengths=lens)
    y_stack = torch.vstack(y_unpad).squeeze(dim=-1)
    return x_stack.numpy(), y_stack.numpy()