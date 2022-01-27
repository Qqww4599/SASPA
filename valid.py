import torch
from skimage import io
import matplotlib.pyplot as plt
from model import Attention

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', required=True)
parser.add_argument('--valid_dataset_path', required=True, type=str, help='訓練資料集位置')
parser.add_argument('--load_state_dict', type=bool, default=True, help='是否只載入權重，默認載入權重')


arg = parser.parse_args()
valid_ds = arg.valid_dataset_path

def load_model():
    model_path = arg.model_path

    model = Attention(3).cuda() # dim = 3
    if arg.load_state_dict:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load(model_path)
    model.eval()
    return model

model = load_model()


def view_valid_data(ds_path, model_input):
    img = io.imread(ds_path)
    model = model_input.eval()
    out = model(img)

    plt.imshow(out)

if __name__ == '__main__':
    view_valid_data(valid_ds, model)