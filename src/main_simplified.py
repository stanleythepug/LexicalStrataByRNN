import argparse
import numpy as np
import sys
import torch
from model_simplified import Emb_RNNLM
from data_process import get_corpus_data, process_data, process_features
from training_simplified import train_lm
from datetime import datetime

DEFAULT_TRAINING_SPLIT = 80
DEFAULT_DEV = True
use_saved = True
interact_only = False
device = "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Generates a vector embedding of sounds in "
                      "a phonological corpus using a RNN."
    )
    parser.add_argument(
        'input_file', type=str, help='Path to the input corpus file.'
    )
    args = parser.parse_args()

    raw_data = get_corpus_data(args.input_file)
    inventory, phone2ix, ix2phone, training_data, dev, num_words, max_chars = process_data(raw_data, device, use_saved)
    inventory_size = len(inventory)

    rnn_params = {}
    rnn_params['d_emb'] = 26
    rnn_params['d_hid'] = 26
    rnn_params['batch_size'] = 1
    rnn_params['learning_rate'] = 0.001
    rnn_params['epochs'] = 4
    rnn_params['tied'] = True
    rnn_params['device'] = "cpu"
    rnn_params['inv_size'] = inventory_size

    start_time = datetime.now().strftime('%Y_%m_%d_%H_%M%S')
    RNN1 = Emb_RNNLM(rnn_params).to(device)
    RNN2 = Emb_RNNLM(rnn_params).to(device)
    print('Fitting embedding model...')

    print('Number of parameters for each rnn')
    print(sum(p.numel() for p in RNN1.parameters() if p.requires_grad))
    print(sum(p.numel() for p in RNN2.parameters() if p.requires_grad))
    train_lm(training_data, dev, rnn_params, RNN1, RNN2, ix2phone, phone2ix, start_time, max_chars, interact_only)
    RNN1.eval()
    RNN2.eval()

