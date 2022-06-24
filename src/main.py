import argparse
import numpy as np
import sys
import torch
from evaluate import get_probs
from model import Emb_RNNLM, Feature_RNNLM, Feature_RNN_for_single_seg_prediction
from data_process import get_corpus_data, process_data, process_features
from training import train_lm

DEFAULT_FEATURES_FILE = None
DEFAULT_D_EMB = 26
DEFAULT_D_HID = 26
DEFAULT_NUM_LAYERS = 1
DEFAULT_BATCH_SIZE = 1
#DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 4
DEFAULT_TIED = True
DEFAULT_TRAINING_SPLIT = 80
DEFAULT_DEV = True
DEFAULT_DEVICE = None
DEFAULT_USE_SAVED = True
from datetime import datetime
two_rnns = True
interact_only = True

print('Embedding dimension', DEFAULT_D_EMB)
print('Hidden dimension', DEFAULT_D_HID)
print('Learning rate', DEFAULT_LEARNING_RATE) 

SPECIAL_LABELS = ['<p>', '<s>', '<e>']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Generates a vector embedding of sounds in "
                      "a phonological corpus using a RNN."
    )
    parser.add_argument(
        'input_file', type=str, help='Path to the input corpus file.'
    )
    parser.add_argument(
        'test_file', type=str,
        help='Path to test data file' 
    )
    parser.add_argument(
        'judgement_file', type=str,
        help='Path to output file with word judgements' 
    )
    parser.add_argument(
        'feature_file', type=str, nargs="?",
        help='Path to feature file. If specified, embeddings will be based on '
        'the provided feature file. If not, embeddings will be learned during '
        'training',
        default=DEFAULT_FEATURES_FILE
    )
    parser.add_argument(
        '--d_emb', type=int, help='Number of dimensions for the output embedding.',
        default=DEFAULT_D_EMB
    )
    parser.add_argument(
        '--d_hid', type=int, help='Number of dimensions for the hidden layer embedding.',
        default=DEFAULT_D_HID
    )
    parser.add_argument(
        '--num_layers', type=int, help='Number of layers in the RNN',
        default=DEFAULT_NUM_LAYERS
    )
    parser.add_argument(
        '--batch_size', type=int, help='Batch size.',
        default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        '--learning_rate', type=float, help='Learning rate.',
        default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument(
        '--epochs', type=int, help='Number of training epochs.',
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        '--tied', default=DEFAULT_TIED, help='Whether to use tied embeddings.', 
        action='store_true'
    )
    parser.add_argument(
        '--training_split', type=int, default=DEFAULT_TRAINING_SPLIT,
        help='Percentage of data to place in training set.'
    )
    parser.add_argument(
        '--dev', default=DEFAULT_DEV, 
        help='Trains on all data and tests on a small subset.'
    )
    parser.add_argument(
        '--device', default=DEFAULT_DEVICE, 
        help='cpu or cuda'
    )
    parser.add_argument(
        '--use_saved', default=DEFAULT_USE_SAVED, 
        help='cpu or cuda'
    )
    parser.add_argument('--linear_only', action='store_true',
                    help="Use only a linear layer instead of an RNN (default: False)")

    args = parser.parse_args()
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print('Running on', device)

    if args.use_saved is not None:
        use_saved = True
    else:
        use_saved = False

    raw_data = get_corpus_data(args.input_file)
    inventory, phone2ix, ix2phone, training_data, dev, num_words, max_chars = process_data(raw_data, device, use_saved, dev=args.dev, training_split=args.training_split)
    inventory_size = len(inventory)

    rnn_params = {}
    rnn_params['d_emb'] = args.d_emb
    rnn_params['d_hid'] = args.d_hid
    rnn_params['num_layers'] = args.num_layers
    rnn_params['batch_size'] = args.batch_size
    rnn_params['learning_rate'] = args.learning_rate
    rnn_params['epochs'] = args.epochs
    rnn_params['tied'] = args.tied
    rnn_params['device'] = device
    rnn_params['inv_size'] = inventory_size
    rnn_params['num_words'] = num_words

    start_time = datetime.now().strftime('%Y_%m_%d_%H_%M%S')
    if not args.feature_file:
        RNN1 = Emb_RNNLM(rnn_params).to(device)
        if two_rnns:
            RNN2 = Emb_RNNLM(rnn_params).to(device)
        else:
            RNN2 = None
        print('Fitting embedding model...')
    else:
        features, num_feats = process_features(args.feature_file, inventory)
        #build feature table, to replace embedding table, No grad b/c features are fixed
        feature_table = torch.zeros(inventory_size, num_feats, requires_grad=False)
        for i in range(inventory_size):
            feature_table[i] = torch.tensor(features[ix2phone[i]])
        rnn_params['d_feats'] = num_feats
        #RNN1 = Emb_encoder_decoder(rnn_params).to(device)
        #RNN1 = Feature_RNNLM(rnn_params, feature_table).to(device)
        RNN1 = Feature_RNNWithCells(rnn_params, feature_table).to(device)
        if two_rnns:
            #RNN2 = Emb_encoder_decoder(rnn_params).to(device)
            #RNN2 = Feature_RNNLM(rnn_params, feature_table).to(device)
            RNN2 = Feature_RNNWithCells(rnn_params, feature_table).to(device)
        else:
            RNN2 = None
        print('Fitting feature model...')

    print('Number of parameters for each rnn')
    print(sum(p.numel() for p in RNN1.parameters() if p.requires_grad))
    if two_rnns:
        print(sum(p.numel() for p in RNN2.parameters() if p.requires_grad))
    #train_lm(training_data, dev, rnn_params, RNN1, RNN2, ix2phone, start_time)
    train_lm(training_data, dev, rnn_params, RNN1, RNN2, ix2phone, phone2ix, start_time, max_chars, interact_only)
    RNN1.eval()
    if two_rnns:
        RNN2.eval()

    #get_probs(args.test_file, RNN1, RNN2, phone2ix, args.judgement_file, device)
