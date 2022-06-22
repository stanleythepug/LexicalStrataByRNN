import torch
import torch.nn as nn
import sys
import random


#single direction RNN with fixed, prespecified features
class Feature_RNNLM(nn.Module):
    def __init__(self, params, feature_table): #the ith row of the feature table is the set of features for word i in phone2ix
        super(Feature_RNNLM, self).__init__()
        self.features = feature_table
        self.vocab_size = params['inv_size']
        self.d_feats = params['d_feats']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.device = params['device']
        #self.encoder = nn.RNN(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers, bidirectional=False).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        self.encoder = nn.LSTM(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers, bidirectional=False).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        #self.decoder = nn.RNN(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers, bidirectional=False).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        self.decoder = nn.LSTM(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers, bidirectional=False).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        self.linear1 = nn.Linear(self.d_hid, self.vocab_size).to(self.device) #recurrent to output layer
        self.softmax = nn.Softmax(dim=2)

    def batch_to_features(self, batch, feature_table):
        batches, seq_len = batch.size()

    def forward(self, batch):
        batches, seq_len = batch.size()
        inventory_size, num_feats = self.features.size()
        full_representation = torch.zeros(batches, seq_len, num_feats, requires_grad=False)#the final will be batch size x seq_len x number of features
        for i in range(batches):
            for j in range(seq_len):
                full_representation[i,j,:] = self.features[batch[i,j]]
        _, (h, c) = self.encoder(full_representation)
        output, (h, c) = self.decoder(full_representation, (h, c))
        outputs = self.linear1(output)
        outputs = self.softmax(outputs)
        return outputs

