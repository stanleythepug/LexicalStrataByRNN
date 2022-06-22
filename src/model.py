import torch
import torch.nn as nn
import sys
import random

class Feature_RNNWithCells(nn.Module):
    def __init__(self, params, feature_table): #the ith row of the feature table is the set of features for word i in phone2ix
        super(Feature_RNNWithCells, self).__init__()
        self.features = feature_table
        self.vocab_size = params['inv_size']
        self.d_feats = params['d_feats']
        #self.n_layers = params['num_layers']
        #self.d_hid = params['d_hid']
        self.device = params['device']
        self.i2R = nn.RNNCell(self.d_feats, self.vocab_size).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        self.R2o = nn.Linear(self.vocab_size, self.vocab_size).to(self.device) #recurrent to output layer
        self.R2h = nn.Linear(self.vocab_size, self.vocab_size).to(self.device) #recurrent to output layer
        self.softmax = nn.Softmax(dim=1)

    def batch_to_features(self, batch, feature_table):
        batches, seq_len = batch.size()


    def forward(self, batch):
        batches, seq_len = batch.size()
        inventory_size, num_feats = self.features.size()
        output = torch.zeros(1, self.vocab_size)
        full_representation = torch.zeros(batches, seq_len, num_feats, requires_grad=False)#the final will be batch size x seq_len x number of features
        for i in range(batches):
            outputs = []
            for j in range(seq_len):
                full_representation = torch.unsqueeze(self.features[batch[i,j]], 0)
                output = self.i2R(full_representation, output) #By passing output instead of hidden as the second argument, we make this a Jordan rather than Elman net
                #This output of the cell is actually a hidden vector of the dimension the same as the output we passed it
                output = self.R2o(output) #Output has dim vocab size
                hidden = self.R2h(output) #This hidden vector also gets passed back to the cell so the network is a Jordan/Elman network
                output = self.softmax(output)
                output = torch.mul(output, torch.rand(output.size()))
                outputs.append(output)
                output = output + hidden
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden


# single-direction RNN, optionally tied embeddings
class Emb_RNNLM(nn.Module):
    def __init__(self, params):
        super(Emb_RNNLM, self).__init__()
        self.vocab_size = params['inv_size']
        self.d_emb = params['d_emb']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.embeddings = nn.Embedding(self.vocab_size, self.d_emb)
        self.device = params['device']
        
        # input to recurrent layer, default nonlinearity is tanh
        self.i2R = nn.RNN(
            self.d_emb, self.d_hid, batch_first=True, num_layers = self.n_layers
        ).to(self.device)
        # recurrent to output layer
        self.R2o = nn.Linear(self.d_hid, self.vocab_size).to(self.device)
        if params['tied']:
            if self.d_emb == self.d_hid:
                self.R2o.weight = self.embeddings.weight
            else:
                print("Dimensions don't support tied embeddings")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        batches, seq_len = batch.size()
        embs = self.embeddings(batch)
        output, hidden = self.i2R(embs)
        outputs = self.R2o(output)
        outputs = self.softmax(outputs)
        #print('outputs', outputs.size())
        return outputs, hidden
        #return outputs, output, embs
# single-direction RNN, optionally tied embeddings
#
class Emb_encoder_decoder(nn.Module):
    def __init__(self, params):
        super(Emb_encoder_decoder, self).__init__()
        self.vocab_size = params['inv_size']
        self.d_emb = params['d_emb']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.embeddings = nn.Embedding(self.vocab_size, self.d_emb)
        self.device = params['device']
        
        # input to recurrent layer, default nonlinearity is tanh
        self.encoder = nn.RNN(
            self.d_emb, self.d_hid, batch_first=True, num_layers = self.n_layers
        ).to(self.device)
        # recurrent to output layer
        self.R2o = nn.Linear(self.d_hid, self.vocab_size).to(self.device)
        if params['tied']:
            if self.d_emb == self.d_hid:
                self.R2o.weight = self.embeddings.weight
            else:
                print("Dimensions don't support tied embeddings")
        self.decoder = nn.RNNCell(self.vocab_size, self.d_hid).to(self.device)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, batch):
        batches, seq_len = batch.size()
        embs = self.embeddings(batch)
        output, hidden = self.encoder(embs)
        outputs = self.R2o(output)
        outputs = self.softmax(outputs)

        #print('outputs', outputs.size())
        return outputs
        #return outputs, output, embs



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
        self.i2R = nn.RNN(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers, bidirectional=True).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        self.R2o = nn.Linear(self.d_hid*2, self.vocab_size).to(self.device) #recurrent to output layer
        self.softmax = nn.Softmax(dim=2)
        #self.lstm = nn.LSTM(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers).to(self.device)

    def batch_to_features(self, batch, feature_table):
        batches, seq_len = batch.size()

    def forward(self, batch):
        batches, seq_len = batch.size()
        inventory_size, num_feats = self.features.size()
        full_representation = torch.zeros(batches, seq_len, num_feats, requires_grad=False)#the final will be batch size x seq_len x number of features
        for i in range(batches):
            for j in range(seq_len):
                full_representation[i,j,:] = self.features[batch[i,j]]
        output, hidden = self.i2R(full_representation)
        #output, (hidden, _) = self.lstm(full_representation)
        outputs = self.R2o(output)
        outputs = self.softmax(outputs)
        return outputs

class Feature_encoder_decoder(nn.Module):
    def __init__(self, params, feature_table): #the ith row of the feature table is the set of features for word i in phone2ix
        super(Feature_encoder_decoder, self).__init__()
        self.features = feature_table
        self.vocab_size = params['inv_size']
        self.d_feats = params['d_feats']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.device = params['device']
        self.encoder = nn.RNN(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        self.decoder = nn.RNNCell(self.vocab_size, self.d_hid).to(self.device) 
        self.R2o = nn.Linear(self.d_hid, self.vocab_size).to(self.device) #recurrent to output layer
        self.softmax = nn.Softmax(dim=1)

    def batch_to_features(self, batch, feature_table):
        batches, seq_len = batch.size()

    def forward(self, batch):
        batches, seq_len = batch.size()
        inventory_size, num_feats = self.features.size()
        full_representation = torch.zeros(batches, seq_len, num_feats, requires_grad=False)#the final will be batch size x seq_len x number of features
        outputs = []
        for i in range(batches):
            for j in range(seq_len):
                full_representation[i,j,:] = self.features[batch[i,j]]
        _, hidden = self.encoder(full_representation)
        output = torch.zeros((batches, self.vocab_size), requires_grad=False)
        hidden = torch.squeeze(hidden,0)
        #print('vocab size', self.vocab_size, 'd hid', self.d_hid)
        #print('output', output.size(), 'hidden', hidden.size())
        for i in range(batches):
            for j in range(seq_len):
                hidden = self.decoder(output, hidden)
                output = self.R2o(hidden)
                output = self.softmax(output)
                outputs.append(output)
        outputs = torch.stack(outputs, dim=0).permute(1,0,2)
        #print('outputs', outputs.size())
        return outputs


class Feature_RNN_for_single_seg_prediction(nn.Module):
    def __init__(self, params, feature_table): #the ith row of the feature table is the set of features for word i in phone2ix
        super(Feature_RNN_for_single_seg_prediction, self).__init__()
        self.features = feature_table
        self.vocab_size = params['inv_size']
        self.d_feats = params['d_feats']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.device = params['device']
        self.i2R = nn.RNN(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers).to(self.device) #input to recurrent layer, default nonlinearity is tanh
        self.R2o = nn.Linear(self.d_hid, self.vocab_size).to(self.device) #recurrent to output layer

    def batch_to_features(self, batch, feature_table):
        batches, seq_len = batch.size()

    def forward(self, batch):
        batches, seq_len = batch.size()
        inventory_size, num_feats = self.features.size()
        full_representation = torch.zeros(batches, seq_len, num_feats, requires_grad=False)#the final will be batch size x seq_len x number of features
        for i in range(batches):
            for j in range(seq_len):
                full_representation[i,j,:] = self.features[batch[i,j]]
        output, hidden = self.i2R(full_representation)
        #print('output', output.size(), 'hidden', hidden.size())
        outputs = self.R2o(output[:,-1,:])
        #print('output', output.size())
        #print('outputs', outputs.size())
        #print('hidden', hidden.size())
        return outputs


class LexicalEmbeddingRNN(nn.Module):
    def __init__(self, params):
        super(LexicalEmbeddingRNN, self).__init__()
        self.device = params['device']
        self.num_words = params['num_words']
        self.vocab_size = params['inv_size']
        self.d_lex_emb = params['d_lex_emb']
        self.d_emb = params['d_emb']
        self.d_hid = params['d_hid']
        self.lexical_embeddings = nn.Embedding(self.num_words, self.d_lex_emb)
        #self.sos_embedding = nn.Embedding(1, self.d_emb)
        self.rnncell = nn.RNNCell(self.d_hid, self.d_hid)
        self.projection1 = nn.Linear(self.d_lex_emb * 2 , self.d_hid)
        self.projection2 = nn.Linear(self.d_hid, self.d_lex_emb)
        self.projection3 = nn.Linear(self.d_lex_emb, self.vocab_size)

    def forward(self, wd, lex_wd):
        wds, seq_len = wd.size()
        emb = self.lexical_embeddings(lex_wd) #lex_wd is a tensor index of the word.
        #We pass to the model a lexical embedding of the word and an encoding of a start of string symbol
        #On each timestep, the RNN cell takes in as its input the concatenation of two things (following Malouf): the last outputted symbol and the embedding of the lexeme.
        #The input to the hidden layer is the output of the hidden layer in the previous cell.
        #How do we deal with the difference in dimensionality between the embedding of the lexeme (which would seem to need to be large sonce there are so many different words) and the encoding of each segment, of which there are fewer?
        #If the concatentation goes through a linear layer, it should be able to find weights that give each of the lexeme embedding and the input segment the correct relative amount of influence, even if the former is of a higher dimension. 
        input_seg = torch.zeros((1, self.d_lex_emb)) #Use a tensor of zeroes for the embedding of the <sos> symbol
        #print('input seg', input_seg.size())
        lex_emb = torch.squeeze(emb, 0) # size: d_lex_emb
        #print('lex_emb', lex_emb.size())
        outputs = []
        for j in range(seq_len):
            #print(lex_emb.size(), input_seg.size())
            m = torch.cat((lex_emb, input_seg), 1) #size: d_lex_emb + d_lex_emb
            #print('m', m.size())
            m = self.projection1(m) # size: d_hid
            #print('m size', m.size())
            new_hidden = self.rnncell(m)
            #print('new hidden size', new_hidden.size())
            output = self.projection2(new_hidden) #size: d_lex_emb
            #print('output size', output.size()) #size: d_lex_emb
            input_seg = torch.add(lex_emb, output)
            #print('input seg2', input_seg.size())
            output_to_return =  self.projection3(output)
            #print('output to return', output_to_return.size())
            outputs.append(output_to_return)
        word_output = torch.stack(outputs,0) 
        #print('word output dim', word_output.size())
        return word_output.permute(1,0,2) 

