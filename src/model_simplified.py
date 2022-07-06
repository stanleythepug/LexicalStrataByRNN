import torch
import torch.nn as nn

# single-direction RNN, optionally tied embeddings
class Emb_RNNLM(nn.Module):
    def __init__(self, params):
        super(Emb_RNNLM, self).__init__()
        self.vocab_size = params['inv_size']
        self.d_emb = params['d_emb']
        self.d_hid = params['d_hid']
        self.embeddings = nn.Embedding(self.vocab_size, self.d_emb)
        self.device = params['device']
        
        # input to recurrent layer, default nonlinearity is tanh
        self.i2R = nn.RNN(
            self.d_emb, self.d_hid, batch_first=True).to(self.device)
        # recurrent to output layer
        self.R2o = nn.Linear(self.d_hid, self.vocab_size).to(self.device)
        if params['tied']:
            if self.d_emb == self.d_hid:
                self.R2o.weight = self.embeddings.weight
            else:
                print("Dimensions don't support tied embeddings")

    def forward(self, batch):
        batches, seq_len = batch.size()
        embs = self.embeddings(batch)
        output, hidden = self.i2R(embs)
        outputs = self.R2o(output)
        return outputs, hidden

