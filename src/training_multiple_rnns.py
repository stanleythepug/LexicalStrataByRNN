import random
import numpy as np
import torch
import torch.nn as nn
import time
import collections

def compute_perplexity(dataset, rnns, device, bsz=1):
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    num_examples, seq_len = dataset.size()
    #print('ne sl', num_examples, seq_len)
    
    #batches = [(start, start + bsz) for start in\
    #           range(0, num_examples, bsz)]
    
    total_unmasked_tokens = 0.
    nll = 0.
    #for b_idx, (start, end) in enumerate(batches):
    for i in range(num_examples):
        wd = torch.unsqueeze(dataset[i], 0).to(device)
        ut = torch.nonzero(wd).to(device).size(0)
        targets = wd[:, 1:].contiguous().view(-1).to(device)
        best_rnn = None
        for rnn in rnns.keys():
            preds = rnns[rnn](wd).to(device)
            preds = preds[:, :-1, :].contiguous().view(-1, rnns[rnn].vocab_size).to(device)
            l = criterion(preds, targets)
            if best_rnn == None:
                best_rnn = rnn
                best_loss = l
            else:
                if l < best_loss:
                    best_loss = l
                    best_rnn = rnn
        loss = best_loss
        nll += loss.detach()
        total_unmasked_tokens += ut

    perplexity = torch.exp(nll / total_unmasked_tokens).cpu()
    #print('nll', nll, total_unmasked_tokens, nll / total_unmasked_tokens, perplexity, perplexity.data)
    return perplexity.data
    
def train_lm(dataset, dev, params, rnns, ix2phone, start_time_for_file_id):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = params['device']
    optimizers = {}
    num_rnns = len(rnns)
    for rnn in rnns.keys():
        optimizers[rnn] =  torch.optim.Adam(rnns[rnn].parameters(), lr=params['learning_rate'])
    num_examples, seq_len = dataset.size()  
    print('There are', num_examples, 'training examples.')
    word_groups = collections.defaultdict(lambda: [])
    
    prev_perplexity = 1e10
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        #random.shuffle(dataset)
        word_groups.clear()
        
        #for b_idx, (start, end) in enumerate(batches):
        for i in range(num_examples):
            wd = torch.unsqueeze(dataset[i], 0).to(device)
            targets = wd[:, 1:].contiguous().view(-1).to(device)
            wd_as_string = ''.join([ix2phone[idx] for idx in dataset[i].detach().cpu().numpy().tolist() if idx != 0])
            best_rnn = None
            margin = None
            for rnn in rnns.keys():
                preds = rnns[rnn](wd).to(device)
                preds = preds[:, :-1, :].contiguous().view(-1, rnns[rnn].vocab_size).to(device)
                l = criterion(preds, targets)
                if best_rnn == None:
                    best_rnn = rnn
                    best_loss = l
                else:
                    margin = abs(best_loss.detach() - l.detach())
                    if l < best_loss:
                        best_loss = l
                        best_rnn = rnn
            if num_rnns > 1:
                word_groups[best_rnn].append((wd_as_string, margin))
            loss = best_loss
            loss.backward()
            optimizers[best_rnn].step()
            optimizers[best_rnn].zero_grad()
            ep_loss += loss.detach() 


        dev_perplexity = compute_perplexity(dev, rnns, device)

        newline = 'epoch ' + str(epoch) + ' loss ' + str(ep_loss) + ' perplexity ' + str(dev_perplexity) + '\n'
        with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
            f.write(newline)
        print('epoch: %d, loss: %0.2f, time: %0.2f sec, dev perplexity: %0.2f' %
              (epoch, ep_loss, time.time()-start_time, dev_perplexity))
        print()
              
        # stop early criterion, increasing perplexity on dev 
        if dev_perplexity - prev_perplexity > 0.01:
            print('Stop early reached')
            break
        else:
            prev_perplexity = dev_perplexity

    if num_rnns > 1:
        for rnn in rnns.keys():
            with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
                newline = 'Group ' + str(rnn) + ' words:\n'
                f.write(newline)
            print('Group', rnn, 'words:')
            for w in word_groups[rnn]:
                with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
                    newline = w[0] + '\t' + w[1] + '\n'
                    f.write(newline)
                print(w[0], w[1])
