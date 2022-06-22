import random
import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys
import collections
import json
import re

def compute_perplexity(dataset, net, device, bsz=1):
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    num_examples, seq_len = dataset.size()
    
    total_unmasked_tokens = 0.
    nll = 0.
    for i in range(num_examples):
        wd = torch.unsqueeze(dataset[i], 0).to(device)
        ut = torch.nonzero(wd).to(device).size(0)
        preds = net(wd)#.to(device)
        preds = preds[:, :-1, :].contiguous().view(-1, net.vocab_size).to(device)
        targets = wd[:, 1:].contiguous().view(-1).to(device)
        l = criterion(preds, targets)
        loss = l
        nll += loss.detach()
        total_unmasked_tokens += ut

    perplexity = torch.exp(nll / total_unmasked_tokens).cpu()
    return perplexity.data

#http://jerinphilip.github.io/posts/sampling-from-distributions.html
def draw_from_dist(probs):
    val = random.random()
    #print('val', val)
    csum = 0
    for i, p in enumerate(probs):
        csum  += p
        #print('csum', csum)
        if csum > val:
            return i

def take_sample_from_softmax(sample):
    indices = []
    wd_length, inv_size = sample.size()
    #print('sample size', sample.size())
    for i in range(wd_length):
        distribution = sample[i,:].detach().cpu().numpy().tolist()
        indices.append(draw_from_dist(distribution))
        #print('indices', indices)
    return indices

def train_lm(dataset, dev, params, net, ix2phone, phone2ix, start_time_for_file_id):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = params['device']
    optimiser = torch.optim.Adam(net.parameters(), lr=params['learning_rate'])
    save_path = 'saved_models/checkpt.pth'
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        net.eval()
    num_examples, seq_len = dataset.size()  
    print('There are', num_examples, 'training examples.')
    
    #prev_perplexity = 1e10
    acc = 0
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        #random.shuffle(dataset)
        net.zero_grad()
        num_correct = 0
        num_wrong = 0
        #for b_idx, (start, end) in enumerate(batches):
        for i in range(num_examples):
            wd = torch.unsqueeze(dataset[i], 0).to(device)
            batches, wd_len = wd.size()
            preds = net(wd)#.to(device)
            preds = preds[:, :-1, :].contiguous().view(-1, net.vocab_size).to(device)
            targets = wd[:, 1:].contiguous().view(-1).to(device)
            l = criterion(preds, targets)
            wd_as_string = ''.join([ix2phone[idx] for idx in dataset[i].detach().cpu().numpy().tolist() if idx not in [0, phone2ix['<s>'], phone2ix['<e>']]])
            #pred_softmaxed1= ''.join([ix2phone[idx] for idx in preds if idx not in [0, phone2ix['<s>']]])
            pred_maxed= ''.join([ix2phone[idx] for idx in torch.argmax(preds, dim=1).detach().cpu().numpy().tolist() if idx not in [0, phone2ix['<s>']]])
            pred_softmaxed= re.sub(r'<e>.*', '', pred_maxed)
            if wd_as_string == pred_softmaxed:
                num_correct += 1
            else:
                num_wrong += 1
                if acc > 0.8:
                    print('word', wd_as_string)
                    print('pred', pred_softmaxed)
                    print()
            #print('targets', targets)
            l.backward(retain_graph=True)
            optimiser.step()
            optimiser.zero_grad()
            ep_loss += l.detach()# + l_reverse.detach() 
        acc = num_correct / (num_correct + num_wrong)
        print('epoch', epoch, 'accuracy', round(acc,3))
        #dev_perplexity = compute_perplexity(dev, net, device)

        #newline = 'epoch ' + str(epoch) + ' loss ' + str(ep_loss) + ' perplexity ' + str(dev_perplexity) + '\n'
        #with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
        #    f.write(newline)
        #print('epoch: %d, loss: %0.2f, time: %0.2f sec, dev perplexity: %0.2f' %
        #      (epoch, ep_loss, time.time()-start_time, dev_perplexity))
        #print()
              
        # stop early criterion, increasing perplexity on dev 
        #if epoch > 10 and dev_perplexity - prev_perplexity > 0.01:
        #    print('Stop early reached')
        #    break
        #else:
        #    prev_perplexity = dev_perplexity
        #    torch.save({'net_state_dict': net.state_dict(),  'optimiser_state_dict': optimiser.state_dict()}, save_path)       

