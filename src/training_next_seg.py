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

def compute_perplexity(dataset, net1, net2, device, bsz=1):
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    num_examples, seq_len = dataset.size()
    
    total_unmasked_tokens = 0.
    nll = 0.
    for i in range(num_examples):
        wd = torch.unsqueeze(dataset[i], 0).to(device)
        ut = torch.nonzero(wd).to(device).size(0)
        preds1, _ = net1(wd)#.to(device)
        preds1 = preds1[:, :-1, :].contiguous().view(-1, net1.vocab_size).to(device)
        targets = wd[:, 1:].contiguous().view(-1).to(device)
        l1 = criterion(preds1, targets)
        if net2 is not None:
            preds2, _ = net2(wd)#.to(device)
            preds2 = preds2[:, :-1, :].contiguous().view(-1, net2.vocab_size).to(device)
            l2 = criterion(preds2, targets)
            loss = min([l1, l2])
            #print('loss', loss)
        else:
            loss = l1
        nll += loss.detach()
        total_unmasked_tokens += ut

    perplexity = torch.exp(nll / total_unmasked_tokens).cpu()
    return perplexity.data
    
def train_lm(dataset, params, net1, net2, ix2phone, phone2ix, start_time_for_file_id):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = params['device']
    optimiser1 = torch.optim.Adam(net1.parameters(), lr=params['learning_rate'])
    if net2 is not None:
        optimiser2 = torch.optim.Adam(net2.parameters(), lr=params['learning_rate'])
    save_path = 'saved_models/checkpt.pth'
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        net1.load_state_dict(checkpoint['net1_state_dict'])
        optimiser1.load_state_dict(checkpoint['optimiser1_state_dict'])
        net1.eval()
        if net2 is not None:
            net2.load_state_dict(checkpoint['net2_state_dict'])
            optimiser2.load_state_dict(checkpoint['optimiser1_state_dict'])
            net2.eval()
    num_examples = len(dataset)
    #num_examples, seq_len = dataset.size()  
    print('There are', num_examples, 'training examples.')
    
    prev_perplexity = 1e10
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        net1.zero_grad()
        
        #for b_idx, (start, end) in enumerate(batches):
        for i in range(num_examples):
            wd = torch.unsqueeze(dataset[i], 0).to(device)
            wd_as_string = ''.join([ix2phone[idx] for idx in dataset[i].detach().cpu().numpy().tolist() ])
            print()
            print('wd', wd_as_string)
            batch, wd_len = wd.size()
            predicted_segs1 = []
            for j in range(1, wd_len):
                seg_pred1 = net1(wd[:, :j])
                #seg_pred1 = torch.squeeze(seg_pred1, 0).to(device)
                predicted_seg1 = ix2phone[torch.argmax(seg_pred1, dim=1).detach().cpu().numpy().tolist()[0]]
                print(wd_as_string[:j] + predicted_seg1)
                #print('predicted seg', predicted_seg)
                predicted_segs1.append(predicted_seg1)
                target = wd[:, j].contiguous().view(-1).to(device)
                #print('target', target.size(), 'pred1', seg_pred1.size())
                l1 = criterion(seg_pred1, target)
                l1.backward(retain_graph=True)
                optimiser1.step()
                optimiser1.zero_grad()
                ep_loss += l1.detach() 
            #print('predicted string 1', ''.join([wd_as_string[0]] + predicted_segs1))

