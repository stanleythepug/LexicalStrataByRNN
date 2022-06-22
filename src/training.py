import random
import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys

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
    
def train_lm(dataset, dev, params, net1, net2, ix2phone, start_time_for_file_id):
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
    num_examples, seq_len = dataset.size()  
    print('There are', num_examples, 'training examples.')
    
    prev_perplexity = 1e10
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        #random.shuffle(dataset)
        grp1_wds = []
        grp1_hidden = []
        net1.zero_grad()
        if net2 is not None:
            net2.zero_grad()
            grp2_wds = []
            grp2_hidden = []
        
        #for b_idx, (start, end) in enumerate(batches):
        for i in range(num_examples):
            wd = torch.unsqueeze(dataset[i], 0).to(device)
            (preds1, hidden1) = net1(wd)#.to(device)
            preds1 = preds1[:, :-1, :].contiguous().view(-1, net1.vocab_size).to(device)
            if net2 is not None:
                (preds2, hidden2) = net2(wd)#.to(device)
                preds2 = preds2[:, :-1, :].contiguous().view(-1, net2.vocab_size).to(device)
            targets = wd[:, 1:].contiguous().view(-1).to(device)
            #print('preds1', preds1.size())
            #print('targets', targets.size())
            l1 = criterion(preds1, targets)
            if net2 is not None:
                l2 = criterion(preds2, targets)
            wd_as_string = [ix2phone[idx] for idx in dataset[i].detach().cpu().numpy().tolist() if idx != 0]
            wd_as_string = ''.join([c for c in wd_as_string if c not in ['<s>', '<e>']])
            #print('word', wd_as_string)
            #print('targets', targets)
            if net2 is not None:
                #if wd_as_string == '<s>zyokyo<e>':
                #    print('zyokyo net 1', preds1.detach().cpu().numpy().tolist())
                #    print('zyokyo net 2', preds2.detach().cpu().numpy().tolist())

                if l1 < l2:
                    #print('Adding to grp 1', l1, l2)
                    #margin = l2.detach() - l1.detach()
                    grp1_wds.append(wd_as_string)
                    #grp1_wds.append((wd_as_string, l1.detach(), l2.detach(), margin))
                    #if i < 1000:
                    #    grp1_hidden.append((wd_as_string, preds1.detach().cpu().numpy().tolist()))
                    l1.backward(retain_graph=True)
                    optimiser1.step()
                    optimiser1.zero_grad()
                    ep_loss += l1.detach() 
                else:
                    #print('Adding to grp 2', l1, l2)
                    #margin = l1.detach() - l2.detach()
                    grp2_wds.append(wd_as_string)
                    #grp2_wds.append((wd_as_string, l2.detach(), l1.detach(), margin))
                    #if i < 1000:
                    #    grp2_hidden.append((wd_as_string, preds2.detach().cpu().numpy().tolist()))
                    l2.backward(retain_graph=True)
                    optimiser2.step()
                    optimiser2.zero_grad()
                    ep_loss += l2.detach() 
            else:
                grp1_wds.append((wd_as_string))
                #grp1_wds.append((wd_as_string, l1.detach()))
                #if i < 1000:
                #    grp1_hidden.append((wd_as_string, preds1.detach().cpu().numpy().tolist()))
                    #grp1_hidden.append((wd_as_string))
                l1.backward(retain_graph=True)
                optimiser1.step()
                optimiser1.zero_grad()
                ep_loss += l1.detach() 
        dev_perplexity = compute_perplexity(dev, net1, net2, device)

        newline = 'epoch ' + str(epoch) + ' loss ' + str(ep_loss) + ' perplexity ' + str(dev_perplexity) + '\n'
        with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
            f.write(newline)
        print('epoch: %d, loss: %0.2f, time: %0.2f sec, dev perplexity: %0.2f' %
              (epoch, ep_loss, time.time()-start_time, dev_perplexity))
        print()
              
        # stop early criterion, increasing perplexity on dev 
        if epoch > 3 and dev_perplexity - prev_perplexity > 0.01:
            print('Stop early reached')
            break
        else:
            prev_perplexity = dev_perplexity
            if net2 is not None:
                torch.save({'net1_state_dict': net1.state_dict(), 'net2_state_dict': net2.state_dict(), 'optimiser1_state_dict': optimiser1.state_dict(), 'optimiser2_state_dict': optimiser2.state_dict() }, save_path)       
            else:
                torch.save({'net1_state_dict': net1.state_dict(),  'optimiser1_state_dict': optimiser1.state_dict()}, save_path)       

    if net2 is not None:
        with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
            newline = 'Group 1 words:\n'
            f.write(newline)
        print('Group 1 words:')
        for w in grp1_wds:
            with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
                newline = w + '\n'
                f.write(newline)
            #print(w)
            print(w)
        with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
            newline = 'Group 2 words:\n'
            f.write(newline)
        print('Group 2 words:')
        for w in grp2_wds:
            with open('results/' + start_time_for_file_id + '.txt', 'a') as f:
                newline = w + '\n'
                f.write(newline)
            print(w)
  
        
