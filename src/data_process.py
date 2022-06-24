import random
import torch
import os
import json

def get_corpus_data(filename):
    """
    Reads input file and coverts it to list of lists, adding word boundary 
    markers.
    """
    raw_data = []
    file = open(filename,'r')
    for line in file:
        line = line.rstrip()
        line = ['<s>'] + line.split(' ') + ['<e>']
        raw_data.append(line)
    print('lrd', len(raw_data))
    return raw_data

def process_data(string_training_data, device, use_saved, dev=True, training_split=60):
    data_loaded = False
    print('std', len(string_training_data))
    num_words = len(string_training_data)
    # all data points need to be padded to the maximum length
    max_chars = max([len(x) for x in string_training_data])
    print('max_chars', max_chars)
    if use_saved and os.path.exists('inventory.json') and os.path.exists('phone2ix.json') and os.path.exists('ix2phone.json') and os.path.exists('training_data.json'):
        print('Using saved')
        with open('inventory.json', 'r') as f1:
            inventory = json.load(f1)
        with open('phone2ix.json', 'r') as f2:
            phone2ix = json.load(f2)
        with open('ix2phone.json', 'r') as f3:
            ix2phone = json.load(f3)
        with open('training_data.json', 'r') as f4:
            string_training_data = json.load(f4)
        data_loaded = True
    else:
        random.shuffle(string_training_data)
        string_training_data = [
            sequence + ['<p>'] * (max_chars - len(sequence)) 
            for sequence in string_training_data]

        # get the inventory and build both directions of dicts  
        # this will store the set of possible phones
        inventory = list(set(phone for word in string_training_data for phone in word))
    
        # dictionaries for looking up the index of a phone and vice versa
        phone2ix = {p: ix for (ix, p) in enumerate(inventory)}
        ix2phone = {ix: p for (ix, p) in enumerate(inventory)}
        index_of_padded = phone2ix['<p>']
        wrongly_assigned = ix2phone[0]
        index_of_wrongly_assigned = phone2ix[wrongly_assigned]
        phone2ix['<p>'] = 0
        phone2ix[wrongly_assigned] = index_of_padded
        ix2phone[0] = '<p>'
        ix2phone[index_of_padded] = wrongly_assigned
        print(phone2ix)

    as_ixs = [
        torch.LongTensor([phone2ix[p] for p in sequence]).to(device) 
        for sequence in string_training_data
      ]
    print('as_ixs', len(as_ixs))

    if not dev:
        training_data = torch.stack(as_ixs, 0).to(device)
        # simpler make a meaningless tiny dev than to have a different eval 
        # training method that doesn't compute Dev perplexity
        dev = torch.stack(as_ixs[-10:], 0).to(device)
    else:
        #split = len(as_ixs) // training_split
        split = int(len(as_ixs) * 0.6)
        training_data = torch.stack(as_ixs[:split], 0).to(device)
        dev = torch.stack(as_ixs[split:], 0).to(device)

    if not data_loaded:
        with open('inventory.json', 'w') as f5:
            json.dump(inventory, f5)
        with open('phone2ix.json', 'w') as f6:
            json.dump(phone2ix, f6)
        with open('ix2phone.json', 'w') as f7:
            json.dump(ix2phone, f7)
        with open('training_data.json', 'w') as f8:
            json.dump(string_training_data, f8)


    return inventory, phone2ix, ix2phone, training_data, dev, num_words, max_chars

def process_features(file_path, inventory):
    feature_dict = {}

    file = open(file_path,'r')
    header = file.readline()

    for line in file:
        line = line.rstrip()
        line = line.split(',')

        if line[0] in inventory:
            feature_dict[line[0]] = [1. if x=='+' else 0. if x=='0' else -1. for x in line[1:]]
            feature_dict[line[0]] += [0.,0.,0.]

    num_feats = len(feature_dict[line[0]])

    feature_dict['<s>'] = [0. for x in range(num_feats-3)] + [-1.0,-1.0,1.0]
    feature_dict['<p>'] = [0. for x in range(num_feats-3)] + [-1.0,1.0,-1.0]
    feature_dict['<e>'] = [0. for x in range(num_feats-3)] + [1.0,-1.0,-1.0]

    return(feature_dict, num_feats)
