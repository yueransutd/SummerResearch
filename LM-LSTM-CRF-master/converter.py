from __future__ import print_function, division
import os
import re
import codecs
import unicodedata
from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes
import model
import string
import random
import numpy as np
import torch

dic={}
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters + " .,;'-"
    )


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []

    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    
    for i, s in enumerate(sentences):
        
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
                
        else:
            raise Exception('Unknown tagging scheme!')

def load_back(path):
    oup=""
    with open(path, 'r') as f:
        for line in f.read().split('\n'):
            line = line[2:-2]
            word = line.split("', '")
            if word[0][0:2]=="'s":
                lis = word[0].split(", '")
                lis[0] = lis[0][:-1]
                oup+=(" ".join(lis)+" "+ word[1]+" "+ word[2])
            else:   
                oup+=" ".join(word)
                if word[0]=="." and word[1]==".":
                    oup+="\n"
            oup+="\n"
    return oup



train_sentences = load_sentences(path="./data/eng.train.txt", lower=True, zeros=True)
update_tag_scheme(train_sentences, tag_scheme="iobes")

oup = load_back('./data/train_data.txt')
with open('./data/train_data2.txt', 'w') as f:
    f.write(oup)


