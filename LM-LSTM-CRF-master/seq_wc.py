from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
import model.utils as utils
from model.predictor import predict_wc

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating LM-BLSTM-CRF')
    parser.add_argument('--load_arg', default='./checkpoint/ner/ner_3_cwlm_lstm_crf.json', help='path to arg json')
    parser.add_argument('--load_check_point', default='./checkpoint/ner_3_cwlm_lstm_crf.model', help='path to model checkpoint file')
    parser.add_argument('--gpu',type=int, default=0, help='gpu id')
    parser.add_argument('--decode_type', choices=['label', 'string'], default='string', help='type of decode function, set `label` to couple label with text, or set `string` to insert label into test')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batch')
    parser.add_argument('--input_file', default='./data/single_out.txt', help='path to input un-annotated corpus')
    parser.add_argument('--output_file', default='output.txt', help='path to output file')
    args = parser.parse_args()

    print('loading dictionary')
    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    c_map = checkpoint_file['c_map']
    in_doc_words = checkpoint_file['in_doc_words']
    #if args.gpu >= 0:
        #torch.cuda.set_device(args.gpu)

    # loading corpus
    print('loading corpus')
    with codecs.open(args.input_file, 'r', 'utf-8') as f:
        lines = f.readlines()

    # converting format
    features = utils.read_features(lines)

    # build model
    print('loading model')
    ner_model = LM_LSTM_CRF(len(l_map), len(c_map), jd['char_dim'], jd['char_hidden'], jd['char_layers'], jd['word_dim'], jd['word_hidden'], jd['word_layers'], len(f_map), jd['drop_out'], large_CRF=jd['small_crf'], if_highway=jd['high_way'], in_doc_words=in_doc_words, highway_layers = jd['highway_layers'])

    ner_model.load_state_dict(checkpoint_file['state_dict'])

    # save transition here
    torch.save(str(ner_model.save_transition()), "output_wc.txt")

    pdb.set_trace()


    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()
    else:
        if_cuda = False
 

    decode_label = (args.decode_type == 'label')
    predictor = predict_wc(if_cuda, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'], l_map['<start>'], decode_label, args.batch_size, jd['caseless'])

    print('annotating')
    with open(args.output_file, 'w') as fout:
        predictor.output_batch(ner_model, features, fout)

#python seq_wc.py --load_arg checkpoint/ner/ner_3_cwlm_lstm_crf.json --load_check_point checkpoint/ner_3_cwlm_lstm_crf.model --gpu 0 --input_file ./data/single_out.txt --output_file output.txt
'''
print(predictor.l_map):
{'S-ORG': 0, 'O': 1, 'S-MISC': 2, 'B-PER': 3, 'E-PER': 4, 
'S-LOC': 5, 'B-ORG': 6, 'E-ORG': 7, 'I-PER': 8, 
'S-PER': 9, 'B-MISC': 10, 'I-MISC': 11, 'E-MISC': 12, 
'I-ORG': 13, 'B-LOC': 14, 'E-LOC': 15, 'I-LOC': 16, 
'<start>': 17, '<pad>': 18}
'''    


