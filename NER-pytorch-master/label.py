# coding=utf-8
from __future__ import print_function
import optparse
import itertools
from collections import OrderedDict
import loader
import torch
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import pdb
from utils import *
from loader import *
from model import BiLSTM_CRF
t = time.time()
models_path = "models/"

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="dataset/eng.train",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="dataset/eng.testa",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="dataset/eng.testb",
    help="Test set location"
)
optparser.add_option(
    '--test_train', default='dataset/eng.train54019',
    help='test train'
)
optparser.add_option(
    '--score', default='evaluation/temp/score.txt',
    help='score file location'
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="200",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="models/glove.6B.100d.txt",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="1",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--loss', default='loss.txt',
    help='loss file location'
)
optparser.add_option(
    '--name', default='test',
    help='model name'
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
opts = optparser.parse_args()[0]

parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['reload'] = opts.reload == 1
parameters['name'] = opts.name
parameters['char_mode'] = opts.char_mode

parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
use_gpu = parameters['use_gpu']

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)
test_train_sentences = loader.load_sentences(opts.test_train, lower, zeros)


name = parameters['name']
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)
update_tag_scheme(test_train_sentences, tag_scheme)

dico_words_train = word_mapping(train_sentences, lower)[0]

dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_train_data = prepare_dataset(
    test_train_sentences, word_to_id, char_to_id, tag_to_id, lower
)




all_word_embeds = {}
for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), opts.word_dim))

    
def restore_checkpoint(file_name, model, optimizer):
    checkpoint = torch.load(file_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    current_epoch = checkpoint['epochs_finished'] + 1
    other_info = checkpoint['other_info']
    return model, optimizer, current_epoch, other_info

model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'])
                   # n_cap=4,
                   # cap_embedding_dim=10)

file_name = './evaluation/saved_checkpoint_wfeats.txt'
if use_gpu:
    model.cuda()
learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#print(restore_checkpoint(file_name, model, optimizer))
model, optimizer, current_epoch, other_info = restore_checkpoint(file_name, model, optimizer)
print("model loaded")
sys.stdout.flush()

def label(model, datas):
    prediction = []
    for data in datas:
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out, feats = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out, feats = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        for (word, pred_id) in zip(words, predicted_id):
            line = ' '.join([word, id_to_tag[pred_id]])
            prediction.append(line)
        prediction.append('')
    return '\n'.join(prediction)
    #predf = eval_temp + '/labeled_single_out_try.txt'
    
    #print("saving to file...")
    #with open(predf, 'w', encoding = 'utf-8-sig') as f:
        #f.write('\n'.join(prediction))


line_idx = 0
length = loader.get_tot_length('./dataset/new_single_out.txt')
with open('./evaluation/temp/labeled_single_out_try.txt', 'w', encoding = 'utf-8-sig') as f:
    while line_idx< length:
        test_single_out_sentences, new_line_idx = loader.load_sentences2("./dataset/new_single_out.txt", lower, zeros, line_idx)
        print("loaded 300 sentencess "+ str(float(new_line_idx)/length))
        line_idx = new_line_idx
        update_tag_scheme(test_single_out_sentences, tag_scheme)
        test_single_out_data = prepare_dataset(
            test_single_out_sentences, word_to_id, char_to_id, tag_to_id, lower
        )
        print("begin labeling")
        f.write(label(model, test_single_out_data))
        print("saved batch")

print("finish labeling")
#print(time.time()-t)

'''def get_true_length(sentences):
    a = torch.zeros([len(sentences), 1])
    for i in range(len(sentences)):
        a[i] = len(sentences[i])
    return a'''
#print(get_true_length(test_train_sentences))

