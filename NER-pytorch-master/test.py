import torch
import torch.nn as nn
a = torch.zeros([5,1])
for i in range(5):
    a[i] = i
#print(a)

e = nn.Embedding(100,4, padding_idx = 5)
input = torch.LongTensor([1,1,1,1])
#print(e(input))

'''m = nn.ConstantPad1d(2, 3.5)
input = torch.randn(2, 4)
print(input)
print(m(input))'''

'''m = nn.ConstantPad2d((0,0,0,2), 3.5)
input = torch.randn(2, 4)
print(input)
print(m(input))'''
'''
tensor_a = torch.randn(2,3)
tensor_b = torch.randn(2,3)
print(torch.stack((tensor_a,tensor_b),0))


from torch.autograd import Variable
from torch.nn import utils as nn_utils
batch_size = 2
max_length = 3
hidden_size = 2
n_layers =1
 
tensor_in = torch.FloatTensor([[1, 2, 3], [1, 0, 0]]).resize_(2,3,1)
print(tensor_in)
tensor_in = Variable( tensor_in ) #[batch, seq, feature], [2, 3, 1]
seq_lengths = [3,1] # list of integers holding information about the batch size at each sequence step
 
# pack it
pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
print (pack)
# initialize
rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
#print(h0)

#forward
out, _ = rnn(pack, h0)
 
# unpack
unpacked = nn_utils.rnn.pad_packed_sequence(out)
print('111',unpacked)
print(_)'''
'''
tensor_in  = torch.FloatTensor([[[1,2,3,4],[1,2,3,4]], [[5,6,7,8],[5,6,7,8]]])
# 2*2*4
batch_size = 2
max_length = 2
n_layers = 4
hidden_size= 4
tensor_in = Variable( tensor_in ) #[batch, seq, feature], [2, 3, 1]
seq_lengths = [2,1] # list of integers holding information about the batch size at each sequence step
 
# pack it
pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
print(pack)

rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
print(h0)

#forward
out, _ = rnn(pack, h0)
 
# unpack
unpacked = nn_utils.rnn.pad_packed_sequence(out)
print('111',unpacked)'''

import pickle

a = {"a": 1,
        "b": 2,
        "c": 3}
f = open('p.txt','ab')
pickle.dump(a, f)
f.close()
f = open('p.txt','rb')
bb = pickle.load(f)
f.close()
#print(bb)

b = {"d": 4}
fi = open('p.txt','ab')
pickle.dump(b, fi)
fi.close()
fi = open('p.txt','rb') 
cc = pickle.load(fi)
fi.close()
#print(cc)

lis = [5,6,7,8,3,5]

def dd():
    #global c
    c=0
    for i in lis:
        c +=1
        if c%2 ==0:
            print(c)
#dd()
print(lis.index(7))





