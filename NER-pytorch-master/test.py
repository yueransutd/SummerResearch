import torch

a = torch.zeros([5,1])
for i in range(5):
    a[i] = i
print(a)