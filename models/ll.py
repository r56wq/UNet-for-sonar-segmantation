import torch
a  = torch.zeros(1, 1, 572, 572)
#print the last two dimensions of a
print(tuple(a.shape[-2:]))