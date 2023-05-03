import torch


x = torch.tensor([1, 2, 3, 4]).view(2,2)
y = torch.tensor([1, 2, 3, 4]).view(2, 2,1)
print(x)
print(y)
temp = x * y
print(temp)