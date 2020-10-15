import torch

x = torch.rand(3, 32, 32)
print(x.size())
print(x.size(0))
a = torch.randint(1, 10, [5])
b = torch.randint(2, 11, [5])
print(a)
print(b)
c = torch.eq(a, b).float().sum().item()
print(type(c))
# print(c.item())
