import torch

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

print(x.dtype)
print(x)
print(x[0].shape)  # x[0, :] -> 25
# get the first feature of all the examples
print(x[:, 0].shape)  # 10
print(x[2, 0:10])  # get the 1~10 feature of the 3rd example

# indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])  # tensor([2, 5, 8])
x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])  # 2 elements: x[1,4] & x[0,0]

# advanced
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])  # tensor([0, 1, 9])
print(x[(x > 2) & (x < 8)])  # tensor([3, 4, 5, 6, 7])
print(x[x.remainder(2) == 0])  # tensor([0, 2, 4, 6, 8])

# useful
print(torch.where(x > 5, x, x * 2))  # tensor([ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9])
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())  # tensor([0, 1, 2, 3, 4])
print(x.ndimension())  # 1
print(x.numel())  # number of elements in x
