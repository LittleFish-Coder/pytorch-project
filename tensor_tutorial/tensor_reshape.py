import torch

x = torch.arange(9)  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
print(x.dtype)  # int64

# difference is small
x_3x3 = x.view(3, 3)  # tensor stored in contiguous memory
print(x_3x3)
x_3x3 = x.reshape(3, 3)  # safer to use, but with performance lost
print(x_3x3)

# x_3x3 -> [[0,1,2], [3,4,5], [6,7,8]]

y = x_3x3.t()  # [[0,3,6], [4,7,2], [2,5,8]]
print(y)
# print(y.view(9))  # [0,3,6,4,7,2,2,5,8] # error
# print(y.contiguous().view(9)) # need to use contiguous memory
print(y.reshape(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)  # 4x5
print(torch.cat((x1, x2), dim=1).shape)  # 2x10

# flatten
z = x1.view(-1)
print(z.shape)  # torch.Size([10])

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)  # torch.Size([64, 10])

# switch channel
x = torch.rand((batch, 2, 5))
z = x.permute(0, 2, 1)
print(z.shape)  # torch.Size([64, 5, 2])

x = torch.arange(10)  # [10]
print(x.unsqueeze(0).shape)  # torch.Size([1,10])
print(x.unsqueeze(1).shape)  # torch.Size([10,1])

x = x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
z = x.squeeze(1)  # 1x10
print(z.shape)  # torch.Size([1, 10])
