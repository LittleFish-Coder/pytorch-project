import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
print(z1)
torch.add(x, y, out=z1)
print(z1)
z2 = torch.add(x, y)
print(z2)
z3 = x + y
print(z3)

# Subtraction
z4 = x - y
print(z4)

# Division
z5 = torch.true_divide(x, y)
# z5 = x / y
print(z5)

# Inplace operations
t = torch.zeros(3)
t.add_(x)  # t += x (but not t = t + x)
print(t)

# Exponentiation
z = x.pow(2)  # z = x**2
print(z)

# Simple Comparision
z = x > 0
print(z)
z = z < 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  # 2x3
x3 = x1.mm(x2)
print(x3)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp)
print(matrix_exp.matrix_power(3))

# elementwise multiplication
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
print(tensor1)
tensor2 = torch.rand((batch, m, p))
print(tensor2)
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)
print(out_bmm.shape)

# Example of Broadcasting
x1 = torch.rand((5, 5))
print(x1)
x2 = torch.rand((1, 5))
print(x2)
z = x1 - x2
print(z)

z = x1**x2
print(z)

# Other useful tensor operations
# x = torch.tensor([1, 2, 3])
sum_x = torch.sum(x, dim=0)
print(sum_x)
values, indicies = torch.max(x, dim=0)  # x.max(dim=0)
print(values, indicies)
values, indicies = torch.min(x, dim=0)
print(values, indicies)
abs_x = torch.abs(x)
print(abs_x)
z = torch.argmax(x, dim=0)  # only return the index
print(z)
z = torch.argmin(x, dim=0)  # only return the index
print(z)

# mean_x = torch.mean(x, dim=0)   # mean requires float type
mean_x = torch.mean(x.float(), dim=0)  # mean requires float type
print(mean_x)
z = torch.eq(x, y)
print(z)
sorted_y, indicies = torch.sort(y, dim=0, descending=False)
print(sorted_y, indicies)

# torch clamp
z = torch.clamp(x, min=0, max=10)  # any value < 0 -> 0 / any value > 10 -> 10
print(z)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)  # False
