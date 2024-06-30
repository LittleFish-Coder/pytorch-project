import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initilization methods
x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros((3, 3))
print(x)
x = torch.rand((3, 3))
print(x)
x = torch.ones((3, 3))
print(x)
x = torch.eye(3, 3)
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(1, 5)).uniform_(0, 1)  # same as torch.rand()
print(x)
x = torch.diag(torch.ones(3))
print(x)

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.dtype)
print(tensor.bool())  # boolean True/False
print(tensor.short())  # int16
print(tensor.long())  # int64   (important)
print(tensor.half())  # float16
print(tensor.float())  # float32    (important)
print(tensor.double())  # float64

# Array to Tensor conversion and vice-versa
import numpy as np

np_array = np.zeros((5, 5))
print(np_array)
tensor = torch.from_numpy(np_array)
print(tensor)
np_array_back = tensor.numpy()
print(np_array_back)
