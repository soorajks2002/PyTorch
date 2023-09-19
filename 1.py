import torch

# doesn't intitalize values, just takes the values present in the memory
# to initialize the tensor
x = torch.empty(2, 3, 2, 4)

print(x.shape)
print(x)

# initialize tensor with random values
x = torch.rand(2, 2)

print(x.shape)
print(x)

# initialize tensor with zeros
x = torch.zeros(2, 2)

print(x.shape)
print(x)

# initialize tensor with ones
x = torch.ones(2, 2)

print(x.shape)
print(x)


# initialize tensor with zeros, but INTEGER values
x = torch.zeros(2, 2, dtype=torch.int)

print(x.shape)
print(x)


# initialize tensor with random INTEGER values
x = torch.randint(0, 2, (2, 2), dtype=torch.int)

print(x.shape)
print(x)


# cusstom data to tensor
x = torch.tensor([1, 2, 3, 4])

print(x.shape)
print(x)
