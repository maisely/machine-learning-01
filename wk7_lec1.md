# SGD on MNIST (cont.)

## Defining Logistic Regression Ourselves

Above, we used pytorch's nn.Linear to create a linear layer. This is defined by a matrix multiplication and then an addition (these are also called affine transformations). Let's try defining this ourselves.

```python

def get_weights(*dims):
  # generate random numbers with dim
  return nn.Parameter(torch.randn(*dims)/dims[0])

class LogReg(nn.Module):
    def __init__(self):
        super().__init__() # inherit from the superclass

        # layer 1 weights (randomly created tensor)
        # equivalent to nn.Linear(28*28, 10)
        self.l1_w = get_weights(28*28, 10)

        # layer 1 bias of length 10
        self.l1_b = get_weights(10)

    # when you create a module in PyTorch,
    def forward(self, x):
        x = x.view(x.size(0), -1)

        # call a Linear Layer
        # new notation:  x = (x @ self.l1_w) + self.l1_b
        x = torch.matmul(x, self.l1_w) + self.l1_b

        # call a Non-linear (LogSoftmax) Layer
        x = torch.log(torch.exp(x)/(1 + torch.exp(x).sum(dim=0)))
        return x
```

#### Soft Max
Take something random and transfer it into something always behave like probability

![](/Users/ThyKhueLy/msan621/notes_ml/softmax.png)

Let's create the neural net and its optimizer

```python
net2 = LogReg().cuda() # it's on the gpu

# parameters is the attribute from nn.Module
# it goes through everything,
# find those with type parameters and optimize them
opt=optim.Adam(net2.parameters())
```

### Generator vs. Iterator:

```python
# iterator in pytorch: asking for another minibatch
dl = iter(md.trn_ld) # training data loader
xmb, ymb = next(dl) # call for the next minibatch, return x and y-axis
```
```python
# wrapping a tensor within Variable() to keep track of its derivative
# has to be on gpu, i.e. cudo because net2 is on gpu
vxmb = Variable(xmbd.cuda())

# size of
preds = net2(vxmb).exp() # matrix of prediction of all minibatches
preds = net2.forward(vxmb).exp() # do the exact same thing
```

### Adding layers
Activation layer = Value that is calculated from a layer, it's not a weight. But adding a lot more layers it's becoming less stable at some point.

Reminding that each input is a vector of size $28\times 28$ pixels and our output is of size $10$ (since there are 10 digits: 0, 1, ..., 9).

```python
# original
net = Sequential(
  nn.Linear(28*28, 100)
  m.LogSoftmax()
).cuda()

# has the same performance without the first layer
# but having 2 consecutive linear layers = 1 linear layer
net = Sequential(
  nn.Linear(28*28, 100)
  nn.Linear(100, 10)
  m.LogSoftmax()
).cuda()

# add a non-linear layer ReLU
net = Sequential(
  nn.Linear(28*28, 100)
  nn.ReLU()
  nn.Linear(100, 10)
  m.LogSoftmax()
).cuda()

# add more layers
# linear -> non-linear -> linear -> non-linear -> linear -> final-linear
# equivalent to linear(ReLU(linear(x)))

net = Sequential(
  nn.Linear(28*28, 100)
  nn.ReLU()
  nn.Linear(100, 100)
  nn.ReLU()
  nn.Linear(100, 10)
  m.LogSoftmax()
).cuda()

```

- Hidden Layers: ReLU() or Leakage()
- Final Non-linear Layer
  + if classification: softmax
  + if regression: often sigmoid or have nothing at all for final layers

## Broadcasting and Matrix Multiplication¶

### Element-wise Operation
- for-loop:
  + 10k times slower
  + SIMD: Taking advantage of storage spaces
- tensor + cuda(): even faster

```python
>> a = T([10, 6, -4])
>> b = T([2, 8, 7])

>> a+b

>> (a < b)
>> (a < b)*a
>> (a < b).mean()
```

### Broadcasting

Copying one of my tensor's axis and treat those as if it is the same-rank object as the other object

```python
# a = rank 1 (vector) where a = T([10, 6, -4])
# 0 = rank 0 (scalar)
>> a < 0

# comparing a to [0, 0, 0]
array([ True,  True, False], dtype=bool)
```

```python
>> m = np.array([[1, 2, 3], [4,5,6], [7,8,9]]); m
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

>> m * 2
array([[ 2,  4,  6],
       [ 8, 10, 12],
       [14, 16, 18]])

>> c = np.array([10,20,30])
>> m + c
array([[11, 22, 33],
       [14, 25, 36],
       [17, 28, 39]])

>> c = np.array([10,20,30])
>> np.expand_dims(c,1)

>> np.broadcast_to(c:None)
>> np.broadcast_to(c, (3,3))
array([[10, 20, 30],
       [10, 20, 30],
       [10, 20, 30]])
```
The numpy expand_dims method lets us convert the 1-dimensional array c into a 2-dimensional array (although one of those dimensions has value 1).

### Broadcasting Rules - super handy!
When operating on two arrays, Numpy/PyTorch compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when:
  + they are equal, or
  + one of them is 1

Arrays do not need to have the same number of dimensions. For example, if you have a $256 \times 256 \times 3$ array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

```
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3
```
```python
>> c = np.array([10,20,30])

>> c[None]
array([[10, 20, 30]])

>> c[:,None]
array([[10],
       [20],
       [30]])

>> # outer product
>> c[None] * c[:,None]
array([[100, 200, 300],
       [200, 400, 600],
       [300, 600, 900]])
```

```python
>> m, c
(array([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]), array([10, 20, 30]))

# element-wise Broadcasting
>> m * c
array([[ 10,  40,  90],
       [ 40, 100, 180],
       [ 70, 160, 270]])

# matrix multiplication
>> m @ c
array([140, 320, 500])

>> T(m) @ T(cs)
140
320
500
[torch.LongTensor of size 3]
```
## Training Loop from Scratch

Train a model:
- Take a mini-batch of the training one at a time
- Have to create an iterator


```python
# Our code from above

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.matmul(x, self.l1_w) + self.l1_b
        x = torch.log(torch.exp(x)/(1 + torch.exp(x).sum(dim=0)))
        return x

net2 = LogReg().cuda()
opt=optim.Adam(net2.parameters())

fit(net2, md, epochs=1, crit=loss_fn, opt=opt, metrics=metrics)
```

```python
net2 = LogReg().cuda()
loss_fn=nn.NLLLoss()
learning_rate = 1e-3
optimizer=optim.Adam(net2.parameters(), lr=learning_rate)

dl = iter(md.trn_dl) #Data loader
x, y = next(dl) # get y and x

# forward pass - the 2 codes are equivalent
y_pred = net2(Variable(x).cuda())
y_pred = net2.forward(Variable(x).cuda())

# compute loss
# Compute and print loss.
# Variable(y).cuda() is the actual loss
loss = loss_fn(y_pred, Variable(y).cuda())
print(loss.data)
```

Pytorch has an automatic differentiation package (autograd) that takes derivatives for us, so we don't have to calculate the derivative ourselves! We just call .backward() on our loss to calculate the direction of steepest descent (the direction to lower the loss the most).

```python
# Before the backward pass, use the optimizer object to
# zero all of the gradients for the variables it will update (which are the learnable weights
# of the model)
optimizer.zero_grad()

# Backward pass: compute gradient of the loss with respect to model parameters, ie.calculate derivative
loss.backward()

# Calling the step function on an Optimizer makes an update to its
# parameters
optimizer.step()
```
let's make another set of predictions and check if our loss
is lower:

```python
x, y = next(dl)
y_pred = net2.forward(Variable(x).cuda())

loss = loss_fn(y_pred, Variable(y).cuda())
print(loss.data)
```

Looping / Training:
```python
for t in range(100):
    x, y = next(dl)
    y_pred = net2.forward(Variable(x).cuda())
    loss = loss_fn(y_pred, Variable(y).cuda())

    if t % 10 == 0:
        accuracy = np.sum(to_np(y_pred).argmax(axis=1) == to_np(y))/len(y_pred)
        print("loss: ", loss.data[0], "\t accuracy: ", accuracy)

    optimizer.zero_grad() # have to zero out the derivative
    loss.backward()
    optimizer.step()
```

backward / back propagation:
```
f(g(x)) --backward--> g'(u)f'(x) for u=f(x) i.e. chain rule
```

stream processing / iterator / generator <-> command-line pipe

##########################

In PyTorch, we can define something as:
The purpose

  ```python
  d = Dataset()
  # we can take length
  len(d)
  # we can index
  d[i]
  # we can call dataloader, shuffle and create `bs` minibatches
  dl = DataLoader(d, shuffle=True, bs=64)
  ```
The purpose of shuffling is to make the minibatches as different as possible to avoid overfitting

### PyTorch Variable

A PyTorch variable is a wrapper around a PyTorch Tensor represents a node in a computational graph. If x is a PyTorch variable:

- x.data is a Tensor giving its value
- x.grad is another Variable holding the gradient of x with respect to some scalar value

```python
net2 = LogReg().cuda()
loss=nn.NLLLoss()
learning_rate = 1e-3
optimizer=optim.Adam(net2.parameters(), lr=learning_rate)

for epoch in range(1):
    losses=[]
    dl = iter(md.trn_dl)
    for t in range(len(dl)):
        # Forward pass: compute predicted y by passing x to the model.
        xt, yt = next(dl)
        y_pred = net2.forward(Variable(xt).cuda())

        # Compute and print loss.
        l = loss(y_pred, Variable(yt).cuda())
        losses.append(l)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        l.backward()
        # print(loss.data)

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    val_dl = iter(md.val_dl)
    val_scores = [score(*next(val_dl)) for i in range(len(val_dl))]
    print(np.mean(val_scores))
```

### Stochastic Gradient Descent

This code is very similar to the `fit` function is fastai

```python
net2 = LogReg().cuda()
loss_fn=nn.NLLLoss()
lr = 1e-2
w , b = net2.l1_w,net2.l1_b # instantiate the weight & bias matrix

# loop through each epoch, then loop through each mini-batch
for epoch in range(1):
    losses=[]
    dl = iter(md.trn_dl)
    for t in range(len(dl)):
        # Forward pass
        # compute predicted y by passing x to the model.
        xt, yt = next(dl)
        y_pred = net2.forward(Variable(xt).cuda())

        # Compute and print loss.
        l = loss(y_pred, Variable(yt).cuda())
        losses.append(loss)

        # Before the backward pass
        # zero the gradients for all of the parameters
        if w.grad is not None:
            w.grad.data.zero_()
            b.grad.data.zero_()

        # Backward pass
        # compute gradient of the loss with respect to model parameters
        l.backward()
        # Stochastic Gradient Descent with 2 lines of codes
        w.data -= w.grad.data * lr
        b.data -= b.grad.data * lr

    val_dl = iter(md.val_dl)
    val_scores = [score(*next(val_dl)) for i in range(len(val_dl))]
    print(np.mean(val_scores))
```

- When you're reaching the minimum, it's good to decrease the learning rate so that it doesn't oscillate

### Regularization / Weight Decay

- How to avoid overfitting? Regularization, can we encourage the weight to be close to zero?
  ==> Add to the loss function L2 = sum of squared coefficients or L1 = sum of absolute coefficients
- Example: a is between 1e-6 and 1e-4
  + L2: loss = 1/n * sum(wX - y)^2 + a*sum(w^2)
  + L1: loss = 1/n * sum(wX - y)^2 + a*sum(|w|)

- Methods:
1. Change our training loop so that it can include the L1/L2 regularization
2. Weight Decay: 2a*sum(w)

```python
loss=nn.NLLLoss()
metrics=[accuracy]
opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9,
  weight_decay = 1e-6)

```

- Validation is better because regularization pushes parameters to zero if they aren't useful enough
