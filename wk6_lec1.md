# SGD on MNIST

- Most of the datasets can be best modeled by either:
  1. Ensembles of decision trees (rf or xgboost) for structured data. But hard to get extremely good results due to the complex correlation structure / level / cardinality within the dataset.
  2. Multilayered neural networks for unstructured data (audio, vision and NLP)

- `pickle` works with python objects but not optimal for all of them. `feather` is a better way to save for dataframe and works with other frameworks besides python (pickle doesn't).
- jupyter notebook, the last cell as underscore is always available

```python
>>> def load_mnist(filename):
    return pickle.load(gzip.open(filename, 'rb'), encoding='latin-1')

>>> get_data(URL+FILENAME, path+FILENAME)
# destructuring, it returns 3 tuples of train, validation and test data
((x, y), (x_valid, y_valid), _) = load_mnist(path+FILENAME)
```
```python
>>> x.shape
(50000, 784) # where 28^2 = 784
```
### Tensor
- vector 1d array = rank1-tensor
- matrix 2d array = rank2-tensor
- cube 3d array = rank3-tensor
- flatten a tensor means lowering the rank of the tensor
- axis 0 = dim 0 (rows) / axis 1 = dim 1 (columns)

### Normalization
- Random forest completely ignores scale and only care about ranking or order => normalization isn't necessary

### Reshape

```python
# -1 == makes it as big or small as you see fit, herein it's 10000
# practicisng reshaping, reordering
x_imgs = np.reshape(x_valid, (-1,28,28))
x_imgs.shape

(10000, 28, 28)
```

### Slicing tensor

```python
# slicing into a tensor
# 0 = grabbing the first slice, reduce the tensor dim by 1
# 10:14 columns and 10:14 columns
x_imgs[0,10:15,10:15]
```
```python
array([[-0.42452, -0.42452, -0.42452, -0.42452,  0.17294],
       [-0.42452, -0.42452, -0.42452,  0.78312,  2.43567],
       [-0.42452, -0.27197,  1.20261,  2.77889,  2.80432],
       [-0.42452,  1.76194,  2.80432,  2.80432,  1.73651],
       [-0.42452,  2.20685,  2.80432,  2.80432,  0.40176]], dtype=float32)
```

```python
# EXAM MATERIALS!
def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation='none', cmap='gray')
```
plt.imshow = take a numpy array and draw a picture with it


## Neural Network

A **neural network** is an infinitely flexible function, consisting of layers.

A **layer** is matrix multiplication (which is *linear*) followed by a *non-linear* function (the **activation)**.

Stochastic gradient descent algorithm is being used to learn the best parameters to fit the line to the data (note: in the gif, the algorithm is stopping before the absolute best parameters are found). This process is called training or fitting.

Neural networks take this to an extreme, and are infinitely flexible. They often have thousands, or even hundreds of thousands of parameters. However the core idea is the same as above. The neural network is a function, and we will learn the best parameters for modeling our data.

Create a neural network process: Apply linear layer --> Activation layer --> ReLU (remove negative values)

### PyTorch & GPU
- The neural network functionality of PyTorch is built on top of the Numpy-like functionality for fast matrix computations on a GPU.
- Graphical processing units (GPUs) allow for matrix computations to be done with much greater speed, as long as you have a library such as PyTorch that takes advantage of them.


### Neural Net for Logistic Regression in PyTorch

```python
# .cuda() = put it on a GPU
import torch.nn as nn
net = nn.Sequential(
    nn.Linear(28*28, 10),
    nn.LogSoftmax()
).cuda()
```

```python
md = ImageClassifierData.from_arrays(path, (x,y), (x_valid, y_valid))
```

```python
loss=nn.NLLLoss() # loss function: loss entropy (Negative log likelihood)
metrics=[accuracy] # metrics
opt=optim.Adam(net.parameters()) # optimizer
```

- Fitting is the process by which the neural net learns the best parameters for the dataset.

- An epoch is completed once each data sample has been used once in the training loop.

```python
# epochs=1 mean go thru each image once
fit(net, md, epochs=1, crit=loss, opt=opt, metrics=metrics)
```
- Categorical variable loss function: multi-class version=add up the probability of each value of the independent variables.


```python
preds = predict(net, md.val_dl)
preds.shape
(10000, 10) # 10 probabilities for each num

preds.argmax(axis=1)[:5]
array([3, 8, 6, 9, 6])

np.sum(preds == y_valid)/len(preds)
0.91679999999999995
```

```python
def get_weights(*dims):
  # use randn to get well-behavior (mean=0)
  return nn.Parameter(torch.randn(*dims)/dims[0])
```

```python
# LogReg is now subclass of the nn.Module
class LogReg(nn.Module):
    def __init__(self
      # always have to initiate the superclass first
        super().__init__()

        # weight matrix - need to have the appropriate scale
        # [10,000, 28x28] * [28*28, 10] = [10,000, 10]
        # bias will bed added to eve row
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    # forward method in pyTorch gets called when your layer is calculated
    def forward(self, x):
        # view means reshape (flattening)
        x = x.view(x.size(0), -1)

        # x * weight_matrix + bias
        x = torch.matmul(x, self.l1_w) + self.l1_b  # Linear Layer

        # softmax activation layer (returns sth behave like probability)
        #e^x so that the value is between 0 and 1
        # softmax = e^x/sum(e^x)

        x = torch.log(torch.exp(x)/(1 + torch.exp(x).sum(dim=0)))        # Non-linear (LogSoftmax) Layer
        return x
```

### Loss Function
- The loss function or cost function is representing the price paid for inaccuracy of predictions.
### 
