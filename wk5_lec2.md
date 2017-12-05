- Besides random forest: neural nets + xgboost, highly applicable although they don't extrapolate well.

- Size of validation set:
  + How accurate do I need to know about this algorithm? Have to take into the context and seriousness of the problem i.e. how serious is a false positive/negative? For example, 0.2% difference == 2 more fraud cases not detected. Magic number: 22 observations (when t-distribution is turning into Gaussian distribution).
  + pick enough
  + pick 5 subset data, train models, find find your validation set's mean accuracy and standard error (stdev/(n-1))


- **One/Zero-shot Learning**: Recognize something you have only seen once or never seen before.
- Imbalanced data: Less common class should be picked with a higher probability
- **Todo**: Take the class and write the random forest interpretation to it

- Tree Classifier: Still average the predicted value of each tree instead of taking the max

# Building a Random Forest

```python
class TreeEnsemble():
    # create a tree
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
        np.random.seed(42)
        self.x, self.y,self.sample_sz, self.min_leaf = x, y, sample_sz, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    # take a random subset and run it through a DecisionTree
    def create_tree(self):
        # drawn without replacement sample-size rows
        # maybe done with np.random.choice
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]

        return DecisionTree(self.x.iloc[idxs], self.y[idxs],
                    idxs=np.array(range(self.sample_sz)), min_leaf=self.min_leaf)

    # return the mean of the predictions of all the trees
    def predict(x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)
```

```python
class DecisionTree():
    def __init__(self, x, y, idxs, min_leaf=5):
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
```

```python
m = TreeEnsemble(X_train, y_train, n_trees=10, sample_sz=1000, min_leaf=3)
```

- The decision tree starts out with an infinite score and we find a split that is better than what we have so far.

```python
class DecisionTree():
    def __init__(self, x, y, idxs=None, min_leaf=5):
        if idxs is None: idxs=np.arange(len(y))
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
        self.n,self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])# average predicted values from each tree
        self.score = float('inf')
        self.find_varsplit() # which variable should we split on and at which level?

    # This just does one decision; we'll make it recursive later
    def find_varsplit(self):
        for i in range(self.c):
          self.find_better_split(i) # max_features -> need to alter the code

    # We'll write this later!
    def find_better_split(self, var_idx): pass

    @property
    def split_name(self): return self.x.columns[self.var_idx]

    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]

    # decorator, herein it's said we don't need "()" when calling the method
    @property
    def is_leaf(self): return self.score == float('inf')

    def __repr__(self): # in order to represent the DecisinTree class
        s = f'n: {self.n}; val:{self.val}' # how many rows and avg of response variable
        if not self.is_leaf: # special kind of method - @property
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s
```

```python
>> m = TreeEnsemble(X_train, y_train, n_trees=10, sample_sz=1000, min_leaf=3)
>> m.trees[0]

n: 1000; val:10.079014121552744
```

## Single Branch
### Find best split given variable

- Have to test if it's given us the same result as RandomForestRegressor

```python
ens = TreeEnsemble(x_sub, y_train, 1, 1000)
tree = ens.trees[0]
x_samp,y_samp = tree.x, tree.y # which subset is being used
x_samp.columns
```
Information gain: Entropy, Gini, etc. By splitting the data into 2, how much info do we gain?

```python
def find_better_split(self, var_idx):
    x = self.x.values[self.idxs,var_idx] # values of the specific variable var_idx
    y = self.y[self.idxs]

    # complexity = n^2
    for i in range(1,self.n-1): # go to every row
        lhs = x<=x[i] # LHS: anything < this value (arrays of booleans)
        rhs = x>x[i] # RHS: anything > this value (arrays of booleans)
        if rhs.sum()==0: continue # if everything is false - pass
        lhs_std = y[lhs].std() # stdev
        rhs_std = y[rhs].std() # std
        curr_score = lhs_std*lhs.sum() + rhs_std*rhs.sum()
        # initially, best score = infinity
        # if better, set it as the current best split
        if curr_score<self.score:
            self.var_idx, self.score, self.split = var_idx, curr_score, x[i]
```

### Reducing complexity

```python
tree = TreeEnsemble(x_sub, y_train, 1, 1000).trees[0]
```

```python
def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)

def find_better_split_foo(self, var_idx):
    x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs]

    sort_idx = np.argsort(x)
    sort_y,sort_x = y[sort_idx], x[sort_idx]
    rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
    lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

    for i in range(0,self.n-self.min_leaf-1):
        xi,yi = sort_x[i],sort_y[i]
        lhs_cnt += 1; rhs_cnt -= 1 # add on the LHS, subtract on RHS
        lhs_sum += yi; rhs_sum -= yi
        lhs_sum2 += yi**2; rhs_sum2 -= yi**2
        if i<self.min_leaf or xi==sort_x[i+1]: # make sure not duplicating the calculation of the same value
            continue

        lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
        rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
        curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt

        if curr_score<self.score:
            self.var_idx,self.score,self.split = var_idx,curr_score,xi
```

```python
# internal function vs. global function
DecisionTree.find_better_split = find_better_split_foo
```

## Full Single Tree

```python
def find_varsplit(self):
    for i in range(self.c):
      # go through all the columns, find if there's anything better
      self.find_better_split(i)

    if self.score == float('inf'): return
    x = self.split_col
    lhs = np.nonzero(x<=self.split)[0]
    rhs = np.nonzero(x>self.split)[0]
    self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs]) # decision tree for the left
    self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs]) # decision tree for the right
```

```python
DecisionTree.find_varsplit = find_varsplit
```
![](/Users/ThyKhueLy/Downloads/fullsingletree.png)

```python
>>> tree = TreeEnsemble(x_sub, y_train, 1, 1000).trees[0]; tree
n: 1000; val:10.079014121552744; score:658.5510186055565; split:1974.0; var:YearMade

>>> tree.lhs
n: 159; val:9.660892662981706; score:76.82696888346362; split:2800.0; var:MachineHoursCurrentMeter

>>> tree.rhs
n: 841; val:10.158064432982941; score:571.4803525045031; split:2005.0; var:YearMade

>>> tree.lhs.lhs
n: 150; val:9.619280538108496; score:71.15906938383463; split:1000.0; var:YearMade

>>> tree.lhs.rhs
n: 9; val:10.354428077535193
```
## Prediction

- Prediction for a tree = prediction for every row
- Leading axis for a vector, a matrix(rows)

```python
def predict(self, x): return np.array([self.predict_row(xi) for xi in x])

DecisionTree.predict = predict

def predict_row(self, xi): # recursive
    if self.is_leaf: return self.val
    # the if-else statemetn is like x = do1() if sth else do2()
    t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
    return t.predict_row(xi) # if not a leaf, return a prediction for a lhs/rhs

DecisionTree.predict_row = predict_row
```
# Cython

**Cython** = superset of python. rather than passing to python's interpreter. It passes the code to C -> compiling => faster

```python
%load_ext Cython
----------------------------------------------
%%cython

```
