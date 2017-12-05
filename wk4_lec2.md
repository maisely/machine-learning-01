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
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]

        return DecisionTree(self.x.iloc[idxs], self.y[idxs],
                    idxs=np.array(range(self.sample_sz)), min_leaf=self.min_leaf)

    # return the mean of the predictions of all the trees
    def predict(x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)
```

```python

```
ToDo: - Try to take one of the models built so far, try the interpretation methods we've done. Think how to implement yourself (?)

### Exam:
  + entirely code-based exam. train a random forest, see which features is important.
  + should repeat steps building up a random forest
  + data will need some pre-processing + pandas df
  + anything that Jeremy coded
  + pre-process data that is not in the right format 

- At this point of the course:
  + Should be able to replicate all the steps how to build a random forest
  + Bagging, OOB score, reducing overfitting,
  + Feature importance, which feature we should remove for better interpretation? All other things equal

- Finish random forest by the end of week 5.   
