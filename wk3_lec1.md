# Random Forest (cont.)

- Use fastai in your library:
  + copy the directory / symlink to local folder

- Evaluation metrics: RMSLE = sum((log(actuals)-log(preds))^2)

- Pre-processing:
  + dates processing
  + handle missing data

- R-squared $R^2$: Coefficient of determination
  + simplest model: Mean --> your $R^2$ will be $SS_{tot}$
  + % of variation of y deviating from mean explained by the model
  + how good your model is verus the naive model of mean
  + value range: anything less than 1
    + i.e. what if your fitted values are all infinite, your $R^2$ will be 1-infinity


- Overfitting concept: The need for validation set.
  + The **most important** thing whenever you need to build a machine learning
  + Need to create a validation set, that share the same functionality with the test set
  + Example: Create the validation set with the same # of observations with the test set
  + Replication prices in medicine
  + Picking validation set with a *time series* dataset, do not randomly pick any observation, have to pick a certain block of observations / time period
    + ```split_vals(a,n): return a[:n].copy(), a[n:].copy()```

-   Longer than 10s: Inefficient to do anything interactive

### First Random Forest Model
- Tree construction:
  + Create first binary variable: try all possible variables / splits that give us the best score using the weighted RMSE
  + It's always acceptable to split twice, because you can split the leaf node again
  + the `value` in each leaf node here is the average of RMSLE
-  Each tree is grown as follow:
  + Sample N cases at random, but with replacement from the original data, the sample will be the training set for growing the tree
  + For a M input variables, a number m variables where m << M is specified such that at each node, m variables are selected at random out of the M and the best split on these m is used to split the node (`max_features` in scikit)

- The forest error rate depends on two things:
  (1) correlation between any two trees. Increase correlation = increase forest error
  (2) strength of each tree. Increase strength of each tree = reduce forest error
    + A tree with low error rate = A strong classifier

- Reducing / Increasing m reduces / increases both (1) and (2)
==> find optimal *m*
==> use OOB error rate

### Bagging
  - What is a forest?
    + Use a statistical technique called **Bagging**
    + Bag of little bootstrap: Can be used for any kind of model to make it more robust
    + Idea: Averaging model --> What if we create 5 different trees, which each model works correlated with each other and gives a different insight into the model --> **Ensemble** method
      + subsets are selected by random with replacement, i.e. **Bootstrapping** (60% of the rows will be represented, some the rows will be picked multiple times)
      + how to determine which tree is the most important?
      + goal: find a model which variable is important and how they interact with each other. Using random forest or tree space to find the nearest neighbor (different from Euclidean space - *why?*)
    + We are going to build different deep trees, each uses only a subset of the observations. They will surely have errors but these errors will be random and are uncorrelated with each other -> average random error = 0
    + `n_estomators` = **# of trees**
    + the more estimators or trees we have (more semi-random models), the more bagging we have, the more generalized our model
    + adding more trees, it will not get worse but the improvement is slower for sure


    + `graphviz`: Each tree is characterized based on:
        + field <= criteria: feature & criteria per field. criteria can be either categorical or continuous
        + sample: number of samples
        + value: mean value
        + mse: the difference   


- Extremely randomized model: Much faster for training, more randomness but was able to building more trees

### Out-of-bag Score
- Some of the observations didn't get used in each tree --> create a validation set for each tree
- If we have enough trees, a decently big validation set
- The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was not included in training that tree. This allows us to see whether the model is over-fitting, without needing a separate validation set.

+ This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.

+ OOB score usually underestimates our generalized model's accuracy, but more trees = better indicator OOB is

### Hyper-parameter Selection
- How to set our hyper-parameter(s) automatically? --> **Grid search**
  + Run the model on all possible combinations of the hyper-parameters

#### Subsampling:
- Use `set_rf_samples(n)`- Each tree use a different randomly selected subset of `n` observations. The trees have access to the entire dataset.
    ```python
    # manually created function to change rf function
    def set_rf_samples(n):
      forest._generate_sample_indices = (lambda rs, n_samples:
          forest.check_random_state(rs).randint(0, n_samples, n))
    ```    

- Use `reset_rf_samples` to turn it off    
- Fit your model on the reasonably large subset of data, don't have to fit on the entire dataset for a new decimal places

#### `min_samples_leaf`: minimum number of rows in every leaf node
  + stop training your model when your training set has less than `x` observations
  + there are less decision rules for each leaf node; simpler models should generalize better
  + the predictions are made by averaging more rows in the leaf node, resulting in less volatility

#### `max_features`: proportion of features to randomly select from at each split
  + Every binary split, we use a different random subset of columns.
  + Note: You never remove the variables from selection
  + Good value: 1, 0.5, log_2

Tree is infinitely flexible, even for categorical variables, it can split in a way that mix between the levels.
