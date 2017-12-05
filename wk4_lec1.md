## How to stack/ensemble model?
### Approach #1 (Bagging)
- Example: You have the RF model and SGM model: Ensemble by taking the **average** of the outputs of the two models
- It will not work if one of the model is much worse

### Approach #2 (Boosting)
- Example: You have the Group Mean model, calculate residuals ($\hat y-y$) or the ratio ($\hat y /y$)
- If your model is uncorrelated with the other models, then the average of the residuals will be zero'd out

- Gradient Boosting is a Random Forest using Boosting instead of Bagging. The performance is similar but the libray `xgboost` is way faster than random forest.

# Random Forest (cont.)

What do we know about random forest so far?
1. Data that you put into the random forest, it needs to be a numerical variable
2. Strings can be turned into categorical, handle missing data using `proc_df`

### Subsampling:

```python
set_rf_sample(20000)
```

A dataset with 1M rows, kick out 20k rows => sub-sample. Take the sub-sample and create a tree from there. We're gonna train until there are $log_2(20k)$ branches - given that it's a binary tree. There will be **20k** leaf nodes since we train until there's one observation/sample.

Less branches and less leaf nodes -> Less effective on its own.

A random forest will be better if:   
1. Each tree has better predictive power  
2. Less correlation between trees (less effective when averaging out)

When sub-sampling, each tree has different subsets, and therefore, less correlation between the trees.

**Choosing a size for subsampling**  
Depending on the speed of the computer, pick the number of sample that returns the result within 10s.
If you got more than 1M rows, definitely need to use sub-sampling. Less than 100k, definitely do not need sub-sampling. Note, you do not need to see the entire dataset to recognize a pattern in the data.

### `min_samples_leaf`:
- minimum number of samples / observations per leaf nodes˛
- If `min_samples_leaf`=2, one less decision to make (so $log_2(20k)-1$ branches) and **10k** leaf nodes.
- trees are less correlated from each other (uncertain?)  
- hyper-parameters: They're part of the models that we created, not created by the model.

### `max_features`:
- randomly picked variables by that proportion
- less correlation between trees

n_jobs = number of GPUs (default=1, should change it to -1)

### Out-of-bag score
- Score ("Metric") from all the observations not in the sample, varied for each tree.
- Work well when dataset is small and we don't have enough observations validation set
- If overfit, OOB score and validation score gets worse
- If doing something that's useful for the training set and not the testing set, OOB score stays unchanged but validation score gets worse  
- Bootstrap=True (with replacement) so usually we will see only about 63.2% of the data

Plot the number of trees, sample size against the RMSE to see if adjustment improves the performance

- High carnality?

## Random Forest Interpretation

**How to calculate feature importance**
- `Price` = dp variable
- `YearMade`, etc. = idp variables
- How to figure out how important `YearMade` is?
- Take the `YearMade` column and randomly shuffle it. Same distribution but no relation to the table at all
- Before: .89, after shuffle, it's now .80, delta = .09
- Do the same thing for `Enclosure`, it goes from .89 to .84, delta = .04


- Categorical Variables Exploration:
  + Create a summary table of average response var (`SalePrice`), predicted values of the response and the standard deviation of the predictions, grouped by the categories
  + Inform us some groups may provide inconsistent product
  + The ratio of the prediction/s.d. --> Larger ratio = less accurate prediction


- The model accuracy may not improve (better status quo) but it definitely gets simpler when we remove less important features.


- Assume we know the parametric form $Y = aX_1 + bX_2 + ...$
Saying the coefficients are the feature importance is very dangerous because there is no guarantee that this is the correct form of the model (unless you have a very good model with great predictive power).

**Interactive Variable**
- When calculating the feature importance using one single variable, it is missing **interaction between variables**. E.g. `SaleDate` and `YearMade` -> `Num_Made_Sold`.
- Leave the original variable after adding the interaction variable, but that collinearity should not impact the predictive power of the model. Add as many as you could actually.

### One-hot Encoding for Categorical Variable
- Instead of assigning an integer to a `categorical`, you want to split a variable into binary-value columns for each of the factor (0/1). E.g. Variable X with levels `[a, b, c]`, we will have 3 extra columns `X_a`, `X_b`, `X_c`, each has 0/1 values.

- Easier to understand feature importance of categorical variables (helping with interpretability).

```python
# any cat variables with less than 7 factors, split into 2 columns
df_trn2, y_trn = proc_df(df_raw, 'SalePrice', max_n_cat=7)
```

- Problem: When test set that has a new category that doesn't exist in the training set -> Ignore?

### Spearman Rho (Rank Correlation)

```python
from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
```

- The rank correlation: Good way to see if the two variables are correlated, linear or not.
- Dendogram: Things that're closer together, are more correlated

![](/Users/ThyKhueLy/Downloads/spearmanr.png)

- Create a baseline rf model, find its oob_score.
- Remove each variable for each pair at a time, compare the oob_score, drop the one with higher resulting oob_score.

### Partial Dependence

- Look at relationship between YearMade and SalePrice, smooth curve --> Do not do this.

- Partial dependence plot shows a much clearer picture behind the scene, remove the ???. Very powerful and is able to show the relationship between variables.

- ICEplot: Individual conditional expectation plots
  + Run a random forest for each individual year
  + - Holding every other variables constant, replace the variable `YearMade` with a constant value and fit a random forest regression model for each value of `YearMade`
  + Each blue line is a prediction from a random subset and highlighted line is the median
  + y-axis is the change in sale price
  + Showing us the direct relationship between `YearMade` and `SalePrice`, holding everything else constant

![](/Users/ThyKhueLy/Downloads/ICEplot.png)

- Reference: https://towardsdatascience.com/introducing-pdpbox-2aa820afd312

- Outliers in independent variables: If outliers are random, it doesn't affect output. Bad outliers

### Tree Interpreter
- `value` went up in each node: (`value` = average log `SalePrice`)

| | Value |  Change |  
|-------|-----|------|
| All | 10.189 |
| Coupler<=0.5 | | 0.156
| Enclosure<=2 | | -0.395
| ModelID<=4573 | | 0.276
| | 10.226 |

- Bias = 10.189
- Contributions = 0.156, -0.395, 0.276
- Prediction = 10.226

```python
row = X_valid.values[None,0] # first row
prediction, bias, contributions = ti.predict(m, row)
prediction[0], bias[0] # bias is the same for all rows
```
```python
(9.3668993249319534, 10.105252964126896)
```

```python
[o for o in zip(df_keep.columns, df_valid.iloc[0], sorted(contributions[0]))]
```
```python
# Remember: This is for one row. We can do for multiple rows, and find out the average contribution
# Interpreter????
[('YearMade', 1999, -0.53031338456085986),
 ('Coupler_System', nan, -0.1228993248476193),
 ('ProductSize', 'Mini', -0.11538607314124585),
 ('fiProductClassDesc',
  'Hydraulic Excavator, Track - 3.0 to 4.0 Metric Tons',
  -0.059591201061703993),
 ('Hydraulics_Flow', nan, -0.040659986723331577),
 ('ModelID', 665, -0.038347329989878484),
 ('saleElapsed', 7912, -0.028415582683501972),
 ('fiSecondaryDesc', nan, -0.027537282430798404),
 ('Enclosure', 'EROPS', -0.013545814595658134),
 ('fiModelDesc', 'KX1212', -0.0090674027858898658),
 ('SalesID', 4364751, 0.0022037191145487965),
 ('fiModelDescriptor', nan, 0.006248731672960561),
 ('MachineID', 2300944, 0.01019005180744248),
 ('Hydraulics', 'Standard', 0.011294453662335257),
 ('Drive_System', nan, 0.014807800391608117),
 ('ProductGroup', 'TEX', 0.02127075166249117),
 ('saleDay', 16, 0.026031859299696959),
 ('state', 'Ohio', 0.028449381484757418),
 ('saleWeek', 37, 0.055648660270215977),
 ('age', 11, 0.071264334259488971)]
```

```python
# sum of contributions = gap between bias and prediction
>>> contributions[0].sum()
-0.7383536391949419
```

- Use MSE could be a good approach for classification tree. but class entropy is usually preferred.

### Extrapolation
- Key thing that makes ML special
- Only care how well it fits into our validation set -> Is the model good at generalizing?
- We pull out a test set, locked it away
- Secondly, we pull out another validation set, fit the model on the remaining observations (training set), keep testing against the validation set. After all the checking against the validation set, we test it against the test set.
- **Creating an appropriate test set = MOST IMPORTANT THING** in the Machine Learning. Test set = good indicator of how good your model is in production. If test set is well-aligned with the data you put into production, it's all good.  
- Test set you created = Randomly chosen -> Not a good test set in practice.
- Data you have is old (time lag from data collection -> production), things changed as time move on. So we need a test of what happened in the real world, e.g. it takes 3 months for data to go from collection to production.
- Also need to make sure that the validations set to align with the real-world assumption as well, e.g. the validation set also has to be the last 3 months of the training set. But now the test set is 6 months gap vs. the training set (due to the validation set)
  + Have 5 models, run each against the test set 5 times. Plot the evaluation metrics between the validation set vs. the test set. The one with a plot of a 45-degree straight line has a good validation set.


- Between the OOB R^2 score vs. the R^2 of the validation set that was separated from the training set.
  + If OOB is much better than the validation set -> you're not overfitting
  + If OOB is much worse -> overfitting (you have some time-dependent factor or column)
  + OOB using less trees -> worse?


- **How to find a time-dependent variable?**
  + Combine the train and the test set, create a new column ("response variable") that identifies if the observation is in the validation set

  ```python
  df_ext = df_keep.copy()
  df_ext['is_valid'] = 1
  df_ext.is_valid[:n_trn] = 0 # is it in the training set?
  x, y = proc_df(df_ext, 'is_valid')
  ```
  + Run a random forest to see determine if it's in the training set ==> find out which features is important in distinguishing between the the training and test set
  + Feature importance: SalesID, salesElapsed is the most important => What sets the training and test set apart?
  + Drop those columns and try to run rf model again
  + We should see an improvement in R^2 of the validation set if the variables are time-dependent --> Remove those variables



- **How to capture seasonality** (i.e. our training set does not capture the factors that exist in the validation and test set)?  

Todo: Check out the video in the morning for missing explanation
