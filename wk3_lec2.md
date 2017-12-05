# Random Forest Interpretation (cont.)
- proc_df: `na_dict=nas`: Keep track of any columns with missing data

#### Confidence based on tree variance

- Instead of taking the mean of the prediction of the tree, take s.d.
- If s.d. is high, each tree is giving you a very different prediction for this particular row ==> Relevant undestanding of how confident we are at this prediction
- The relative confidence of predictions - that is, for rows where the trees give very different results, you would want to be more cautious of using those results, compared to cases where they are more consistent.

```python
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
```

- List comprehension in Python: Runs in serial (runs on single core) and can be slow

```python
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
np.mean(preds[:,0]), np.std(preds[:,0])
```
```python
(9.2545995667019501, 0.21277472524565924)
```
- We can speed things  up using parallel processing with `parallel_trees`

```python
def get_preds(t): return t.predict(X_valid)
preds = np.stack(parallel_trees(m, get_preds))
np.mean(preds[:,0]), np.std(preds[:,0])
```
```python
(9.2545995667019501, 0.21277472524565924)
```
==> different trees are giving different estimates this this auction

- Look at confidence interval by group

```python
x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)

x.Enclosure.value_counts().plot.barh();
x.ProductSize.value_counts().plot.barh();
```


#### Feature Importance

- In this random forest, which one is important

```python
fi = rf_feat_importance(m, df_trn)
fi[:10]
```
```python
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
```
- In every dataset: A handful of columns that matter and a lot of them aren't important at all

- Filter out variables with "score" > 0.05 to remove redundant columns and fit the model again. When you remove redundant columns, you also reduce collinearity. Collinearity doesn't reduce your predictive power but the features importance are split between the two variables. The feature importance score is more reliable with redundant columns removed.

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

- Assume we know the parametric form $Y = aX_1 + bX_2 + ...$ saying that the coefficients are the feature importance is very dangerous because there is no guarantee that this is the correct form of the model (unless you have a very good model with great predictive power).

**Interactive Variable**
- When calculating the feature importance using one single variable, it is missing **interaction between variables**. E.g. `SaleDate` and `YearMade` -> `Num_Made_Sold`.
- Leave the original variable after adding the interaction variable, but that collinearity should not impact the predictive power of the model. Add as many as you could actually.

#### Data leakage:
  + A feature of the data becomes available that was not originally intended when the original data was input or when the dataset released. In other words, there’s information about dataset that you have that the client didn’t have at the time the dataset was created.
  + This unintended feature can be surfaced during data exploration and interviews with the data stakeholders.
  + For example, Jeremy worked on predicting successful applications for a university grant program, and he found out a single feature–whether the applications were submitted or not, determined whether a grant was was funded. However, he talked and listened to all the people involved in the dataset’s creation. He discovered due to the fact that it was administratively burdensome to log the data around the grant applications, administrators only entered successful grants into database. To make a valid model, this feature needed to be left out of the analysis.
  + Understanding data leakage is important because either this data leakage feature leads the analyst to make a mistaken conclusion or to build an inaccurate model.
  + Investigating data leakage takes legwork and exploration that may lie beyond the data in front of you. On the other hand, a data leak can used as additional feature to make a better performing model in some situations.


- **ToDo**: Go through the top 5 predictors, and plot out the relationship between them. Is there some noise in the columns that we could fix?  


# Handling large dataset
- Communicate about the problem: Start with identifying your dependent and independent variables
- Star Schema: There is a central transaction table that contains the main table that can be used to join with other meta-data tables to add extra information
- Snow-flakes Schema: Additional information

### Data types
  - In order to limit the amount of info: Create a dict that specifies the data types
  -  `pandas` can actually parse 1.24M rows within 1.5s. Python itself is not fast, but whatever we runs in Python was written in C+, which is faster that what we thought.
  - Use `$shuf` to get a sample of data and find out the data type

  - Small the smallest `intXXX` but take into consideration of the format of the variables (e.g. `int8`, `int32`, `int64`). Smaller data type (**todo** lookup `cindy`) also runs faster.
  - `object`: general data type, slow but use it for variables that contain mixed data types e.g. NaN/boolean

- `dates`: extremely important(!) make sure that your dates do not overlap between the training and the test set. Which timeframe are you training your model on vs. predicting?

```python
# clip all negative sales to zero
# np.clip: Given an interval, values outside the interval are clipped to the interval edges.
np.clip(df_all.unit_sales,0,None)
 ```
 - do not have to run `train_cats` and `apply_cats` because we have no categorical variables

 ```python
 m = RandomForestRegressor(n_estimators=20, min_samples_leaf=, n_jobs=8)
 ```

- `n_jobs` = number of cores your computer has

- Profiling: Add `%prun` to check the code profile to see if there is any specific lines of code that are being repeated all the time and could be avoided re-calculating

- Do not set `oob_score`=True, takes a long time and you already had a validation set with the most recent data


- Why is random forest not helpful?
  + Does not know which store is in the center of the city, or the volatility of gasonline prices
  + Need a lot of data but run into a problem of data from too ago
  + Average them out! But how do we improve the model afterwards?
  + Create a graph comparing the score between the averaging model vs. the tweaked version
  + Create **lottttssssss** columns e.g. specific time periods like holidays, or seasonality

- Another example to learn from: Rossmann Store sales   
- Validation set is extremely important, need one that you know is reliable. The score on the validation set should have a linear relationship, if not y=x, with the score on the test set.
