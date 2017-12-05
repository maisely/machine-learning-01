# Random Forest

### Data Import
Data from [Kaggle Bulldozer](https://www.kaggle.com/c/bluebook-for-bulldozers)
```python
#reload modules automatically before execution
%load_ext autoreload
%autoreload 2

# inline matplotlib
%matplotlib inline
```

- Do not have to follow software engineer approach here
  -- can use `import *` for efficiency

```python
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics
```

#### How to understand a particular Python function?

```python
display
```
```python
<function IPython.core.display.display>
```

```python
# a full documentation
?display

# source code of the function (user-defined functions)
??display
```

To see the full description on the functions:
  + Shift+Tab: short summary (params, return)
  + Shift+Tab x 2: more detailed documentation
  + Shift+Tab x 3: full documentation in new window

#### git ignore `.gitignore`

Path: `$HOME/.config/git/ignore`

A `gitignore` file specifies *intentionally* untracked files that Git should ignore. Each line in a `gitignore` file specifies a pattern.
When deciding whether to ignore a path, Git normally checks `gitignore` patterns from multiple sources, with the following order of precedence, from highest to lowest

```bash
# ignore all the zip files
$ cat .git/info/exclude
*.zip

# ignore all the generated html files
$ cat ./gitignore
*.html
!foo.html # except foo.html
```

#### symlink / alias

#### How to download data to AWS instance?
*(Only work with Firefox)*
- Download link then Cancel
- Developer Toggle Tools > Network tab
- Right-click on domain link > Copy to cURL
- Paste to text editor, remove "--02"
- Copy the edited command to the terminal

```bash
curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/3316/Train.7z?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1509335820&Signature=abDDh82K15LK%2BjZZNaOrzAoiS3XKpAbXZQqjR8YucsXNbOIGuRy6yeaiA7i1tPIrZnvkhPqfCzuEvgWvaN4SMSk8Qrl5KzlUOnQARp0LhqHNbXWEsWH7pF%2BmyBns%2FLkJ6dYHViHQSXgC3ydkj1uP5MbEh8IwZ86hskq0StILMHDvZvTNE83s5JDIU5Quop7xEuhe0tsVDkQLhG7Or5z4KL3XOz2te2ulTeOSpPp54RfP%2BBsIIXS%2FQFZXuimVl3pkKdFuglH8ODDX9O60bCM9wlsbx7fcZXcmK0FWtUPJlTb8QFKrG%2Fv5yNW%2FvTKf1CYHB86pqVogVrBMaCe1lOQQmw%3D%3D' -H 'Host: storage.googleapis.com' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:56.0) Gecko/20100101 Firefox/56.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Referer: https://www.kaggle.com/c/bluebook-for-bulldozers/data' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' > train.csv
```

```python
PATH = "data/bulldozers/"
```

### EDA

The key fields are in train.csv:
- SalesID: the unique identifier of the sale
- MachineID: the unique identifier of a machine. A machine can be sold multiple times
- saleprice: what the machine sold for at auction (only provided in train.csv)
- saledate: the date of the sale


```python
# using pandas
# parse_dates: specified which field is date
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, parse_dates=["saledate"])
```

- Do not have to go over extremely deep analysis for data, it can be done while constructing the model
- But it's still important to look at the data: data types and values
- Note: `pandas` has fairly similar methods to bash command (e.g. head, tail)

```python
# first lines on data
df_raw.head()
```

```python
# transpose: better view, esp. when having lots of cols
# describe: summary statistics for each variables
df_raw.describe(include='all').transpose()
```

- Taking log of `salesprice` - kaggle's evaluation metrics

```python
# can interchangably use between numpy and pandas
df_raw.SalePrice = np.log(df_raw.SalePrice)
```

### Initial Processing

- Two types: Regressor vs. Classification
  + Regressor: Continuous variable (*not just OLS*)
  + Classification: Multi-class discrete variable


- Random Forest:
  + trivial parallelizable: distribute workload across machines
  + Function: `sklearn.ensemble.RandomForestRegressor`
  + `n_jobs`:  number of jobs to run in parallel for both fit and predict


  ```python
  # why negative values???
  # if -1, # of jobs set to # of cores
  n_jobs = -1
  m = RandomForestRegressor(n_jobs=-1)
  ```

  + best techniques, can be applied to a lot of problems (basically a true **free lunch**).

#### Categorical Variables:

First attempt to fit the model:
```python
# drop the response variable
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
```
Return error:
  ```
  ValueError: could not convert string to float: 'Conventional'
  ```

This dataset contains a mix of **continuous** and **categorical** variables.

The following method extracts particular date fields from a complete datetime for the purpose of constructing categoricals.  **You should always consider this feature extraction step when working with date-time.** Without expanding your date-time into these additional fields, you can't capture any **trend/cyclical** behavior as a function of time at any of these granularities (e.g. year, month, quarter, weekend, dow, **holiday**)

```python
add_datepart
<function fastai.structured.add_datepart>

def add_datepart(df, fldname):
    fld = df[fldname]
    targ_pre = re.sub('[Dd]ate$', '', fldname)

    # these are built-in properties of pandas DateTimeProperties object
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = (fld - fld.min()).dt.days
    df.drop(fldname, axis=1, inplace=True)
```
```python
# customized function to add additional cols
add_datepart(df_raw, 'saledate')
```

**Separating continuous variables and categorical in pandas**

```python
train_cats
<function fastai.structured.train_cats>

def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c):
          df[n] = c.astype('category').cat.as_ordered()
```

```python
# UsageBand is a categorical variable in dataset
# get the unique values of UsageBand
>>> df_raw.UsageBand.cat.categories
Index(['High', 'Low', 'Medium'], dtype='object')
```
```python
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
```


`pandas` dataframe has `dtype` of `category`, which object has 2 properties:

+ categories: `pandas.self.cat.categories`
+ ordered: `pandas.self.cat.ordered`


`pandas.Series.cat.categories`: Setting assigns new values to each category (effectively a rename of each individual category).

The assigned value has to be a list-like object. All items must be unique and the number of items in the new categories must be the same as the number of items in the old categories.

```python
df = pd.DataFrame({'A':['a', 'b', 'c', 'a']})
# create categorical Series
df["B"] = df["A"].astype('category')

>>> df.B.cat.cagories
Index(['a', 'b', 'c'], dtype='object')
```

#### Missing Data

```python
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
```

- Replace categories with their numeric code, handle missing continuous values, split dependent variables:
  + Create an indicator column for NA values
  + In the original column, replace with median values
  + It does not affect the overall result, random forests know which variable needs imputation and which doesn't

```python
df, y = proc_df(df_raw, 'SalePrice')
```

```python
# replace cat. with num., imputation
def proc_df(df, y_fld, skip_flds=None, do_scale=False,
            preproc_fn=None, max_n_cat=None, subset=None):

    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    df = df.copy()
    if preproc_fn: preproc_fn(df)
    y = df[y_fld].values
    df.drop(skip_flds+[y_fld], axis=1, inplace=True)


    for n,c in df.items(): fix_missing(df, c, n) # missing data
    if do_scale: mapper = scale_vars(df)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    res = [pd.get_dummies(df, dummy_na=True), y]
    if not do_scale: return res
    return res + [mapper]

# median imputation
def fix_missing(df, col, name):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum():
          df[name+'_na'] = pd.isnull(col)
        df[name] = col.fillna(col.median()) # median imputation
```

#### Save File

Use `feather`, much faster format

```python
os.makedirs('tmp', exist_ok=TRUE)
df_raw.to_feather('tmp/raw')
df_raw = pd.read_feather('tmp/raw')
```

#### Data Partition - Validation Set

```python
m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df, y) # return r.squared
```
==> Problem: overfitting - that leads to a poor MSE due to high model variance

```python
def split_vals(a,n):
  return a[:n].copy(), a[n:].copy()

n_valid = 12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
```

```python
>>> X_train.shape, y_train.shape, X_valid.shape
((389125, 66), (389125,), (12000, 66))
```

### Base Model

```python
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)

m.predict(X_train) # return fitted values

def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)
```

- Can call `self.getattrs()` and `self.hasattr()`

```python
>>> print_score(m) # rmse_train, rmse_test, r.sqrd_train, r.sqrd_test
[0.0902435011024215, 0.2507924131328525, 0.98295721706791372, 0.88767491329235182]
```
- From validation test score: Over-fitting badly
- Solution: Simplify to a single small tree


### Speed things up

```python
# get a subset of the original raw data
df_trn, y_trn = proc_df(df_raw, 'SalePrice', subset=30000)

# within function proc_df, default is subset=None
if subset: df = get_sample(df,subset)

def get_sample(df,n):
    return df.iloc[np.random.permutation(len(df))[:n]].copy()
```

```python
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)
```

```python
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
```


### Single Tree

```python
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
```

```python
>>> print_score(m)
[0.5388850423199081, 0.5748224262205432, 0.40303936660147011, 0.40991390505110342]
```

```python
graphviz.draw_tree(m.estimators_[0], df_trn)
```

```python
# create a bigger tree
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
```

```python
>>> print_score(m)
[6.526751786450488e-17, 0.38473652894699306, 1.0, 0.73565273648797624]
```

Training set looks good. But the validation set is worse than our original model.
==> This is why we need to use **bagging** of multiple trees to get more generalizable results.


### Bagging

Get the predictions for each tree:

```python
m.estimators_ # return a list of trees
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
```

```python
>>> preds[:,0], np.mean(preds[:,0]), y_valid[0]
(array([  9.2103,  10.1659,   9.2103,   9.6486,   9.159 ,   8.9872,   9.5468,   9.1901,   9.159 ,
         10.1659]), 9.4443220929962219, 9.1049798563183568)
```


The shape of this curve suggests that adding more trees isn't going to help us much. By increasing the number of variables / estimators `n_estomators` (20, 40, 80), the `rmse_test` score doesn't significantly increase.

**COMMENT**:

```python
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
[0.10785936183258912, 0.2714731473911183, 0.97608506928894023, 0.86838610856809706]

m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
[0.10217136535646293, 0.2695280876401239, 0.97854088395622285, 0.87026533528275862]

m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
[0.09857776108121721, 0.26662325882251386, 0.98002387097515298, 0.87304668705184318]
```
#### Out-of-bag Score (OOB)

Is our validation set worse than our training set:
  + because we're over-fitting, or
  + because the validation set is for a different time period, or a bit of both?

With the existing information we've shown, we can't tell.

Solution: random forests's **out-of-bag (OOB) error**

The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was *not* included in training that tree.

This allows us to see whether the model is over-fitting, without needing a separate validation set.

This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.
This is as simple as adding one more parameter to our model constructor.

We print the OOB error last in our `print_score` function below.

```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
[0.10198464613020647, 0.2714485881623037, 0.9786192457999483, 0.86840992079038759, 0.84831537630038534]
```
This shows that our validation set time difference is making an impact, as is model over-fitting.
**WHY?**

### Handle Over-Fitting

**Solution (1)**: **Subsampling** - avoid overfitting & speed up analysis
  + Rather than limiting the total amount of data the model can access, limit it to a different random subset per tree
  + Given enough trees, model can still see **all** data but for each tree, it'll be much faster

```python
df_trn, y_trn = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
```

```python
set_rf_samples(20000)

def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))
```

```python
m = RandomForestRegressor(n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
```

```python
CPU times: user 8.08 s, sys: 424 ms, total: 8.51 s
Wall time: 3.21 s
[0.24089575932830098, 0.2768141035742647, 0.87654282167327779, 0.8631564283141987, 0.86607762881070971]
```

Since each additional tree allows the model to see more data, this approach can make additional trees more useful.
```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

[0.22717383030468585, 0.26244270029093636, 0.89115044708468416, 0.87699664163032043, 0.88063003924348748]
```


#### Tree Building Parameters

We revert to using a full bootstrap sample in order to show the impact of other over-fitting avoidance methods.

```python
reset_rf_samples()

def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
```

**Solution (2)**: **Grow our trees less deeply** --> `min_samples_leaf` that we require some minimum number of rows in every leaf node.

This has two benefits:
+ There are less decision rules for each leaf node; simpler models should generalize better
+ The predictions are made by averaging more rows in the leaf node, resulting in less volatility

```python
# set min_sample_leaf=3
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

[0.11545740375724732, 0.23413712659756555, 0.96782354474819665, 0.90209868110901226, 0.90864682465413171]
```

**Solution (3)**: increase the amount of variation amongst the trees by not only use a sample of rows for each tree, but to also using a sample of *columns* for each *split* --> `max_features`, which is the proportion of features to randomly select from at each split.

```python
# add max_features
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

[0.11914559149841383, 0.22708164691801616, 0.96907881524216199, 0.90791009083569141, 0.91189790130217363]

```
