# Structured Data & Neural Network

- Feature engineering is important, throw in anything that you think is important. There is little harm in adding them.
- Should always do left join and then check if there is any missing values (i.e. null). Also check the number of rows. If the key is not unique ==> more rows than original.
- SQLite is a handy tool that could be used for large relational database (which don't sit in memory at once)
- Reminding that in pandas, we could use `.cat` to access categorical data attributes. Same applied for `.str` or `.dt`

```python
# subtract the values on the column Date with the column CompetitionOpenSince
df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days
```
- Prefer categorical over continuous. If category -> need to use embedding matrix (linear layer), very flexible. If continuous, have to learn its mathematical. If there are too many levels of the categorical, we can limit by setting a threshold that groups any level above that to be in one single bucket.

- `.apply` is extremely slow since it doesn't take advantage of the parallelization of pandas but instead working as a loop in python

```python
def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    last_item = 0
    res = []

    for s,i,v,d in zip(df.store_nbr.values, df.item_nbr.values, df[fld].values, df.date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_item = i
            last_store = s
        if v: last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1).astype(int))
    df[pre+fld] = res
```
