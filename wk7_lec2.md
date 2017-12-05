Â# NLP

```python
trn,trn_y = texts_from_folders(f'{PATH}train',names)
val,val_y = texts_from_folders(f'{PATH}test',names)
```

## Bag of Words

- Ignore the language and only care about the term frequency of each word within the document
- Vocabulary is the list of all words appearing in the documents
- The probablity of each class for a given document
by using the ratio:

  R = Prob(c1 | d) / Prob(c0 | d) where

+ Prob(c1 | d) = Prob(d | c1) *  Prob(c1) / Prob(d)
+ Prob(c0 | d) = Prob(d | c0) *  Prob(c0) / Prob(d)

==> R = [Prob(d | c1) / Prob(d | c0)] * [Prob(c0) / Prob(c1)]

```python
# currently using fastai tokenize
veczr = CountVectorizer(tokenizer=tokenize)

# trn_term_doc[i] represents training document i
# it contains a # count of words for each document for each word in the vocabulary.
trn_term_doc = veczr.fit_transform(trn)

# use the same vocabulary with the training set
val_term_doc = veczr.transform(val)
```

```python
# 75132 unique words, but mostly don't appear in all documents
# 3749745 non-zero elements
trn_term_doc
<25000x75132 sparse matrix of type '<class 'numpy.int64'>'
	with 3749745 stored elements in Compressed Sparse Row format>

# 93 tokens, may not split on space!!!
trn_term_doc[0]
<1x75132 sparse matrix of type '<class 'numpy.int64'>'
	with 93 stored elements in Compressed Sparse Row format>

# document zero, word #1297 -> what's its frequency?
trn_term_doc[0,1297]

```

```python
 # list of all words
vocab = veczr.get_feature_names()
```

### Naive Bayes

Calculate  log-count ratio $r$ for each word $f$

```python
x=trn_term_doc
y=trn_y

p = x[y==1].sum(0)+1 # ratio of word f in positive documents
q = x[y==0].sum(0)+1 # ratio of word f in negative document
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(p)/len(q))
```

Formula for Naive Bayes

```python
pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()

# binarized Naive Bayes
pre_preds = val_term_doc.sign() @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()
```

### Logistic Regression

- Work better than Naiive Bayes and calculate r and b on our own, assuming there is a linear relationship

```python
x=trn_term_doc
y=trn_y

m = LogisticRegression(C=1e8, dual=True)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()
```

How to improve?
1. Binarize:
```python
# dual=True
# use when we have more columns than rows, run faster
m = LogisticRegression(C=1e8, dual=True)
m.fit(trn_term_doc.sign(), y)
preds = m.predict(val_term_doc.sign())
(preds==val_y).mean()
```

2. Regularize:
```python
# higher C, higher penalty
m = LogisticRegression(C=0.1, dual=True)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()
```

3. Combine both
```python
m = LogisticRegression(C=0.1, dual=True)
m.fit(trn_term_doc.sign(), y)
preds = m.predict(val_term_doc.sign())
(preds==val_y).mean()
```

### Trigram with NB Features

**n-gram**: contiguous sequence of n items from a given sequence of text or speech. By adopting this concept, we are capturing the sentiment of the words instead of just their relative frequency.

  + `ngram_range(1, 2)` means unigrams and `bigrams, (2, 2)` means only bigrams, and `ngram_range=(1,3)` means triagrams

For every document we compute binarized features as described above, but this time we use bigrams and trigrams too. Each feature is a log-count ratio. A logistic regression model is then trained to predict sentiment.

```python
veczr =  CountVectorizer(ngram_range=(1,3),
                       tokenizer=tokenize,
                        max_features=800000)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
```

Note that punctuations actually have meaning!

Neural network works better with the hidden layer comparing to regularization.

```python
y=trn_y
x=trn_term_doc.sign()
val_x = val_term_doc.sign()
p = x[y==1].sum(0)+1 # ratio of word f in positive documents
q = x[y==0].sum(0)+1 # ratio of word f in negative documents
# take logs due to the high-dim -> matrix multiplication -> very small values
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(p)/len(q))
```
```python
>>> r.shape, r
matrix([[-0.04911, -0.15543, -0.24226, ...,  1.10419, -0.68757, -0.68757]]))
>>> np.exp(r) # gives you the sentiment of the word
```
```python
m = LogisticRegression(C=0.1, dual=True)
m.fit(x, y);

preds = m.predict(val_x)
(preds.T==val_y).mean()
```

Now we fit the regularized matrix. Regularization moves the coefficient towards zero. It guides our prior towards expectation of how "r" should be. So "r" is basically capturing the information between p (how positive the term is) and q (how negative the term is). So "r" --> squashing the coefficients that are not relevant of the neutral words (p=q) towards zero.   

```python
x_nb = x.multiply(r) # regularize matrix
m = LogisticRegression(dual=True, C=0.1)
m.fit(x_nb, y);

val_x_nb = val_x.multiply(r)
preds = m.predict(val_x_nb)
(preds.T==val_y).mean()
```
![](/Users/ThyKhueLy/msan621/notes_ml/nlp_regularization.png)

## fastai NLP


```python
sl = 2000

md  = TextClassifierData.from_bow(trn_term_doc, trn_y, val_term_doc, val_y, sl)
```

```python
# wds = weight decay
learner = md.dotprod_nb_learner()
learner.fit(0,02, 1, wds=1e-6, cycle_len=1) # train for a few epochs
learner.fit(...)
learner.fit(...)
```

Let's look into the fastai package. Note that X is a pretty sparse matrix -> Use Embedding Matrix, applies a one-hot encoding to all the words, then multiply that by a random weight matrix. This helps to compress the number of columns (N=number of words) to one vector of length 1

```python
class DotProdNB(nn.Module):
    def __init__(self, nf, ny, w_adj=0.4, r_adj=10):
        # nf = number of features
        # if you have no weight -> keep a naive Bayes to reduce regularization when using w_adj=0.4 and r_adj=10

        super().__init__()
        self.w_adj,self.r_adj = w_adj,r_adj
        # embedding matrix
        self.w = nn.Embedding(nf+1, 1, padding_idx=0)
        self.w.weight.data.uniform_(-0.1,0.1)
        self.r = nn.Embedding(nf+1, ny)

    def forward(self, feat_idx, feat_cnt, sz):
        # now acts like a regularized logit with prior knowledge
        w = self.w(feat_idx)
        r = self.r(feat_idx)
        x = ((w+self.w_adj)*r/self.r_adj).sum(1)
        return F.softmax(x)
```
