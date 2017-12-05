### Machine Learning Application
DriveTrain Approach
  1. Define Objective
  2. Levers: What input we can control to change the outcome
  3. Data: What data we can collect
  4. Model: How the levers influence


Better predictive model + Simulation model + Optimizer = The one that can stimulate the new / different consumption behavior of the consumers, therefore **optimizing** maximizing the their purchase ability. Really need the simulation that is embedded within the predictive model: What follows after we take action based on the predictive model result? (E.g. A has the highest probability of making a purchase, then what is the impact on the revenue when we send or does not send the sale-man to him?)

Most of these predictive modeling, more about interpretation that identifies the strong driver of the outcome, i.e. identifies the possible changes that we can execute.

Example: Churn Rate -> Predict a consumer leaving, what characteristics do they share? -> Probability that he will change his behavior given a taken action

Fraud Detection: Who is fraudulent just about the time we're delivering the product i.e. the operational constraint regarding the **product delivering time**

## Random Forest Interpretation (cont.)

### OOB Score
  + Measure: Mean prediction error on each training sample x_i, using only the trees that did not have x_i in their bootstrap sample
  + Help avoid the need for an independent validation dataset, but often underestimates actual performance improvement and the optimal number of iterations

### Confidence based on tree variance
  + Variance of the predictions
  + Risk modeling: I think he's a good risk but I'm not at all confident -> Not lending!
  + There is any group of values within a predictor that  
  + How can we know the confidence of the estimate? One simple way is to use the standard deviation of predictions, instead of just the mean. This tells us the relative confidence of predictions.
  + standard_deviation/mean

### Feature importance
  + Idea: If the variable is **not** important (the null hypothesis), then rearranging the values of that variable will **not** degrade prediction accuracy. i.e. If it's important, then rearranging other values for that predictor randomly should have a negative influence on the prediction.
  + Measuring:
    - Take the `YearMade` column and randomly shuffle it. Same distribution but no relation to the table at all
    - Before: MSE is .89 and after shuffle, the MSE is now .80, delta = .09
    - Do the same thing for `Enclosure`, it goes from .89 to .84, delta = .04
    - Quantifying? **Relative importance** of the difference (delta) comparing to the baseline model evaluation metrics.

  #### One-hot encoding - Binarization
  - Label encoding: Assuming the categorical value with a higher assigned value is a better category.
  - Instead of assigning an integer to a `categorical`, you want to split a variable into binary-value columns for each of the factor (0/1). E.g. Variable X with levels `[a, b, c]`, we will have 3 extra columns `X_a`, `X_b`, `X_c`, each has 0/1 values.
  - Easier to understand feature importance of categorical variables (helping with interpretability).

### Partial dependence
  + Idea: Tell how one variable affects the prediction, how the response will change as you change the predictor. There're a lot of interaction between variables.
  + Measure: with everything else constant (leave them as they are), replace the interested variable with one of its values, and train the model with every possible value of that model.  
  + **ICE plot** (Individual conditional expectation) visualizes the dependence of each instance’s predicted response on a feature. Each of blue line is the predicted value of one row for every possible value.
  + Do not drop any variable. Only drop them after doing feature importance. Before you plot ICE, think of what shape you're expecting.


### Contributions - Tree Interpreter:
  + Return the contribution of each node and its related feature to the final predicted value. Tell the importance of each feature for a particular observation(s).
  + Measuring:
    + For each prediction_i = bias_i + sum of feature_j contributions where the bias term is the mean of the training set
    + It can also be applied for a subset of observations, which is useful if comparing between the two subsets' contributions (biases are the same for both subsets, so the contributions are the one that creating differences)
  + Interpretation: For a forest, the prediction is simply the average of the bias terms plus the average contribution of each feature

### Interactive Feature Importance

**10**m --en--> **9.5** --ym--> **9.7** --meter--> **9.4**

Enclosure * YearMade = 9.7 - 10 = -0.3
YearMade  * Meter = 9.4 - 9.5 = -0.1
Meter * Enclosure = 9.4 - 10 = -0.6

### Extrapolation:
  + Idea: If one variable is important in distinguishing between the training and the validation set, it should not be in the training set (reducing the generalizing properties of the model)
  + Measuring:
    + Merge the training and validation set, and create an indicator telling if the observation is from the validation set or not.
    + Fit a random forest model to the data, in order to build a model that predict/classify if the observation belongs to a validation set. Measure the feature importance from the model, it is important, it has the potential to be excluded from the training data.
    + Run random forest models, each of which will consecutively drop one of the identified important features above. If the validation OOB or MSE score improves comparing to the baseline model, it should be excluded from the training data.


- Missing values: Missing-ness is actually very often, and can play an important role in the random forest.
