# Machine Learning Snippets

## REGRESSION
## Linear Regression
```python
# Load the library
from sklearn.linear_model import LinearRegression
# Create an instance of the model
reg = LinearRegression()
# Fit the regressor
reg.fit(X,y)
# Do predictions
reg.predict([[2540],[3500],[4000]])
```
## Train-Test Split in Sklearn
```python
# Load the library
from sklearn.model_selection import train_test_split
# Create 2 groups each with input and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
# Fit only with training data
reg.fit(X_train,y_train)
```
## MAE in sklearn
```python
# Load the scorer
from sklearn.metrics import mean_absolute_error
# Use against predictions
mean_absolute_error(reg.predict(X_test),y_test)
```
## MAPE is not in Sklearn, so we implement ourselves
```python
np.mean(np.abs(reg.predict(X_test)-y_test)/y_test)
```
## k Nearest Neighbors in Sklearn
```python
# Load the library
from sklearn.neighbors import KNeighborsRegressor
# Create an instance
regk = KNeighborsRegressor(n_neighbors=2)
# Fit the data
regk.fit(X,y)
```
## RMSE in Sklearn
```python
# Load the scorer
from sklearn.metrics import mean_squared_error
# Use against predictions (we must calculate the square root of the MSE)
np.sqrt(mean_squared_error(reg.predict(X_test),y_test))
```
## Cross Validation in Sklearn
```python
# Load the library
from sklearn.model_selection import cross_val_score
# We calculate the metric for several subsets (determine by cv)
# With cv=5, we will have 5 results from 5 training/test
cross_val_score(reg,X,y,cv=5,scoring="neg_mean_squared_error")
```
## Testing Parameters: GridSearchCV
We could then try to find the best parameters by testing all of the combinations of
them. We test a GRID of parameters.
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
reg_test = GridSearchCV(KNeighborsRegressor(),
param_grid={"n_neighbors":np.arange(3,50)})
# Fit will test all of the combinations
reg_test.fit(X,y)
# Best estimator and best parameters
reg_test.best_score_
reg_test.best_estimator_
reg_test.best_params_
```
## Decision Tree in Sklearn
```python
# Load the library
from sklearn.tree import DecisionTreeRegressor
# Create an instance
regd = DecisionTreeRegressor(max_depth=3)
# Fit the data
regd.fit(X,y)
```
## Metric: Correlation
Correlation measures the correlation between the predictions and the real value.
```python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]
# Custom Scorer
from sklearn.metrics import make_scorer
def corr(pred,y_test):
return np.corrcoef(pred,y_test)[0][1]
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(corr))
```
## Metric: Correlation
Correlation measures the correlation between the predictions and the real value.
```python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]
# Custom Scorer
from sklearn.metrics import make_scorer
def corr(pred,y_test):
return np.corrcoef(pred,y_test)[0][1]
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(corr))
```
## Drawing the Decision Tree
```python
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,filled=True, rounded=True,
special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
```

## CLASSIFICATION
## Logistic Regression in sklearn
```python
# Load the library
from sklearn.linear_model import LogisticRegression
# Create an instance of the classifier
clf=LogisticRegression()
# Fit the data
clf.fit(X,y)
```
## Metric: Accuracy
```python
# With Metrics
from sklearn.metrics import accuracy_score
accuracy_score(y_test,clf.predict(X_test))
# Cross Validation
cross_val_score(clf,X,y,scoring="accuracy")
```
## k nearest neighbors Classifier (Same Parameters)
parameters = n_neighbors
```python
# Import Library
from sklearn.neighbors import KNeighborsClassifier
# Create instance
clf = KNeighborsClassifier (n_neighbors = 5)
# Fit
clf.fit(X,y)
```
## Metric: Precision and Recall
```python
# Metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
precision_score(y_test,clf.predict(X_test))
classification_report(y_test,clf.predict(X_test))
# Cross Validation
cross_val_score(clf,X,y,scoring="precision")
cross_val_score(clf,X,y,scoring="recall")
```
## Decision Tree Clasiffier in Sklearn
```python
# Import library
from sklearn.tree import DecisionTreeClassifier
# Create instance
clf = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)
# Fit the data
clf.fit(X,y)
```
## ROC Curve in Python
```python
# Load the library
from sklearn.metrics import roc_curve
# We chose the target
target_pos = 1 # Or 0 for the other class
fp,tp,_ = roc_curve(y_test,pred[:,target_pos])
plt.plot(fp,tp)
```
## AUC metric
```python
# Metrics
from sklearn.metrics import roc_curve, auc
fp,tp,_ = roc_curve(y_test,pred[:,1])
auc(fp,tp)
# Cross Validation
cross_val_score(clf,X,y,scoring="roc_auc")
```
## Saving and delivering a model
```python
clf = DecisionTreeClassifier(max_depth=17)
clf.fit(X,y)
import pickle
pickle.dump(clf,open("modelo.pickle","wb"))
clf_loaded = pickle.load(open("modelo.pickle","rb"))
ndf = pd.read_csv("nuevosind.csv")
clf_loaded.predict(ndf)
```
