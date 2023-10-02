from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

#Using the decision tree method to predict handrwitten digits dataset

digits = datasets.load_digits()
X=digits.data
y=digits.target
#print(X)
#print(X.shape)
#print(y)
#print(y.shape)

dtc=DecisionTreeClassifier()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
dtc.fit(X_train,y_train)

print("Predicted Digit: ",dtc.predict(X_test))
print("Actual Digit: ",y_test)

#compute percentage accuracy of the model of predictions vs actual value
print("Percentage Accuracy: ",dtc.score(X_test,y_test))