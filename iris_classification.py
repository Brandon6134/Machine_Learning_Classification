from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#Using Random Forest method to predict the Iris dataset

iris = datasets.load_iris()
X=iris.data
y=iris.target

clf=RandomForestClassifier()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
clf.fit(X_train,y_train)
#Below line predicts the probabilities of what group iris colour that the 
#entered parameter likely has (e.g. 20% belongs to the first group, 40% belongs to second group, 40% belongs to third group)
print(clf.predict_proba([[5.1,3.5,1.4,0.2]]))

#using model, predicts which group iris each of the test x arrays belong to
#y_test are the actual iris group values
print("Predicted Iris Group Values: ",clf.predict(X_test))
print("Actual Iris Group Values: ",y_test)

#compute percentage accuracy of the model of predictions vs actual value
print("Percentage Accuracy: ",clf.score(X_test,y_test))

