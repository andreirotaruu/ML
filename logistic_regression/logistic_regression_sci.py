from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import(
    accuracy_score, 
    confusion_matrix, 
    precision_score,
    recall_score
)


#load wisconsin breast cancer dataset
data = load_breast_cancer()

#X = features 
#y = class labels
X = data.data
y = data.target

#split the testing and training data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=.2,
    random_state=0
)

#create pipeline 
#scaling data first before fitting the model
pipe = make_pipeline(
    StandardScaler(), 
    LogisticRegression()
)

#train data
pipe.fit(X_train, y_train)

#predict
y_pred = pipe.predict(X_test)

#evaluate
#to-do: figure out how these work
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
