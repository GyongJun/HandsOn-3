from sklearn.model_selection import cross_val_predict
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

mnist = fetch_openml('mnist_784', as_frame=False)

x, y = mnist.data, mnist.target
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')

sgd_clf = SGDClassifier(random_state=42)

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

y_train_perfect_predictions = y_train_5
cm1 = confusion_matrix(y_train_5, y_train_perfect_predictions)
print(cm1)

print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))