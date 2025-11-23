from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784', as_frame=False)

x, y = mnist.data, mnist.target
some_digit = x[0]
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

#print(cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy"))
print(sgd_clf.predict([some_digit]))