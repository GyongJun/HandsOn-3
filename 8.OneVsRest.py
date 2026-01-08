from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

mnist = fetch_openml('mnist_784', as_frame=False)
x, y = mnist.data, mnist.target

x_train, y_train, x_test, y_test = x[:60000], y[:60000], x[60000:], y[60000:]
some_digit = x[0]

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(x_train[:2000], y_train[:2000])
# print(ovr_clf.predict([some_digit]))
# print(len(ovr_clf.estimators_))

