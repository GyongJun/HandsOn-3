from sklearn.datasets import fetch_openml
from sklearn.svm import SVC

mnist = fetch_openml('mnist_784', as_frame=False)
x, y = mnist.data, mnist.target
x_train, y_train, x_test, y_test = x[:60000], y[:60000], x[60000:], y[60000:]
some_digit = x[0]

svm_clf = SVC(random_state=42)
svm_clf.fit(x_train[:2000], y_train[:2000])
# print(svm_clf.predict([some_digit]))

some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))