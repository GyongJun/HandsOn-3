from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)

x, y = mnist.data, mnist.target
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')

sgd_clf = SGDClassifier(random_state = 42)
threshold = 3000

y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
plt.show()