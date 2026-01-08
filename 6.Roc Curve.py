from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
x, y = mnist.data, mnist.target

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')

sgd_clf = SGDClassifier(random_state=42)

y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.show()