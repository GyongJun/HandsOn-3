from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy

mnist = fetch_openml('mnist_784', as_frame=False)

x, y = mnist.data, mnist.target
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')

sgd_clf = SGDClassifier(random_state = 42)
threshold = 3000

y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

#precisions/recalls trade-off 곡선 그라프 그리는 그리기

# plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
# plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold") 

#precisoin 과 recalls 사이 관계를 나타내는 그라프 그리기

# plt.grid(True)
# plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
# plt.show()

# precision 90% 가 넘는 첫 옳은 값의 index
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision] 

#prediction 을 classifier 의 predict() 가 아닌 경계값을 리용하여 예측
y_train_pred_90 = (y_scores >= threshold_for_90_precision)

new_precision_score = precision_score(y_train_5, y_train_pred_90)
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print(new_precision_score)
#90% 
print(recall_at_90_precision)
#48%