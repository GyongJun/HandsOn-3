from sklearn.datasets import fetch_openml
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot  as plt

mnist = fetch_openml('mnist_784', as_frame=False)
x, y = mnist.data, mnist.target
x_train, y_train, x_test, y_test = x[:60000], y[:60000], x[60000:], y[60000:]

scaler = StandardScaler()
sgd_clf = SGDClassifier(random_state=42)
x_train_scaled = scaler.fit_transform(x_train.astype('float64'))

# print(cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy"))
# print(x_train_scaled)
# print(cross_val_score(sgd_clf, x_train_scaled, y_train, cv=3, scoring="accuracy"))

y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()