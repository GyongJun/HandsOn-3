from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

mnist = fetch_openml('mnist_784', as_frame=False)

x, y = mnist.data, mnist.target

x_train, y_train = x[:60000], y[:60000]

y_train_5 = (y_train == 5)