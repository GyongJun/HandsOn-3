from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)

x,y = mnist.data, mnist.target #mnist.data: input datas, mnist.target: labels
# print(x, y)
# print(x.shape)
# print(y.shape)

# print(y)
# print(y.shape)

y_train = y[60000:]
print(y_train, y_train.shape)

# import matplotlib.pyplot as plt

# def plot_digit(image_data):
#     image = image_data.reshape(28, 28)
#     plt.imshow(image, cmap="binary")
#     plt.axis("off")

# some_digit = x[0]
# plot_digit(some_digit)
# plt.show()