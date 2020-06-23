import gzip
import numpy as np

from PIL import Image


def get_train_data():
    data = gzip.open('train-images-idx3-ubyte.gz', 'r')
    label = gzip.open('train-labels-idx1-ubyte.gz', 'r')

    data.read(16)
    label.read(8)

    data_buffer = data.read()
    label_buffer = label.read()

    imgs = np.frombuffer(data_buffer, dtype=np.uint8).astype(np.int32)
    labels = np.frombuffer(label_buffer, dtype=np.uint8).astype(np.int32)

    imgs = imgs.reshape(-1, 28, 28)
    imgs = imgs.reshape((60000, -1))

    return imgs, labels


def get_test_data():
    data = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
    label = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')

    data.read(16)
    label.read(8)

    data_buffer = data.read()
    label_buffer = label.read()

    imgs = np.frombuffer(data_buffer, dtype=np.uint8).astype(np.int32)
    labels = np.frombuffer(label_buffer, dtype=np.uint8).astype(np.int32)

    imgs = imgs.reshape(-1, 28, 28)
    imgs = imgs.reshape((10000, -1))

    return imgs, labels


def evaluate(results, labels):
    correct = 0
    for i in range(len(results)):
        if results[i] == labels[i]:
            correct += 1

    print('accuracy: ' + str(correct / 10000 * 100) + '%')


def visualize(img, model):

    im = Image.fromarray(img)
    im.show()

    img = img.reshape((1, -1))
    result = model.predict(img)
    print(result)
