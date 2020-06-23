import cv2
import util
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm

train_imgs, train_labels = util.get_train_data()
test_imgs, test_labels = util.get_test_data()

linear_svm_model = svm.LinearSVC()
linear_svm_model.fit(train_imgs, train_labels)
linear_svm_results = linear_svm_model.predict(test_imgs)

util.evaluate(linear_svm_results, test_labels)

test_imgs = test_imgs.reshape(-1, 28, 28)
test_imgs = np.random.shuffle(test_imgs)

for i in range(10):
    util.visualize(test_imgs[i], linear_svm_model)
