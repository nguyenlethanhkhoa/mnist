import cv2
import util
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

train_imgs, train_labels = util.get_train_data()
test_imgs, test_labels = util.get_test_data()

random_forest_model = RandomForestClassifier()
random_forest_model.fit(train_imgs, train_labels)
random_forest_results = random_forest_model.predict(test_imgs)

util.evaluate(random_forest_results, test_labels)

test_imgs = test_imgs.reshape(-1, 28, 28)
np.random.shuffle(test_imgs)

for i in range(10):
    util.visualize(test_imgs[i], random_forest_model)
