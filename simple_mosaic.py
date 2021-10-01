import cv2
import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm

input_img = cv2.imread("./Lenna.png")
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train[:1000]


def get_best_match(patch):
    best_match = 0
    min_dist = np.inf
    for idx, item in enumerate(x_train):
        dist = np.sum((item.astype("float") - patch.astype("float")) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_match = idx
    return best_match


mosaic = np.zeros((512 * 4, 512 * 4, 3), dtype="uint8")

for i in range(64):
    for j in tqdm(range(64)):
        patch = input_img[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, :]
        match = x_train[get_best_match(cv2.resize(patch, (32, 32)))]
        mosaic[i * 8 * 4 : (i + 1) * 8 * 4, j * 8 * 4 : (j + 1) * 8 * 4, :] = match


mosaic = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
cv2.imwrite("./mosaic.png", mosaic)
cv2.imshow("Mosaic", cv2.resize(mosaic, [256, 256]))
cv2.waitKey()
