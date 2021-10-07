import cv2
import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm

input_img = cv2.imread("./Lenna.png")
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train_dummy = np.array([cv2.resize(x, (8, 8)) for x in x_train]).astype("float")

dont_use_same_image = True


def get_best_match(patch):
    best_match = 0
    best_match = np.argmin(np.sum((x_train_dummy - patch) ** 2, axis=(1, 2, 3)))
    if dont_use_same_image:
        x_train_dummy[best_match] += 1000
    return best_match


mosaic = np.zeros((512 * 4, 512 * 4, 3), dtype="uint8")
input_img_float = input_img.astype("float")

for i in tqdm(range(64)):
    for j in range(64):
        patch = input_img_float[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, :]
        match = x_train[get_best_match(cv2.resize(patch, (8, 8)))]
        mosaic[i * 8 * 4 : (i + 1) * 8 * 4, j * 8 * 4 : (j + 1) * 8 * 4, :] = match
    mosaic_wip = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./mosaic_wip.png", mosaic_wip)

mosaic = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
cv2.imwrite("./mosaic.png", mosaic)
cv2.imshow("Mosaic", cv2.resize(mosaic, [512, 512]))
cv2.waitKey()
