import os
import glob

import keras
import numpy as np
from keras.preprocessing.image import load_img

from net import *


class BodyParts14(keras.utils.all_utils.Sequence):
    def __init__(self,
        batch_size=128, img_size=(256, 256), img_aug=False,
        ids_list="", input_dir="", target_dir="",
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.ids_list = [i.strip() for i in open(ids_list)]
        self.files = []

        for file_name in self.ids_list:
            img_file = os.path.join(input_dir, "{}.jpg".format(file_name))
            label_file = os.path.join(target_dir, "{}.png".format(file_name))
            self.files.append({
                'image': img_file,
                'label': label_file,
                'fname': file_name,
            })

    def __len__(self):
        return len(self.files) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_paths = self.files[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_paths):
            image = load_img(path['image'], target_size=self.img_size)
            label = load_img(path['label'], target_size=self.img_size, color_mode="grayscale")
            x[j] = np.array(image) / 255.
            y[j] = np.expand_dims(label, 2)
            # y[j] -= 1
        return x, y


if __name__ == '__main__':
    from PIL import Image
    train_gen = BodyParts14(
        batch_size=1,
        img_size=(256,256),
        ids_list='train_list.txt',
        input_dir='images',
        target_dir='images',
    )

    for image_batch, labels_batch in train_gen:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
