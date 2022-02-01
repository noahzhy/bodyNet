import keras
from keras.losses import *
from keras.metrics import *
from keras.callbacks import *

from net import *
from datasets import *


epochs = 100
n_class = 15
img_size = (224, 224)

train_gen = BodyParts14(
    batch_size=1,
    img_size=img_size,
    ids_list='train_list.txt',
    input_dir='images',
    target_dir='images',
)

val_gen = BodyParts14(
    batch_size=1,
    img_size=img_size,
    ids_list='train_list.txt',
    input_dir='images',
    target_dir='images',
)

model = build_model((224,224,3), n_class)
model.summary()
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("bodyParts14_best.h5", save_best_only=True)
]


model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks
)

if __name__ == '__main__':
    pass
