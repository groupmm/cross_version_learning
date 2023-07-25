import sys

import numpy as np
import tensorflow as tf

from config import adam_lr, embedding_size, max_epochs
from lib.data import TrainingDataset
from lib.helper import enable_gpu_memory_growth
from lib.loss import simple_triplet_loss
from lib.model import get_tiplet_model

if len(sys.argv) != 2 or sys.argv[1] not in ["CV", "SV"]:  # TODO training of the Sup model currently not supported
    print("Usage: python train_model X\nwhere X is an element of [CV, SV] (the model to be trained)")
    sys.exit(1)
mode = sys.argv[1]

enable_gpu_memory_growth()

print("Loading training data")
dataset_train = TrainingDataset(mode=mode)
train_mean, train_std = dataset_train.mean, dataset_train.std
np.save(f"outputs/models/{mode}/train_mean.npy", train_mean)
np.save(f"outputs/models/{mode}/train_std.npy", train_std)

print("Building model")
model = get_tiplet_model(dataset_train.get_network_input_shape(), embedding_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr), loss={"tf.reshape_1": simple_triplet_loss})
model.summary()

print("Starting training")
callbacks = [tf.keras.callbacks.TerminateOnNaN()]
model.fit(dataset_train.tf_dataset, epochs=max_epochs, callbacks=callbacks)
model.save(f"outputs/models/{mode}")
