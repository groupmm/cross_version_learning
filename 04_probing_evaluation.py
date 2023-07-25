import os
import sys

import scipy
import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy

from config import embedding_size
from lib.data import TestingProbingDataset, TrainingProbingDataset
from lib.eval import framewise_instrument_scores
from lib.helper import enable_gpu_memory_growth
from lib.model import get_probing_model

if len(sys.argv) != 3 or sys.argv[1] not in ["CV", "SV", "Sup", "MFCC", "Chroma"] or sys.argv[2] not in ["Inst", "PitchClass"]:
    print("Usage: python train_model X\nwhere X is an element of [CV, SV, Sup, MFCC, Chroma] (the model/representation to be probed) and Y is an element of [Inst, PitchClass] (the type of target class to be used)")
    sys.exit(1)
mode = sys.argv[1]
target = sys.argv[2]

enable_gpu_memory_growth()

print("Loading training data")
dataset_train = TrainingProbingDataset(mode, target)

print("Building probing model")
num_targets = 12 if target == "PitchClass" else 13
model = get_probing_model(dataset_train.output_size, num_targets)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss={"logits": BinaryCrossentropy(from_logits=True)})
model.summary()

print("Starting training")
callbacks = [tf.keras.callbacks.TerminateOnNaN()]
model.fit(dataset_train.tf_dataset, epochs=10, callbacks=callbacks)

print("Loading test data")
dataset_test = TestingProbingDataset(mode, target, emb_mean=dataset_train.emb_mean, emb_var=dataset_train.emb_var)
predictions = model.predict(dataset_test.tf_dataset)
predictions = scipy.special.expit(predictions)  # Sigmoid activation
os.makedirs(f"outputs/probing/{mode}/{target}", exist_ok=True)
if target == "Inst":
    target_names = ["Flute", "Oboe", "Clarinet", "Bassoon", "French Horn", "Trumpet", "Timpani", "Male", "Female", "Violin", "Viola", "Cello", "Contrabass"]
else:
    target_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
framewise_instrument_scores(f"outputs/probing/{mode}/{target}", dataset_test.labels, predictions, 0.5, target_names)
