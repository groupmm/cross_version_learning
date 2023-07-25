import glob
import os
import sys

import numpy as np
import tensorflow as tf

from config import embedding_size
from lib.data import EmbeddingExtractionDataset
from lib.helper import enable_gpu_memory_growth
from lib.loss import simple_triplet_loss
from lib.model import get_classification_forwarding_model, get_forwarding_model

if len(sys.argv) != 2 or sys.argv[1] not in ["CV", "SV", "Sup"]:
    print("Usage: python train_model X\nwhere X is an element of [CV, SV, Sup] (the model to extract embeddings for)")
    sys.exit(1)
mode = sys.argv[1]

enable_gpu_memory_growth()

print("Loading recordings for extracting embeddings")
train_mean = np.load(f"outputs/models/{mode}/train_mean.npy")[0]
train_std = np.load(f"outputs/models/{mode}/train_std.npy")[0]
pieces = ["Ours_Beethoven_Symphony3", "Ours_Dvorak_Symphony9", "Ours_Tschaikowsky_ViolinConcerto", "WagnerRing_Wagner_WalkuereAct1"]
recording_datasets = [EmbeddingExtractionDataset(f, train_mean, train_std) for piece in pieces for f in glob.glob(f"data/01_RawData/audio_wav/{piece}*.wav")]

print("Loading trained model")
model = tf.keras.models.load_model(f"outputs/models/{mode}", custom_objects={"simple_triplet_loss": simple_triplet_loss})

print("Constructing extraction model and copying over weights")
forwarding_input_shape = recording_datasets[0].get_output_signature().shape
if mode == "Sup":
    forwarding_model = get_classification_forwarding_model(forwarding_input_shape, embedding_size)
    assert len(model.layers) == 31 and len(forwarding_model.layers) == 30
    zipped_model_layers = zip(model.layers[:-2], forwarding_model.layers[:-1])
else:
    forwarding_model = get_forwarding_model(forwarding_input_shape, embedding_size)
    assert len(model.layers) == 34 and len(forwarding_model.layers) == 32
    zipped_model_layers = zip(model.layers[2:-1], forwarding_model.layers[1:])
for i, (pretrained_layer, forwarding_model_layer) in enumerate(zipped_model_layers):
    forwarding_model_layer.set_weights(pretrained_layer.get_weights())
forwarding_model.compile()
forwarding_model.summary()

print("Extracting embeddings for recordings")
embeddings_folder = f"outputs/embeddings/{mode}"
os.makedirs(embeddings_folder, exist_ok=True)
for dataset in recording_datasets:
    recording_name = os.path.split(dataset.filename)[-1][:-4]
    embeddings_file = f"{embeddings_folder}/{recording_name}.npy"
    if not os.path.exists(embeddings_file):
        embeddings = forwarding_model.predict(dataset.tf_dataset, verbose=1)
        np.save(embeddings_file, embeddings)
