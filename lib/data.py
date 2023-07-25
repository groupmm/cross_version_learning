import glob
import os.path
from functools import partial

import numpy as np
import tensorflow as tf
from librosa import time_to_frames
from scipy.interpolate import interp1d
from welford import Welford

from config import batch_size, bins_per_octave, embedding_size, estimate_tuning, harmonics, hop_length, n_bins, num_batches_estimate_mean, num_input_frames, pairs_per_epoch, samplerate, tau_n, tau_p, use_augmentations
from lib.augment import augment_stacked_example
from lib.cached import get_chroma, get_hcqt_representation, get_mfcc
from lib.helper import pitch_to_chroma


class TrainingDataset:
    def __init__(self, mode):
        assert mode in ["CV", "SV"]
        self.mode = mode

        self.training_pieces = ["Ours_Beethoven_Symphony3Mvmt2", "Ours_Beethoven_Symphony3Mvmt3", "Ours_Beethoven_Symphony3Mvmt4", "Ours_Dvorak_Symphony9Mvmt1", "Ours_Dvorak_Symphony9Mvmt2", "Ours_Tschaikowsky_ViolinConcertoMvmt1", "Ours_Tschaikowsky_ViolinConcertoMvmt2", "WagnerRing_Wagner_WalkuereAct1"]
        self.min_measures, self.max_measures, self.input_reps, self.interpolators_measures_to_sec = dict(), dict(), dict(), dict()
        for piece in self.training_pieces:
            files_for_piece = glob.glob(f"data/01_RawData/audio_wav/{piece}_V*.wav")
            self.min_measures[piece] = None
            self.max_measures[piece] = None
            self.input_reps[piece] = {}
            self.interpolators_measures_to_sec[piece] = {}
            for f in files_for_piece:
                version = int(f.split("_")[-1].split(".")[0][1:])
                if piece == "WagnerRing_Wagner_WalkuereAct1" and (version == 1 or version > 6):
                    continue  # Don't use test versions
                self.input_reps[piece][version] = get_hcqt_representation(f, hop_length=hop_length, harmonics=harmonics, n_bins=n_bins, bins_per_octave=bins_per_octave, sr=samplerate, estimate_tuning=estimate_tuning)
                csv_path_search = f"data/02_Annotations/ann_audio_sync/{piece}_V{version}.csv"
                csv_path = glob.glob(csv_path_search, recursive=True)
                assert len(csv_path) == 1, csv_path_search
                symbolic_to_audio_wp = np.loadtxt(csv_path[0], delimiter=",")
                if self.min_measures[piece] is None:
                    self.min_measures[piece] = symbolic_to_audio_wp[0, 1]
                else:
                    assert self.min_measures[piece] == symbolic_to_audio_wp[0, 1], csv_path_search
                if self.max_measures[piece] is None:
                    self.max_measures[piece] = symbolic_to_audio_wp[-1, 1]
                else:
                    assert self.max_measures[piece] == symbolic_to_audio_wp[-1, 1], csv_path_search
                self.interpolators_measures_to_sec[piece][version] = interp1d(symbolic_to_audio_wp[:, 1], symbolic_to_audio_wp[:, 0], kind="linear", bounds_error=True, fill_value=(symbolic_to_audio_wp[0, 0], symbolic_to_audio_wp[-1, 0]))

        self.mean, self.std = 0, 1
        self.tf_dataset = self.build_tf_dataset()
        print("Estimating mean and variance from training data (for normalization). This can take some time...")
        self.mean, self.std = self.estimate_mean_std_from_tf_dataset()
        self.tf_dataset = self.build_tf_dataset()

    def build_tf_dataset(self):
        tf_dataset = tf.data.Dataset.from_generator(self.generator, output_signature=self.get_output_signature())
        if use_augmentations:
            tf_dataset = tf_dataset.map(augment_stacked_example, num_parallel_calls=tf.data.AUTOTUNE)
        normalization_func = partial(normalize_example_with_target, mean=self.mean, std=self.std)
        tf_dataset = tf_dataset.map(normalization_func, num_parallel_calls=tf.data.AUTOTUNE)
        tf_dataset = tf_dataset.take(pairs_per_epoch)
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=False)
        tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return tf_dataset

    @staticmethod
    def get_output_signature():
        return tf.TensorSpec(shape=(3, num_input_frames, 252, len(harmonics)), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)

    def get_network_input_shape(self):
        return self.get_output_signature()[0].shape

    def estimate_mean_std_from_tf_dataset(self):
        pitch_axis = 3
        non_pitch_axes = (0, 1, 2, 4)

        w = Welford()
        for batch in self.tf_dataset.take(int(num_batches_estimate_mean)):
            batch = batch[0]
            batch = batch.numpy()
            excerpts = batch.transpose(non_pitch_axes + (pitch_axis,)).reshape((-1, batch.shape[pitch_axis]))
            w.add_all(excerpts)

        w_mean = w.mean
        w_std = np.sqrt(w.var_s)

        for a in [0, 1, 3]:
            w_mean = np.expand_dims(w_mean, a)
            w_std = np.expand_dims(w_std, a)

        return w_mean, w_std

    def generator(self):
        while True:
            pos_anchor_piece = np.random.choice(self.training_pieces)
            neg_piece = pos_anchor_piece

            # neg_piece = np.random.choice(self.training_pieces)

            def sample_measure_from_piece(piece):
                m = self.min_measures[piece] + np.random.rand() * (self.max_measures[piece] - self.min_measures[piece])
                if piece == "B-1":
                    while 697 <= m < 955:
                        m = self.min_measures[piece] + np.random.rand() * (self.max_measures[piece] - self.min_measures[piece])
                return m

            versions_for_pos_anchor_piece = np.array(list(self.input_reps[pos_anchor_piece].keys()))
            anchor_version = np.random.choice(versions_for_pos_anchor_piece)
            if self.mode == "SV":
                pos_version = anchor_version
                neg_version = anchor_version
            else:  # CV
                pos_version = np.random.choice(versions_for_pos_anchor_piece)
                while anchor_version == pos_version:
                    pos_version = np.random.choice(versions_for_pos_anchor_piece)
                versions_for_neg_piece = np.array(list(self.input_reps[neg_piece].keys()))
                neg_version = np.random.choice(versions_for_neg_piece)

            pos_anchor_m = sample_measure_from_piece(pos_anchor_piece)
            neg_m = sample_measure_from_piece(neg_piece)
            while neg_piece == pos_anchor_piece and np.abs(self.interpolators_measures_to_sec[neg_piece][neg_version](neg_m) - self.interpolators_measures_to_sec[neg_piece][neg_version](pos_anchor_m)) < tau_n:
                neg_m = sample_measure_from_piece(neg_piece)

            def get_excerpt(m, piece, version, with_tolerance=False):
                m_sec = self.interpolators_measures_to_sec[piece][version](m)
                if with_tolerance:
                    m_sec += (2 * np.random.rand() - 1.0) * tau_p  # Only apply on one version, of course
                m_idx = time_to_frames(m_sec, sr=samplerate, hop_length=hop_length)
                m_idx = np.clip(m_idx, 0, self.input_reps[piece][version].shape[0] - num_input_frames)

                return get_excerpt_with_padding(self.input_reps[piece][version], m_idx, num_input_frames // 2)

            excerpts = [get_excerpt(pos_anchor_m, pos_anchor_piece, anchor_version),
                        get_excerpt(pos_anchor_m, pos_anchor_piece, pos_version, with_tolerance=True),
                        get_excerpt(neg_m, neg_piece, neg_version)]

            yield np.stack(excerpts, axis=0), 0


class EmbeddingExtractionDataset:
    def __init__(self, filename, mean, std):
        self.filename = filename
        self.input_rep = get_hcqt_representation(filename, hop_length=hop_length, harmonics=harmonics, n_bins=n_bins, bins_per_octave=bins_per_octave, sr=samplerate, estimate_tuning=estimate_tuning)
        self.mean, self.std = mean, std
        normalization_func = partial(normalize_example, mean=self.mean, std=self.std)

        self.tf_dataset = tf.data.Dataset.from_generator(self.generator, output_signature=self.get_output_signature())
        self.tf_dataset = self.tf_dataset.map(normalization_func, num_parallel_calls=tf.data.AUTOTUNE)
        self.tf_dataset = self.tf_dataset.batch(batch_size, drop_remainder=False)
        self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def get_output_signature():
        return tf.TensorSpec(shape=(num_input_frames, 252, len(harmonics)), dtype=tf.float32)

    def generator(self):
        for idx in range(self.input_rep.shape[0]):
            yield get_excerpt_with_padding(self.input_rep, idx, num_input_frames // 2)


class ProbingDataset:
    def __init__(self, mode, target, filenames, shuffle, emb_mean=None, emb_var=None):
        num_targets = None
        embeddings_per_rec = []
        labels_per_rec = []
        for f in filenames:
            if mode == "MFCC":
                embeddings = get_mfcc(f, samplerate=samplerate, hop_length=hop_length)
                self.output_size = 10
            elif mode == "Chroma":
                embeddings = get_chroma(f, samplerate=samplerate, hop_length=hop_length)
                self.output_size = 12
            else:
                embeddings = np.load(f"outputs/embeddings/{mode}/{f}.npy")
                assert embeddings.shape[1] == embedding_size
                self.output_size = embedding_size
            if target == "Inst":
                labels = np.load(f"data/02_Annotations/ann_audio_instruments_npz/{f}.npz")["arr_0"]
                labels = labels[:, [6, 7, 8, 9, 10, 11, 3, 13, 12, 14, 15, 16, 17]]
                num_targets = 13
            elif target == "PitchClass":
                labels = np.load(f"data/02_Annotations/ann_audio_note_npz/{f}.npz")["arr_0"]
                labels = (pitch_to_chroma(labels.T).T > 0)
                num_targets = 12
            else:
                assert False, target
            assert labels.shape[0] == embeddings.shape[0]
            if f.startswith("WagnerRing_Wagner_WalkuereAct1"):
                test_region_end, test_region_start = get_opera_act_test_excerpt(f)
                indices_to_use = np.arange(labels.shape[0])
                version = int(f.split("_")[-1][1:])
                if version == 1 or version > 6:
                    indices_to_use = indices_to_use[np.logical_and(indices_to_use >= test_region_start, indices_to_use < test_region_end)]
                else:
                    indices_to_use = indices_to_use[np.logical_not(np.logical_and(indices_to_use >= test_region_start, indices_to_use < test_region_end))]
                labels = labels[indices_to_use]
                embeddings = embeddings[indices_to_use]
            embeddings_per_rec.append(embeddings)
            labels_per_rec.append(labels)
        self.embeddings = np.concatenate(embeddings_per_rec, axis=0)
        self.labels = np.concatenate(labels_per_rec, axis=0).astype(np.int32)
        assert self.labels.shape[0] == self.embeddings.shape[0]

        if emb_mean is None:
            self.emb_mean = np.mean(self.embeddings, axis=0)
            self.emb_var = np.var(self.embeddings, axis=0)
        else:
            self.emb_mean = emb_mean
            self.emb_var = emb_var
        self.embeddings = (self.embeddings - self.emb_mean) / self.emb_var

        self.tf_dataset = tf.data.Dataset.from_generator(self.generator, output_signature=(tf.TensorSpec(self.output_size, dtype=tf.float32), tf.TensorSpec(num_targets, tf.int32)))
        self.tf_dataset = self.tf_dataset.batch(32, drop_remainder=False)
        self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.shuffle = shuffle

    def generator(self):
        indices = np.arange(self.labels.shape[0])
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield self.embeddings[i], self.labels[i]


class TrainingProbingDataset(ProbingDataset):
    def __init__(self, mode, target):
        training_pieces = ["Ours_Beethoven_Symphony3Mvmt2", "Ours_Beethoven_Symphony3Mvmt3", "Ours_Beethoven_Symphony3Mvmt4", "Ours_Dvorak_Symphony9Mvmt1", "Ours_Dvorak_Symphony9Mvmt2", "Ours_Tschaikowsky_ViolinConcertoMvmt1", "Ours_Tschaikowsky_ViolinConcertoMvmt2"]
        files_for_pieces = list(np.concatenate([glob.glob(f"data/01_RawData/audio_wav/{piece}_V*.wav") for piece in training_pieces]))
        files_for_pieces = [os.path.basename(f)[:-4] for f in files_for_pieces]
        files_for_pieces.append("WagnerRing_Wagner_WalkuereAct1_V2")
        files_for_pieces.append("WagnerRing_Wagner_WalkuereAct1_V3")
        files_for_pieces.append("WagnerRing_Wagner_WalkuereAct1_V4")
        files_for_pieces.append("WagnerRing_Wagner_WalkuereAct1_V5")
        files_for_pieces.append("WagnerRing_Wagner_WalkuereAct1_V6")
        super().__init__(mode, target, filenames=files_for_pieces, shuffle=True, emb_mean=None, emb_var=None)


class TestingProbingDataset(ProbingDataset):
    def __init__(self, mode, target, emb_mean, emb_var):
        files_for_pieces = ["WagnerRing_Wagner_WalkuereAct1_V1",
                            "WagnerRing_Wagner_WalkuereAct1_V7",
                            "WagnerRing_Wagner_WalkuereAct1_V8",
                            "Ours_Beethoven_Symphony3Mvmt1_V1",
                            "Ours_Dvorak_Symphony9Mvmt4_V1",
                            "Ours_Tschaikowsky_ViolinConcertoMvmt3_V1"]
        super().__init__(mode, target, filenames=files_for_pieces, shuffle=False, emb_mean=emb_mean, emb_var=emb_var)


def normalize_example(example, mean, std):
    return (example - mean) / std


def normalize_example_with_target(example, target, mean, std):
    return normalize_example(example, mean, std), target


def get_excerpt_with_padding(full_array, idx, num_context_frames):
    start = idx - num_context_frames
    end = idx + num_context_frames + 1

    pre_pad = 0
    if start < 0:
        pre_pad = -start
        start = 0
    post_pad = 0
    if end >= full_array.shape[0]:
        post_pad = end - full_array.shape[0]
        end = full_array.shape[0]

    excerpt = full_array[start:end, ...]
    if pre_pad > 0 or post_pad > 0:
        excerpt = np.pad(excerpt, ((pre_pad, post_pad), (0, 0), (0, 0)))
    return excerpt


def get_opera_act_test_excerpt(recording_name):
    csv_path_search = f"data/02_Annotations/ann_audio_sync/{recording_name}.csv"
    csv_path = glob.glob(csv_path_search, recursive=True)
    assert len(csv_path) == 1, csv_path_search
    symbolic_to_audio_wp = np.loadtxt(csv_path[0], delimiter=",")
    interpolator_measures_to_sec = interp1d(symbolic_to_audio_wp[:, 1], symbolic_to_audio_wp[:, 0], kind="linear", bounds_error=True, fill_value=(symbolic_to_audio_wp[0, 0], symbolic_to_audio_wp[-1, 0]))
    test_region_start = time_to_frames(interpolator_measures_to_sec(697), sr=samplerate, hop_length=hop_length)
    test_region_end = time_to_frames(interpolator_measures_to_sec(955), sr=samplerate, hop_length=hop_length)
    return test_region_end, test_region_start