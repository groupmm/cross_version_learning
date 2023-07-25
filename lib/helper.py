import numpy as np
import scipy.ndimage
import tensorflow as tf


def enable_gpu_memory_growth():
    """
    Make gpu memory grow dynamically. Sometimes needed to avoid memory errors when Tensorflow starts.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def median_filter(input_sequence, feature_rate, median_filter_length_sec):
    median_filter_length_frames = int(median_filter_length_sec * feature_rate)
    if len(input_sequence.shape) == 2:
        median_filter_length_frames = (median_filter_length_frames, 1)
    return scipy.ndimage.median_filter(input_sequence, size=median_filter_length_frames, mode="constant", cval=0)


# https://github.com/meinardmueller/synctoolbox/blob/master/synctoolbox/feature/chroma.py
def pitch_to_chroma(f_pitch: np.ndarray,
                    midi_min: int = 21,
                    midi_max: int = 108) -> np.ndarray:
    """Aggregate pitch-based features into chroma bands.

    Parameters
    ----------
    f_pitch : np.ndarray [shape=(128, N)]
        MIDI pitch-based feature representation, obtained e.g. through
        ``audio_to_pitch_features``.

    midi_min : int
        Minimum MIDI pitch index to consider (default: 21)

    midi_max : int
        Maximum MIDI pitch index to consider (default: 108)

    Returns
    -------
    f_chroma: np.ndarray [shape=(12, N)]
        Rows of 'f_pitch' between ``midi_min`` and ``midi_max``,
        aggregated into chroma bands.
    """
    f_chroma = np.zeros((12, f_pitch.shape[1]))
    for p in range(midi_min, midi_max + 1):
        chroma = np.mod(p, 12)
        f_chroma[chroma, :] += f_pitch[p, :]
    return f_chroma


