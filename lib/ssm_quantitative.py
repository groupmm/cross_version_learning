import os
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
from libfmp.c4 import compute_kernel_checkerboard_gaussian, compute_novelty_ssm
from scipy import signal

from config import downsampled_feature_rate, hop_length, kernel_sizes_secs, orig_feature_rate, peak_min_dist_sec, samplerate, smoothing_size_secs, target_downsampling_feature_rate
from lib.cached import get_chroma, get_mfcc
from lib.data import get_opera_act_test_excerpt
from lib.helper import median_filter, pitch_to_chroma


# https://github.com/meinardmueller/synctoolbox/blob/master/synctoolbox/feature/utils.py
def smooth_downsample_feature(f_feature: np.ndarray,
                              input_feature_rate: float,
                              win_len_smooth: int = 0,
                              downsamp_smooth: int = 1) -> Tuple[np.ndarray, float]:
    """Temporal smoothing and downsampling of a feature sequence

    Parameters
    ----------
    f_feature : np.ndarray
        Input feature sequence, size dxN

    input_feature_rate : float
        Input feature rate in Hz

    win_len_smooth : int
        Smoothing window length. For 0, no smoothing is applied.

    downsamp_smooth : int
        Downsampling factor. For 1, no downsampling is applied.

    Returns
    -------
    f_feature_stat : np.ndarray
        Downsampled & smoothed feature.

    new_feature_rate : float
        New feature rate after downsampling
    """
    if win_len_smooth != 0 or downsamp_smooth != 1:
        # hack to get the same results as on MATLAB
        stat_window = np.hanning(win_len_smooth+2)[1:-1]
        stat_window /= np.sum(stat_window)

        # upfirdn filters and downsamples each column of f_stat_help
        f_feature_stat = signal.upfirdn(h=stat_window, x=f_feature, up=1, down=downsamp_smooth)
        seg_num = f_feature.shape[1]
        stat_num = int(np.ceil(seg_num / downsamp_smooth))
        cut = int(np.floor((win_len_smooth - 1) / (2 * downsamp_smooth)))
        f_feature_stat = f_feature_stat[:, cut: stat_num + cut]
    else:
        f_feature_stat = f_feature

    new_feature_rate = input_feature_rate / downsamp_smooth

    return f_feature_stat, new_feature_rate


def process_embeddings(embeddings, smoothing_size_secs, feature_rate, downsampling_to_feature_rate, normalize=True):
    smoothing_size_frames = int(smoothing_size_secs * feature_rate + 0.5)
    if smoothing_size_frames % 2 == 0:
        smoothing_size_frames += 1
    downsampling_frames = int(feature_rate / downsampling_to_feature_rate + 0.5)
    embeddings, new_feature_rate = smooth_downsample_feature(embeddings.T, feature_rate, smoothing_size_frames, downsampling_frames)
    embeddings = embeddings.T
    if normalize:
        embeddings = librosa.util.normalize(embeddings, norm=2.0, axis=-1, fill=True)
    return embeddings, new_feature_rate


def build_similarity_matrix(a, b):
    S = a.dot(b.T)
    return S


def get_foote_activations(mode, recording_name, kernel_size):
    curve_filename = f"outputs/foote_activations/{mode}_{recording_name}_{kernel_size}.npy"
    if os.path.exists(curve_filename):
        return np.load(curve_filename)
    else:
        print("creating", curve_filename)
        if mode == "Ref_I":
            embeddings = np.load(f"data/02_Annotations/ann_audio_instruments_npz/{recording_name}.npz")["arr_0"]
            embeddings = embeddings[:, [6, 7, 8, 9, 10, 11, 3, 13, 12, 14, 15, 16, 17]]
        elif mode == "Ref_H":
            embeddings = np.load(f"data/02_Annotations/ann_audio_note_npz/{recording_name}.npz")["arr_0"]
            embeddings = (pitch_to_chroma(embeddings.T).T > 0)
        elif mode == "MFCC":
            embeddings = get_mfcc(recording_name, samplerate=samplerate, hop_length=hop_length)
        elif mode == "Chroma":
            embeddings = get_chroma(recording_name, samplerate=samplerate, hop_length=hop_length)
        else:
            embeddings = np.load(f"outputs/embeddings/{mode}/{recording_name}.npy")

        if recording_name.startswith("WagnerRing_Wagner_WalkuereAct1"):
            test_region_end, test_region_start = get_opera_act_test_excerpt(recording_name)
            embeddings = embeddings[int(test_region_start):int(test_region_end)]
        embeddings, downsampled_feature_rate_computed = process_embeddings(embeddings, smoothing_size_secs, orig_feature_rate, target_downsampling_feature_rate)
        assert downsampled_feature_rate == downsampled_feature_rate_computed
        S = build_similarity_matrix(embeddings, embeddings)

        # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html
        kernel_size_frames = int(kernel_size * downsampled_feature_rate + 0.5)
        kernel = compute_kernel_checkerboard_gaussian(kernel_size_frames, var=1.0, normalize=True)
        # This function is VERY slow. Reimplement using convolutions?
        curve = compute_novelty_ssm(S, kernel=kernel, L=kernel_size_frames, exclude=True)
        np.save(curve_filename, curve)
        return curve


def get_peak_list(curve, kernel_size, feature_rate):
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_PeakPicking.html
    global_threshold = 0.1 * np.max(curve)
    local_threshold = median_filter(curve, feature_rate, kernel_size)
    threshold = np.maximum(global_threshold, local_threshold)
    peak_list = signal.find_peaks(curve, distance=int(peak_min_dist_sec * feature_rate) + 1, height=threshold)[0]
    peak_list = peak_list / feature_rate
    curve = curve - threshold
    curve = np.maximum(0, curve)
    return peak_list, curve


def get_curves_peaks(mode, recording_name):
    curves = [get_foote_activations(mode, recording_name, k) for k in kernel_sizes_secs]
    curves_smoothed = []
    peak_lists = []
    for c, k in zip(curves, kernel_sizes_secs):
        a, b = get_peak_list(c, k, downsampled_feature_rate)
        peak_lists.append(a)
        curves_smoothed.append(b)
    return curves_smoothed, peak_lists


def boundary_evaluation(boundaries_predicted, boundaries_target, tau):
    if np.any(np.diff(boundaries_predicted) <= 2 * tau):
        print("Minimal distance condition for boundaries violated for predictions!")
    if np.any(np.diff(boundaries_target) <= 2 * tau):
        print("Minimal distance condition for boundaries violated for targets!")
    if len(boundaries_target) == 0:
        TPs = []
        FPs = boundaries_predicted
        FNs = []
        if len(boundaries_predicted) == 0:
            P = 1
        else:
            P = 0
        R = 1
    elif len(boundaries_predicted) == 0:
        TPs = []
        FPs = []
        FNs = boundaries_target
        P = 1
        R = 0
    else:
        distances_boundaries = np.abs(boundaries_predicted[:, np.newaxis] - boundaries_target[np.newaxis, :])
        TPs_indices = np.min(distances_boundaries, axis=-1) <= tau
        TPs = boundaries_predicted[TPs_indices]
        FPs = boundaries_predicted[np.logical_not(TPs_indices)]
        FNs = boundaries_target[np.min(distances_boundaries, axis=0) > tau]
        P = len(TPs) / len(boundaries_predicted)
        R = len(TPs) / (len(TPs) + len(FNs))  # divisor will always be !=0

    df_boundary_p_r_f = pd.DataFrame([[P, R, 2 * P * R / (P + R) if (P + R) > 0 else 0, len(boundaries_target)]], columns=["P", "R", "F", "Support"], index=[f"tau={tau}"])

    return df_boundary_p_r_f, TPs, FPs, FNs
