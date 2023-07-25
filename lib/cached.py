import librosa
import numpy as np
from joblib import Memory

memory = Memory("outputs/cache")


@memory.cache
def get_mag_cqt_representation(wav_path, hop_length, n_bins, bins_per_octave, sr, estimate_tuning):
    audio, _ = librosa.core.load(wav_path, sr=sr, mono=True)
    audio /= np.max(np.abs(audio))
    assert len(audio) > 0
    if estimate_tuning:
        tuning = librosa.core.estimate_tuning(y=audio, sr=sr)
    else:
        tuning = 0.0
    C = librosa.core.cqt(audio, sr=sr, hop_length=hop_length, tuning=tuning, n_bins=n_bins, bins_per_octave=bins_per_octave)
    C = np.abs(C)
    return np.expand_dims(C.T, axis=-1)


def get_hcqt_representation(wav_path, hop_length, harmonics, n_bins, bins_per_octave, sr, estimate_tuning):
    cqt = get_mag_cqt_representation(wav_path, hop_length, n_bins, bins_per_octave, sr, estimate_tuning)
    cqt = cqt[..., 0].T
    cqt_frequencies = librosa.cqt_frequencies(n_bins, fmin=librosa.note_to_hz("C1"), bins_per_octave=bins_per_octave)
    hcqt = librosa.interp_harmonics(cqt, freqs=cqt_frequencies, harmonics=harmonics)
    hcqt = hcqt.transpose([2, 1, 0])
    return hcqt


@memory.cache
def get_mfcc(recording_name, samplerate, hop_length):
    audio, _ = librosa.core.load(f"data/01_RawData/audio_wav/{recording_name}.wav", sr=samplerate, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=samplerate, hop_length=hop_length).T
    mfcc = mfcc[:, 4:14]
    mfcc = librosa.util.normalize(mfcc, norm=2.0, axis=-1, fill=True)
    return mfcc


@memory.cache
def get_chroma(recording_name, samplerate, hop_length):
    audio, _ = librosa.core.load(f"data/01_RawData/audio_wav/{recording_name}.wav", sr=samplerate, mono=True)
    chroma = librosa.feature.chroma_stft(y=audio, sr=samplerate, hop_length=hop_length, norm=None).T
    chroma = librosa.util.normalize(chroma, norm=2.0, axis=-1, fill=True)
    return chroma
