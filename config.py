# Augmentations
linear_time_warp_prob = 0.5
linear_time_warp_max_abs_warp_factor = 0.1
shift_pitch_bins_nearest_pad_prob = 0.3
shift_pitch_bins_nearest_pad_max_shifted_bins = 36
mask_pitch_bins_prob = 0.1
mask_pitch_bins_max_num_masked_bins = 36
mask_time_steps_prob = 0.1
mask_time_steps_max_masked_time_steps = 21
add_uniform_noise_prob = 0.5
add_uniform_noise_max_abs_noise_amount = 0.05
change_magnitude_prob = 0.5
change_magnitude_max_change = 0.2
use_random_eq = True
random_eq_max_alpha = 20.0

# CQT from C1 to C8 with 3 bins per semitone -> 6×3×12=252 bins
samplerate = 22050
hop_length = 512
num_input_frames = 201
harmonics = [0.5, 1, 2, 3, 4]
n_bins = 252
bins_per_octave = 3 * 12
estimate_tuning = True

# Model configuration constants
num_prefiltering_convs = 1
embedding_size = 128
tau_p = 0.2
tau_n = 10.0
triplet_loss_alpha = 1.0
use_augmentations = True  # Configuration of individual augmentations is found above
adam_lr = 0.002
batch_size = 8
pairs_per_epoch = 16000
max_epochs = 500
num_batches_estimate_mean = 1000

# Settings for quantitative, SSM-based evaluation
smoothing_size_secs = 1.0
target_downsampling_feature_rate = 5
orig_feature_rate = samplerate / hop_length
downsampled_feature_rate = orig_feature_rate / int(orig_feature_rate / target_downsampling_feature_rate + 0.5)
kernel_sizes_secs = list(range(1, 61, 1))
peak_picking_local_average_size_sec = 3.0
foote_tau_sec = 3.0
peak_min_dist_sec = 2.0 * foote_tau_sec
