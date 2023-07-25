import tensorflow as tf

from config import add_uniform_noise_max_abs_noise_amount, add_uniform_noise_prob, bins_per_octave, change_magnitude_max_change, change_magnitude_prob, harmonics, linear_time_warp_max_abs_warp_factor, linear_time_warp_prob, mask_pitch_bins_max_num_masked_bins, mask_pitch_bins_prob, mask_time_steps_max_masked_time_steps, mask_time_steps_prob, random_eq_max_alpha, shift_pitch_bins_nearest_pad_max_shifted_bins, shift_pitch_bins_nearest_pad_prob, use_random_eq


def linear_time_warp(spec, warp_factor):
    orig_num_time_steps = tf.shape(spec)[0]
    orig_num_time_steps_f = tf.cast(orig_num_time_steps, tf.float32)
    new_num_time_steps = tf.cast(orig_num_time_steps_f * warp_factor, dtype=tf.int32)
    spec = tf.image.resize(spec, [new_num_time_steps, tf.shape(spec)[1]], method=tf.image.ResizeMethod.BILINEAR)
    if new_num_time_steps < orig_num_time_steps:
        # This is actually very unnatural
        pre_padding = (orig_num_time_steps - new_num_time_steps) // 2
        spec = tf.pad(spec, ((pre_padding, orig_num_time_steps - new_num_time_steps - pre_padding), (0, 0), (0, 0)))
    else:
        first_time_step = (new_num_time_steps - orig_num_time_steps) // 2
        spec = spec[first_time_step:first_time_step + orig_num_time_steps, :, :]
    return spec


def shift_pitch_bins_nearest_pad(spec, shift_amount):  # + means up, - means down
    first_bin = spec[:, :1, :]
    last_bin = spec[:, -1:, :]
    truncated_spec = spec[:, tf.math.maximum(-shift_amount, 0):tf.shape(spec)[1] - tf.math.maximum(shift_amount, 0), :]
    spec = tf.concat([tf.repeat(first_bin, tf.math.maximum(shift_amount, 0), axis=1), truncated_spec, tf.repeat(last_bin, tf.math.maximum(-shift_amount, 0), axis=1)], axis=1)
    return spec


def mask_pitch_bins(spec, start_bin, end_bin):
    mask = tf.range(tf.shape(spec)[1])
    mask = tf.logical_or(mask < start_bin, mask >= end_bin)
    return spec * tf.cast(mask[tf.newaxis, :, tf.newaxis], tf.float32)


def mask_time_steps(spec, start_time_step, end_time_step):
    mask = tf.range(tf.shape(spec)[0])
    mask = tf.logical_or(mask < start_time_step, mask >= end_time_step)
    return spec * tf.cast(mask[:, tf.newaxis, tf.newaxis], tf.float32)


def add_uniform_noise(spec, max_abs_noise_amount):
    noise = tf.random.uniform(tf.shape(spec), -max_abs_noise_amount, max_abs_noise_amount)
    spec = spec + noise
    return tf.maximum(spec, 0)  # remove negative values


def change_magnitude(spec, factor):
    return factor * spec


def _tf_log2(x, dtype=tf.float32):
    numerator = tf.math.log(tf.cast(x, dtype=dtype))
    denominator = tf.math.log(tf.constant(2, dtype=dtype))
    return numerator / denominator


def random_eq(spec, max_alpha, hcqt_harmonics):
    # alpha: how steep does the curve fall off
    # beta: peak of eq curve
    num_pitch_bins = tf.shape(spec)[1]

    minval = -1 * tf.ones([])
    while minval < 0:
        alpha = tf.random.uniform(shape=[], minval=0, maxval=max_alpha)
        beta = tf.random.uniform(shape=[], minval=0, maxval=num_pitch_bins, dtype=tf.int32)
        filtvecs = []
        for hcqt_harmonic in hcqt_harmonics:
            offset = int(bins_per_octave * _tf_log2(hcqt_harmonic))
            beta_harm = beta - offset
            currfiltvec = (1.0 - (2e-6 * alpha * tf.cast(tf.range(num_pitch_bins) - beta_harm, dtype=tf.float32) ** 2))
            filtvecs.append(currfiltvec)
        filtmat = tf.stack(filtvecs, axis=-1)
        minval = tf.reduce_min(filtmat)
    return tf.expand_dims(filtmat, axis=0) * spec


def augment_input(input):
    spec = input

    if use_random_eq:
        spec = random_eq(spec, random_eq_max_alpha, harmonics)

    if linear_time_warp_prob > 0.0 and tf.random.uniform(shape=[]) < linear_time_warp_prob:
        warp_factor = tf.random.uniform(shape=[], minval=1 - linear_time_warp_max_abs_warp_factor, maxval=1 + linear_time_warp_max_abs_warp_factor)
        spec = linear_time_warp(spec, warp_factor)

    if shift_pitch_bins_nearest_pad_prob > 0.0 and tf.random.uniform(shape=[]) < shift_pitch_bins_nearest_pad_prob:
        shift_amount = tf.random.uniform(shape=[], minval=-shift_pitch_bins_nearest_pad_max_shifted_bins, maxval=shift_pitch_bins_nearest_pad_max_shifted_bins + 1, dtype=tf.int32)
        spec = shift_pitch_bins_nearest_pad(spec, shift_amount)

    if mask_pitch_bins_prob > 0.0 and tf.random.uniform(shape=[]) < mask_pitch_bins_prob:
        num_masked_pitch_bins = tf.random.uniform(shape=[], minval=0, maxval=mask_pitch_bins_max_num_masked_bins + 1, dtype=tf.int32)
        first_masked_pitch_bin = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(spec)[1] - num_masked_pitch_bins, dtype=tf.int32)
        spec = mask_pitch_bins(spec, first_masked_pitch_bin, first_masked_pitch_bin + num_masked_pitch_bins)

    if mask_time_steps_prob > 0.0 and tf.random.uniform(shape=[]) < mask_time_steps_prob:
        num_masked_time_steps = tf.random.uniform(shape=[], minval=0, maxval=mask_time_steps_max_masked_time_steps + 1, dtype=tf.int32)
        first_masked_time_step = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(spec)[0] - num_masked_time_steps, dtype=tf.int32)
        spec = mask_time_steps(spec, first_masked_time_step, first_masked_time_step + num_masked_time_steps)

    if add_uniform_noise_prob > 0.0 and tf.random.uniform(shape=[]) < add_uniform_noise_prob:
        spec = add_uniform_noise(spec, add_uniform_noise_max_abs_noise_amount)

    if change_magnitude_prob > 0.0 and tf.random.uniform(shape=[]) < change_magnitude_prob:
        spec = change_magnitude(spec, tf.random.uniform(shape=[], minval=1.0 - change_magnitude_max_change, maxval=1.0 + change_magnitude_max_change))

    return spec


def augment_example(input, target):
    spec = augment_input(input)
    return spec, target


def augment_stacked_example(input, target):
    specs = tf.map_fn(augment_input, input)
    return specs, target
