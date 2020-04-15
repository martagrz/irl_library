import numpy as np


def get_demonstrations_spilt(demonstrations, context_share=0.5, target_share=None):
    n_demonstrations, n_steps, _ = demonstrations.shape
    context_range = np.int(n_demonstrations*n_steps*context_share)
    # randomly shuffle the demonstrations
    demonstrations = demonstrations.reshape([1, n_demonstrations*n_steps, 2])
    np.random.shuffle(demonstrations)
    context_data = demonstrations[:, :context_range, :]

    if target_share == 'all':
        target_data = demonstrations
    else:
        target_data = demonstrations[:, context_range:, :]

    x_context = context_data[:, :, 0][..., np.newaxis]
    y_context = context_data[:, :, 1][..., np.newaxis]
    x_target = target_data[:, :, 0][..., np.newaxis]
    y_target = target_data[:, :, 1][..., np.newaxis]
    return [x_context, y_context], [x_target, y_target]
