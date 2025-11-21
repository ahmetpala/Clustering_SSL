import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve, binary_closing


def add_cluster_channels(cluster_map, num_clusters_):
    cluster_map = np.nan_to_num(cluster_map, nan=-1)
    cluster_channels = np.zeros(
        (cluster_map.shape[0], cluster_map.shape[1], num_clusters_), dtype=np.int32)
    cluster_masks = [(cluster_map == i) for i in range(num_clusters_)]

    for j, cluster_mask in enumerate(tqdm(cluster_masks, desc="Processing clusters")):
        if j == -1:
            continue
        kernel = np.ones((8, 8), dtype=np.int32)
        convolved = convolve(cluster_mask.astype(
            np.int32), kernel, mode='constant', cval=0)
        cluster_channels[:, :, j] = convolved

    return cluster_channels / 64


def db(data, eps=1e-10):
    """ Decibel (log) transform """
    return 10 * np.log10(data + eps)


def db_with_limits(data, labels, echogram, frequencies, limit_low=-75, limit_high=0):
    data = db(data)
    data[data > limit_high] = limit_high
    data[data < limit_low] = limit_low
    return data, labels, echogram, frequencies


class P_refine_label_boundary():
    def __init__(self,
                 frequencies=[18, 38, 120, 200],
                 threshold_freq=200,
                 threshold_val=[1e-7, 1e-4],
                 ignore_val=-100,
                 ignore_zero_inside_bbox=True
                 ):
        self.frequencies = frequencies
        self.threshold_freq = threshold_freq
        self.threshold_val = threshold_val
        self.ignore_val = ignore_val
        self.ignore_zero_inside_bbox = ignore_zero_inside_bbox

    def __call__(self, data, labels, echogram):
        '''
        Refine existing labels based on thresholding with respect to pixel values in image.
        :param data: (numpy.array) Image (C, H, W)
        :param labels: (numpy.array) Labels corresponding to image (H, W)
        :param echogram: (Echogram object) Echogram
        :param threshold_freq: (int) Image frequency channel that is used for thresholding
        :param threshold_val: (float) Threshold value that is applied to image for assigning new labels
        :param ignore_val: (int) Ignore value (specific label value) instructs loss function not to compute gradients for these pixels
        :param ignore_zero_inside_bbox: (bool) labels==1 that is relabeled to 0 are set to ignore_value if True, 0 if False
        :return: data, new_labels, echogram
        '''

        closing = np.array([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0]
        ])

        if self.ignore_val == None:
            self.ignore_val = 0

        # Set new label for all pixels inside bounding box that are below threshold value
        if self.ignore_zero_inside_bbox:
            label_below_threshold = self.ignore_val
        else:
            label_below_threshold = 0

        # Get refined label masks
        freq_idx = self.frequencies.index(self.threshold_freq)

        # Relabel
        new_labels = labels.copy()

        mask_threshold = (labels != 0) & (labels != self.ignore_val) & \
            (data[freq_idx, :, :] > self.threshold_val[0]) & (
            data[freq_idx, :, :] < self.threshold_val[1])
        # Padding
        mask_threshold = np.pad(mask_threshold, ((3, 3), (3, 3)), 'constant')

        mask_threshold_closed = binary_closing(
            mask_threshold, structure=closing)

        # Remove Padding
        mask_threshold_closed = mask_threshold_closed[3:-3, 3:-3]

        mask = (labels != 0) & (labels != self.ignore_val) & (
            mask_threshold_closed == 0)

        new_labels[mask] = label_below_threshold
        new_labels[labels == self.ignore_val] = self.ignore_val

        return data, new_labels, echogram
