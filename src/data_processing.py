import numpy as np
from src.utils import db_with_limits, P_refine_label_boundary


def process_data(survey, labels, bottom, prediction_UNET, ping_start, range_end, patch_size=8):
    ping_end = ping_start + 1000
    ping_slice = slice(ping_start, ping_end)
    range_slice = slice(0, range_end)

    sv_all = survey.sv.isel(ping_time=ping_slice, range=range_slice).sel(
        frequency=[18, 38, 120, 200])
    dat1_all = db_with_limits(sv_all.values, 1, 2, [18, 38, 120, 200])[0]
    labels_portion = labels.annotation.sel(category=27).isel(
        ping_time=ping_slice, range=range_slice).T.values
    modified_labels_portion = P_refine_label_boundary(ignore_zero_inside_bbox=False).__call__(
        sv_all.values, labels_portion.T, [1])[1].T
    bottom_portion = bottom.bottom_range.isel(
        ping_time=ping_slice, range=range_slice).T.values
    UNET_probabilities = prediction_UNET.sel(category=27).isel(
        ping_time=ping_slice, range=range_slice).values.T

    loader_output = {'data': [], 'center_coordinates': []}
    for i in range(0, dat1_all.shape[1] - patch_size, 1):
        for j in range(0, dat1_all.shape[2] - patch_size, 1):
            Sv_patch = dat1_all[:, i:i + patch_size, j:j + patch_size]
            center_x = i + patch_size // 2 + ping_start
            center_y = j + patch_size // 2
            loader_output['data'].append(Sv_patch)
            loader_output['center_coordinates'].append([center_y, center_x])

    center_coordinates = np.array(loader_output['center_coordinates'])
    data_patch_tensor = np.array(loader_output['data'])

    return dat1_all, labels_portion, modified_labels_portion, bottom_portion, UNET_probabilities, center_coordinates, data_patch_tensor
