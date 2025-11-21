import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve
from matplotlib import rcParams
from batch.label_transforms.ahmet_refine_label_boundary import P_refine_label_boundary

from batch.data_transforms.db_with_limits import db_with_limits
from extract_features_acoustic_Project_3_for_Clustering_Function import Extraction_for_Clustering

from scipy.ndimage import convolve


def add_cluster_channels(cluster_map, num_clusters_):
    # Replace NaN values with a placeholder value
    cluster_map = np.nan_to_num(cluster_map, nan=-1)

    cluster_channels = np.zeros(
        (cluster_map.shape[0], cluster_map.shape[1], num_clusters_), dtype=np.int32)

    # Create masks for each cluster
    cluster_masks = [(cluster_map == i) for i in range(num_clusters_)]

    # Iterate over each cluster mask
    for j, cluster_mask in enumerate(tqdm(cluster_masks, desc="Processing clusters")):
        if j == -1:  # Skip placeholder value
            continue

        # Create a kernel of 1s for convolution
        kernel = np.ones((8, 8), dtype=np.int32)

        # Convolve the cluster mask with the kernel to count the number of pixels in the 8x8 window
        convolved = convolve(cluster_mask.astype(
            np.int32), kernel, mode='constant', cval=0)

        # Assign the result to the corresponding cluster channel
        cluster_channels[:, :, j] = convolved

    return cluster_channels/64


class SurveyDataProcessor:
    def __init__(self, year, ping_start, range_end):
        self.year = year
        self.ping_start = ping_start
        self.range_end = range_end
        self.UNET_probabilities = None
        self.sorted_df = None
        self.sv_all = None
        self.dat1_all = None
        self.labels_portion = None
        self.modified_labels_portion = None
        self.bottom_portion = None
        self.data_patch_tensor = None
        self.features = None
        self.center_coordinates = None
        self.patch_size = 8
        self.load_data()
        self.process_data()

        self.final_data_numpy_features = None
        self.cluster_map_features = None
        self.cluster_channel_features = None

        self.final_data_numpy_data = None
        self.cluster_map_data = None
        self.cluster_channel_data = None

    def load_data(self):
        year_dict = {2007: 2007205, 2008: 2008205, 2009: 2009107, 2010: 2010205, 2011: 2011206, 2013: 2013842,
                     2014: 2014807, 2015: 2015837, 2016: 2016837, 2017: 2017843, 2018: 2018823}

        code = year_dict[self.year]
        data_location = f'/scratch/disk5/ahmet/data/{self.year}/{code}'

        self.survey = xr.open_zarr(
            f'{data_location}/ACOUSTIC/GRIDDED/{code}_sv.zarr')
        self.labels = xr.open_zarr(
            f'{data_location}/ACOUSTIC/GRIDDED/{code}_labels.zarr')
        self.bottom = xr.open_zarr(
            f'{data_location}/ACOUSTIC/GRIDDED/{code}_bottom.zarr')
        self.prediction_UNET = xr.open_dataarray(
            f'/scratch/disk5/ahmet/data/UNET_Predictions/{code}_pred.zarr')

        objects = pd.read_csv(
            f'{data_location}/ACOUSTIC/GRIDDED/{code}_objects_parsed.csv')
        filtered_df = objects[objects.category == 27]
        filtered_df['ping_difference'] = filtered_df['endpingindex'] - \
            filtered_df['startpingindex']
        self.sorted_df = filtered_df.sort_values(
            by='ping_difference', ascending=False)

    def process_data(self):
        ping_end = self.ping_start + 1000
        ping_slice = slice(self.ping_start, ping_end)
        range_start = 0
        range_slice = slice(range_start, self.range_end)

        self.sv_all = self.survey.sv.isel(
            ping_time=ping_slice, range=range_slice).sel(frequency=[18, 38, 120, 200])
        self.dat1_all = db_with_limits(
            self.sv_all.values, 1, 2, [18, 38, 120, 200])[0]
        self.labels_portion = self.labels.annotation.sel(category=27).isel(
            ping_time=ping_slice, range=range_slice).T.values
        self.modified_labels_portion = P_refine_label_boundary(
            ignore_zero_inside_bbox=False).__call__(self.sv_all.values, self.labels_portion.T, [1])[1].T
        self.bottom_portion = self.bottom.bottom_range.isel(
            ping_time=ping_slice, range=range_slice).T.values
        self.UNET_probabilities = self.prediction_UNET.sel(category=27).isel(
            ping_time=ping_slice, range=range_slice).values.T

        loader_output = {'data': [], 'center_coordinates': []}
        for i in range(0, self.dat1_all.shape[1] - self.patch_size, 1):
            for j in range(0, self.dat1_all.shape[2] - self.patch_size, 1):
                Sv_patch = self.dat1_all[:, i:i +
                                         self.patch_size, j:j + self.patch_size]
                center_x = i + self.patch_size // 2 + self.ping_start
                center_y = j + self.patch_size // 2
                loader_output['data'].append(Sv_patch)
                loader_output['center_coordinates'].append(
                    [center_y, center_x])

        self.center_coordinates = np.array(loader_output['center_coordinates'])
        self.data_patch_tensor = np.array(loader_output['data'])


# Train
echogram_2017_1 = SurveyDataProcessor(
    year=2017, ping_start=1289700, range_end=328)
echogram_2017_2 = SurveyDataProcessor(
    year=2017, ping_start=613265, range_end=328)
echogram_2017_3 = SurveyDataProcessor(
    year=2017, ping_start=1107958, range_end=328)
echogram_2017_4 = SurveyDataProcessor(
    year=2017, ping_start=621254, range_end=328)
echogram_2017_5 = SurveyDataProcessor(
    year=2017, ping_start=628792, range_end=328)
echogram_2017_6 = SurveyDataProcessor(
    year=2017, ping_start=621584, range_end=328)
echogram_2017_7 = SurveyDataProcessor(
    year=2017, ping_start=696195, range_end=328)
echogram_2017_8 = SurveyDataProcessor(
    year=2017, ping_start=229855, range_end=328)
echogram_2017_9 = SurveyDataProcessor(
    year=2017, ping_start=1130000, range_end=328)  # layer
echogram_2017_10 = SurveyDataProcessor(
    year=2017, ping_start=1310000, range_end=328)  # layer
echogram_2017_11 = SurveyDataProcessor(
    year=2017, ping_start=1340000, range_end=328)  # layer
echogram_2017_12 = SurveyDataProcessor(
    year=2017, ping_start=1400000, range_end=328)  # layer

# Test
echogram_2017_13 = SurveyDataProcessor(
    year=2017, ping_start=1347000, range_end=328)  # layer
echogram_2017_14 = SurveyDataProcessor(
    year=2017, ping_start=1377000, range_end=328)  # layer
echogram_2017_15 = SurveyDataProcessor(
    year=2017, ping_start=1397000, range_end=328)  # layer
echogram_2017_16 = SurveyDataProcessor(
    year=2017, ping_start=1407000, range_end=328)  # layer
echogram_2017_17 = SurveyDataProcessor(
    year=2017, ping_start=321624, range_end=328)
echogram_2018 = SurveyDataProcessor(
    year=2018, ping_start=684985, range_end=328)
echogram_2018_test1 = SurveyDataProcessor(
    year=2018, ping_start=433954, range_end=328)
echogram_2018_test2 = SurveyDataProcessor(
    year=2018, ping_start=822681, range_end=328)


# List to store all tensors
tensor_list = [
    echogram_2017_1.data_patch_tensor,
    echogram_2017_2.data_patch_tensor,
    echogram_2017_3.data_patch_tensor,
    echogram_2017_4.data_patch_tensor,
    echogram_2017_5.data_patch_tensor,
    echogram_2017_6.data_patch_tensor,
    echogram_2017_7.data_patch_tensor,
    echogram_2017_8.data_patch_tensor,
    echogram_2017_9.data_patch_tensor,
    echogram_2017_10.data_patch_tensor,
    echogram_2017_11.data_patch_tensor,
    echogram_2017_12.data_patch_tensor,
    echogram_2017_13.data_patch_tensor,
    echogram_2017_14.data_patch_tensor,
    echogram_2017_15.data_patch_tensor,
    echogram_2017_16.data_patch_tensor,
    echogram_2017_17.data_patch_tensor,
    echogram_2018.data_patch_tensor,
    echogram_2018_test1.data_patch_tensor,
    echogram_2018_test2.data_patch_tensor,
]

# Concatenate all tensors along the first axis
all_tensor = np.concatenate(tensor_list, axis=0)

all_data = all_tensor.reshape(all_tensor.shape[0], -1)

all_features = Extraction_for_Clustering(all_tensor)


echograms = [
    echogram_2017_1,
    echogram_2017_2,
    echogram_2017_3,
    echogram_2017_4,
    echogram_2017_5,
    echogram_2017_6,
    echogram_2017_7,
    echogram_2017_8,
    echogram_2017_9,
    echogram_2017_10,
    echogram_2017_11,
    echogram_2017_12,
    echogram_2017_13,
    echogram_2017_14,
    echogram_2017_15,
    echogram_2017_16,
    echogram_2017_17,
    echogram_2018,
    echogram_2018_test1,
    echogram_2018_test2]

# Number of samples in each echogram's features
num_samples = 317440

# Update features for each echogram
for i, echogram in enumerate(echograms):
    start_idx = i * num_samples
    end_idx = start_idx + num_samples
    echogram.features = all_features[start_idx:end_idx, :]


# Applying k-means clustering
num_clusters = 350
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans = kmeans.fit(all_features)  # USING All DATA TO TRAIN KMEANS
kmeans_classes = kmeans.predict(all_features)


def update_echogram_attributes(echogram, kmeans_class_array, kmeans_start_index, kmeans_end_index, input_type='features', number_of_clusters=60):
    # Concatenate center coordinates with kmeans classes
    final_data_numpy = np.concatenate([echogram.center_coordinates, kmeans_class_array[kmeans_start_index:kmeans_end_index].reshape(
        len(kmeans_class_array[kmeans_start_index:kmeans_end_index]), 1)], axis=1)
    final_data_numpy[:, 1] = final_data_numpy[:, 1] - \
        min(final_data_numpy[:, 1]) + 4

    # Extract x and y coordinates and clusters
    y_coords = final_data_numpy[:, 0].astype(int)
    x_coords = final_data_numpy[:, 1].astype(int)
    clusters = final_data_numpy[:, 2].astype(int)

    # Initialize a 2D numpy array with a placeholder value, e.g., -1 for unassigned
    cluster_map = np.full((np.max(y_coords) + 1, np.max(x_coords) + 1), -1)

    # Populate the 2D array with cluster values
    cluster_map[y_coords, x_coords] = clusters

    # Generate cluster channels
    cluster_channel = add_cluster_channels(
        cluster_map, num_clusters_=number_of_clusters)

    # Update echogram attributes
    if input_type == 'features':
        echogram.final_data_numpy_features = final_data_numpy
        echogram.cluster_map_features = cluster_map
        echogram.cluster_channel_features = cluster_channel
    elif input_type == 'data':
        echogram.final_data_numpy_data = final_data_numpy
        echogram.cluster_map_data = cluster_map
        echogram.cluster_channel_data = cluster_channel


# Number of samples in each echogram's features
num_samples = 317440

# Update features for each echogram
for i, echogram in enumerate(echograms):
    start_idx = i * num_samples
    end_idx = start_idx + num_samples
    update_echogram_attributes(echogram, kmeans_classes, start_idx,
                               end_idx, input_type='features', number_of_clusters=num_clusters)


# Initialize an empty list to collect cluster map features
all_cluster_features = []

# Loop through each echogram to extract and concatenate cluster map features
for echogram in echograms:
    clusters = echogram.cluster_map_features[echogram.modified_labels_portion[:-4, :-4] == 1]
    all_cluster_features.append(clusters)

# Concatenate all collected cluster map features
all_cluster_features = np.concatenate(all_cluster_features)

# Get unique clusters and their counts
sandeel_clusters = np.array(
    np.unique(all_cluster_features, return_counts=True)).T

# Sort clusters by count in descending order
sandeel_clusters_sorted = sandeel_clusters[sandeel_clusters[:, 1].argsort()[
    ::-1]]
sandeel_clusters_sorted = sandeel_clusters_sorted[sandeel_clusters_sorted[:, 0] != -1]
sandeel_clusters_sorted

yeni_sorted = sandeel_clusters_sorted[sandeel_clusters_sorted[:, 1] > 100]

# Initialize empty lists to collect data
cluster_channel_features_list = []
labels_portion_list = []
UNET_probabilities_list = []
bottom_portion_list = []

# Loop through each echogram to collect the features
for echogram in echograms:
    cluster_channel_features_list.append(echogram.cluster_channel_features)
    labels_portion_list.append(echogram.modified_labels_portion[:-4, :-4])
    UNET_probabilities_list.append(echogram.UNET_probabilities[:-4, :-4])
    bottom_portion_list.append(echogram.bottom_portion[:-4, :-4])

# Concatenate collected features along the specified axis
cluster_channels_all = np.concatenate(cluster_channel_features_list, axis=1)
labels_portion_all = np.concatenate(labels_portion_list, axis=1)
UNET_all = np.concatenate(UNET_probabilities_list, axis=1)
bottom_portion_all = np.concatenate(bottom_portion_list, axis=1)


def calculate_individual_F1_scores(sorted_clusters, all_cluster_channels, all_labels_portion, all_bottom_portion):
    # Initialize arrays for storing results
    f1_scores_list, precision_list, recall_list, thresholds_list = [], [], [], []
    # Iterate over the remaining clusters
    for k in range(len(sorted_clusters)):
        # Filtering out bottom
        clusters_final_channel = all_cluster_channels[:, :,
                                                      sorted_clusters[k, 0]][all_bottom_portion != 1]
        labels_flat = all_labels_portion[all_bottom_portion != 1].flatten()
        clusters_final_flat = clusters_final_channel.flatten()
        real_labels = (labels_flat == 1).astype(int)

        precision_, recall_, thresholds_ = precision_recall_curve(
            real_labels, clusters_final_flat)
        f1_scores_ = []

        for p, r in zip(precision_, recall_):
            if p + r == 0:
                f1_scores_.append(0)
            else:
                f1_scores_.append(2 * (p * r) / (p + r))

        if len(f1_scores_) > 0:
            max_f1_index_ = np.argmax(f1_scores_)
            f1_scores_list.append(f1_scores_[max_f1_index_])
            precision_list.append(precision_[max_f1_index_])
            recall_list.append(recall_[max_f1_index_])
            best_threshold = thresholds_[
                max_f1_index_ - 1] if max_f1_index_ > 0 else thresholds_[max_f1_index_]
            thresholds_list.append(best_threshold)

    return np.array(f1_scores_list).reshape(-1, 1)


last_index_training = 996 * 12

f1_scores_ara_features = calculate_individual_F1_scores(
    yeni_sorted, cluster_channels_all[:, :last_index_training, :], labels_portion_all[:, :last_index_training], bottom_portion_all[:, :last_index_training])


sandeel_clusters_sorted_final_features = np.hstack(
    (yeni_sorted, f1_scores_ara_features))
sorted_indices = np.argsort(sandeel_clusters_sorted_final_features[:, 2])
sandeel_clusters_sorted_final_features = sandeel_clusters_sorted_final_features[
    sorted_indices[::-1]]
sandeel_clusters_sorted_final_features


def analyze_clusters(sorted_data, input_cluster_channels_all, input_labels_portion_all, input_bottom_portion_all):
    # Initialize arrays for storing results
    global best_f1_score
    f1_scores_list, precision_list, recall_list, thresholds_list, indices_to_sum_list = [], [], [], [], []

    # Initialize the indices_to_sum with the first element
    indices_to_sum = [int(sorted_data[0, 0])]

    # Function to calculate F1 score for a given combination of clusters
    def calculate_f1(indices):
        clusters_to_plot_all = sum(
            input_cluster_channels_all[:, :, idx][input_bottom_portion_all != 1] for idx in indices)
        labels_values_flat = input_labels_portion_all[input_bottom_portion_all != 1].flatten(
        )
        clusters_to_plot_flat = clusters_to_plot_all.flatten()
        true_labels = (labels_values_flat == 1).astype(int)

        precision, recall, thresholds = precision_recall_curve(
            true_labels, clusters_to_plot_flat)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        max_f1_index = np.argmax(f1_scores)

        best_threshold = thresholds[max_f1_index -
                                    1] if max_f1_index > 0 else thresholds[max_f1_index]
        return f1_scores[max_f1_index], precision[max_f1_index], recall[max_f1_index], best_threshold

    # Calculate the initial F1 score
    max_f1_score, best_precision, best_recall, best_threshold = calculate_f1(
        indices_to_sum)
    f1_scores_list.append(max_f1_score)
    precision_list.append(best_precision)
    recall_list.append(best_recall)
    thresholds_list.append(best_threshold)
    indices_to_sum_list.append(indices_to_sum.copy())

    # Iterate to find the best combination of clusters
    remaining_clusters = set(range(1, len(sorted_data)))
    while remaining_clusters:
        best_improvement = 0
        best_candidate = None

        for candidate in remaining_clusters:
            current_indices = indices_to_sum + [int(sorted_data[candidate, 0])]
            f1_score, precision, recall, threshold = calculate_f1(
                current_indices)

            if f1_score > max_f1_score:
                improvement = f1_score - max_f1_score
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = candidate
                    best_f1_score = f1_score
                    best_precision = precision
                    best_recall = recall
                    best_threshold = threshold

        if best_candidate is not None:
            indices_to_sum.append(int(sorted_data[best_candidate, 0]))
            remaining_clusters.remove(best_candidate)
            max_f1_score = best_f1_score
            f1_scores_list.append(max_f1_score)
            precision_list.append(best_precision)
            recall_list.append(best_recall)
            thresholds_list.append(best_threshold)
            indices_to_sum_list.append(indices_to_sum.copy())
            print('One added!')
        else:
            break

    # Find the combination with the maximized F1 score
    max_f1_index = np.argmax(f1_scores_list)
    best_combination = indices_to_sum_list[max_f1_index]
    max_f1_score = f1_scores_list[max_f1_index]
    best_precision = precision_list[max_f1_index]
    best_recall = recall_list[max_f1_index]
    best_threshold = thresholds_list[max_f1_index]

    print(f'Best Combination: {best_combination}')
    print(f'Maximized F1 Score: {max_f1_score:.4f}')
    print(f'Precision at max F1: {best_precision:.4f}')
    print(f'Recall at max F1: {best_recall:.4f}')
    print(f'Best Threshold: {best_threshold:.4f}')

    # Plot the F1 scores for each combination
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_scores_list) + 1),
             f1_scores_list, marker='o', label='F1 Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Maximized F1 Score')
    plt.title('Maximized F1 Score for Each Combination of Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the precision-recall curve for the best combination
    clusters_to_plot_best = sum(
        input_cluster_channels_all[:, :, idx] for idx in best_combination)
    labels_values_flat_best = input_labels_portion_all.flatten()
    clusters_to_plot_flat_best = clusters_to_plot_best.flatten()
    true_labels_best = (labels_values_flat_best == 1).astype(int)
    precision_best, recall_best, _ = precision_recall_curve(
        true_labels_best, clusters_to_plot_flat_best)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_best, precision_best, label='Precision-Recall curve')
    plt.scatter(best_recall, best_precision, color='red',
                label=f'Max F1 Score: {max_f1_score:.2f}', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


def analyze_clusters_N_steps(sorted_data, input_cluster_channels_all, input_labels_portion_all, input_bottom_portion_all, steps, save_name):
    # Initialize arrays for storing results
    global best_f1_score
    f1_scores_list, precision_list, recall_list, thresholds_list, indices_to_sum_list = [], [], [], [], []

    # Initialize the indices_to_sum with the first element
    indices_to_sum = [int(sorted_data[0, 0])]

    # Function to calculate F1 score for a given combination of clusters
    def calculate_f1(indices):
        clusters_to_plot_all = sum(
            input_cluster_channels_all[:, :, idx][input_bottom_portion_all != 1] for idx in indices)
        labels_values_flat = input_labels_portion_all[input_bottom_portion_all != 1].flatten(
        )
        clusters_to_plot_flat = clusters_to_plot_all.flatten()
        true_labels = (labels_values_flat == 1).astype(int)

        precision, recall, thresholds = precision_recall_curve(
            true_labels, clusters_to_plot_flat)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        max_f1_index = np.argmax(f1_scores)

        best_threshold = thresholds[max_f1_index -
                                    1] if max_f1_index > 0 else thresholds[max_f1_index]
        return f1_scores[max_f1_index], precision[max_f1_index], recall[max_f1_index], best_threshold

    # Calculate the initial F1 score
    max_f1_score, best_precision, best_recall, best_threshold = calculate_f1(
        indices_to_sum)
    f1_scores_list.append(max_f1_score)
    precision_list.append(best_precision)
    recall_list.append(best_recall)
    thresholds_list.append(best_threshold)
    indices_to_sum_list.append(indices_to_sum.copy())

    # Iterate to find the best combination of clusters, continue for 'steps' iterations
    remaining_clusters = set(range(1, len(sorted_data)))
    best_combination_index = 0  # Track the step where the best combination is found

    for step in range(steps):
        best_improvement = 0
        best_candidate = None
        # To track the least bad candidate when no improvement is found
        max_negative_improvement = None

        for candidate in remaining_clusters:
            current_indices = indices_to_sum + [int(sorted_data[candidate, 0])]
            f1_score, precision, recall, threshold = calculate_f1(
                current_indices)

            improvement = f1_score - max_f1_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = candidate
                best_f1_score = f1_score
                best_precision = precision
                best_recall = recall
                best_threshold = threshold

            # Track the cluster that gives the least decrease (negative improvement)
            if max_negative_improvement is None or improvement > max_negative_improvement:
                max_negative_improvement = improvement
                best_negative_candidate = candidate

        if best_candidate is not None:
            indices_to_sum.append(int(sorted_data[best_candidate, 0]))
            remaining_clusters.remove(best_candidate)
            max_f1_score = best_f1_score
            f1_scores_list.append(max_f1_score)
            precision_list.append(best_precision)
            recall_list.append(best_recall)
            thresholds_list.append(best_threshold)
            indices_to_sum_list.append(indices_to_sum.copy())
            best_combination_index = len(f1_scores_list) - 1
            print(f'Step {step + 1}: One added! F1: {max_f1_score:.4f}')
        else:
            # No improvement, add the cluster that causes the least negative impact
            indices_to_sum.append(int(sorted_data[best_negative_candidate, 0]))
            remaining_clusters.remove(best_negative_candidate)
            f1_scores_list.append(max_f1_score)
            precision_list.append(best_precision)
            recall_list.append(best_recall)
            thresholds_list.append(best_threshold)
            indices_to_sum_list.append(indices_to_sum.copy())
            print(
                f'Step {step + 1}: No improvement, adding cluster with least decrease in F1.')

    # Plot the F1 scores for each combination
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_scores_list) + 1),
             f1_scores_list, marker='o', label='F1 Score')
    plt.axvline(x=best_combination_index + 1, color='red', linestyle='--',
                label=f'Best F1 Score at Step {best_combination_index + 1}')
    plt.xlabel('Number of Steps')
    plt.ylabel('Maximized F1 Score')
    # plt.title('F1 Score Trend for Each Step')
    plt.legend()
    plt.grid(True)
    # Set x-ticks to show every integer value
    plt.xticks(range(1, len(f1_scores_list) + 1))
    plt.savefig(f'{save_name}_cluster_selection_steps.jpg', dpi=600)
    plt.show()

    # Plot the precision-recall curve for the best combination
    best_combination = indices_to_sum_list[best_combination_index]
    clusters_to_plot_best = sum(
        input_cluster_channels_all[:, :, idx] for idx in best_combination)
    labels_values_flat_best = input_labels_portion_all.flatten()
    clusters_to_plot_flat_best = clusters_to_plot_best.flatten()
    true_labels_best = (labels_values_flat_best == 1).astype(int)
    precision_best, recall_best, _ = precision_recall_curve(
        true_labels_best, clusters_to_plot_flat_best)

    # Save the data required for both plots as a dictionary
    save_data = {
        'f1_scores': f1_scores_list,
        'best_combination_index': best_combination_index,
        'indices_to_sum_list': indices_to_sum_list,
        'precision_list': precision_list,
        'recall_list': recall_list,
        'precision_best': precision_best,
        'recall_best': recall_best,
        'best_combination': indices_to_sum_list[best_combination_index]
    }

    np.save(f'{save_name}_plot_data.npy', save_data)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_best, precision_best, label='Precision-Recall curve')
    plt.scatter(recall_list[best_combination_index], precision_list[best_combination_index],
                color='red', label=f'Max F1 Score: {f1_scores_list[best_combination_index]:.2f}', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve for Best Combination')
    plt.legend()
    plt.show()

    print(
        f'Best Combination (at step {best_combination_index + 1}): {best_combination}')
    print(f'Maximized F1 Score: {f1_scores_list[best_combination_index]:.4f}')
    print(f'Precision at max F1: {precision_list[best_combination_index]:.4f}')
    print(f'Recall at max F1: {recall_list[best_combination_index]:.4f}')


yeni_final_list = sandeel_clusters_sorted_final_features[
    sandeel_clusters_sorted_final_features[:, 2] > 0.078]


analyze_clusters_N_steps(yeni_final_list, cluster_channels_all[:, :last_index_training, :], labels_portion_all[:,
                         :last_index_training], bottom_portion_all[:, :last_index_training], steps=17, save_name='SSL_features_cluster_merging')


# Applying k-means clustering
num_clusters = 350
kmeans_data = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans_data = kmeans_data.fit(all_data)  # SELECTING ALL DATA FOR TRAINING
kmeans_classes_data = kmeans_data.predict(all_data)


# Number of samples in each echogram's features
num_samples = 317440

# Update features for each echogram
for i, echogram in enumerate(echograms):
    start_idx = i * num_samples
    end_idx = start_idx + num_samples
    update_echogram_attributes(echogram, kmeans_classes_data, start_idx,
                               end_idx, input_type='data', number_of_clusters=num_clusters)


# Initialize an empty list to collect cluster map features
all_cluster_data = []

# Loop through each echogram to extract and concatenate cluster map features
for echogram in echograms:
    clusters_data_ara = echogram.cluster_map_data[echogram.modified_labels_portion[:-4, :-4] == 1]
    all_cluster_data.append(clusters_data_ara)

# Concatenate all collected cluster map features
all_cluster_data = np.concatenate(all_cluster_data)

# Get unique clusters and their counts
sandeel_clusters_data = np.array(
    np.unique(all_cluster_data, return_counts=True)).T

# Sort clusters by count in descending order
sandeel_clusters_sorted_data = sandeel_clusters_data[sandeel_clusters_data[:, 1].argsort()[
    ::-1]]
sandeel_clusters_sorted_data = sandeel_clusters_sorted_data[
    sandeel_clusters_sorted_data[:, 0] != -1]
sandeel_clusters_sorted_data


# Initialize empty lists to collect data
cluster_channel_data_list = []

# Loop through each echogram to collect the features
for echogram in echograms:
    cluster_channel_data_list.append(echogram.cluster_channel_data)

# Concatenate collected features along the specified axis
cluster_channels_all_data = np.concatenate(cluster_channel_data_list, axis=1)


last_index_training = 996 * 12

f1_scores_ara_data = calculate_individual_F1_scores(
    sandeel_clusters_sorted_data, cluster_channels_all_data[:, :last_index_training, :], labels_portion_all[:, :last_index_training], bottom_portion_all[:, :last_index_training])


sandeel_clusters_sorted_final_data = np.hstack(
    (sandeel_clusters_sorted_data, f1_scores_ara_data))
sorted_indices_data = np.argsort(sandeel_clusters_sorted_final_data[:, 2])
sandeel_clusters_sorted_final_data = sandeel_clusters_sorted_final_data[
    sorted_indices_data[::-1]]
sandeel_clusters_sorted_final_data

yeni_final_list_data = sandeel_clusters_sorted_final_data[
    sandeel_clusters_sorted_final_data[:, 2] > 0.078]
yeni_final_list_data

analyze_clusters_N_steps(sandeel_clusters_sorted_final_data[sandeel_clusters_sorted_final_data[:, 1] > 100], cluster_channels_all_data[:, :last_index_training, :],
                         labels_portion_all[:, :last_index_training], bottom_portion_all[:, :last_index_training], steps=20, save_name='Raw_Data_cluster_merging')

analyze_clusters(sandeel_clusters_sorted_final_data[sandeel_clusters_sorted_final_data[:, 1] > 100],
                 cluster_channels_all_data[:, :last_index_training, :], labels_portion_all[:, :last_index_training], bottom_portion_all[:, :last_index_training])


def visualize_echogram_yeni(echogram, features_indices_=None, data_indices_=None, show_final=True):
    # Configure global font sizes
    rcParams.update({
        'axes.titlesize': 10, 'xtick.labelsize': 6, 'ytick.labelsize': 6})

    # Sum the cluster channels
    if data_indices_ is None:
        data_indices_ = []
    if features_indices_ is None:
        features_indices_ = []
    clusters_to_plot_features = sum(
        echogram.cluster_channel_features[:, :, idx] for idx in features_indices_)
    clusters_to_plot_data = sum(
        echogram.cluster_channel_data[:, :, idx] for idx in data_indices_)

    # Subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    im1 = axes[0].imshow(echogram.dat1_all[3].T[:-4, :-4])
    axes[0].set_title('Data in 200 kHz')
    fig.colorbar(im1, ax=axes[0], pad=0.01, fraction=0.015)

    if show_final:
        im2 = axes[1].imshow(clusters_to_plot_features*(echogram.bottom_portion[:-4, :-4] != 1)
                             # *(echogram.bottom_portion[:-4,:-4]!=1)
                             > 0.25, cmap='seismic')
    else:
        # *(echogram.bottom_portion[:-4,:-4]!=1)
        im2 = axes[1].imshow(clusters_to_plot_features, cmap='seismic')
    axes[1].set_title('Clusters - SSL Features')
    fig.colorbar(im2, ax=axes[1], pad=0.01, fraction=0.015)

    im3 = axes[2].imshow(
        clusters_to_plot_data*(echogram.bottom_portion[:-4, :-4] != 1), cmap='seismic')
    axes[2].set_title('Clusters - Raw Data')
    fig.colorbar(im3, ax=axes[2], pad=0.01, fraction=0.015)

    if show_final:
        im4 = axes[3].imshow(echogram.UNET_probabilities[:-4, :-4]
                             > 0.8560, cmap='seismic')  # >0.7930
    else:
        im4 = axes[3].imshow(
            echogram.UNET_probabilities[:-4, :-4], cmap='seismic')  # >0.7930
    axes[3].set_title('UNET Predictions')
    fig.colorbar(im4, ax=axes[3], pad=0.01, fraction=0.015)

    im5 = axes[4].imshow(
        echogram.modified_labels_portion[:-4, :-4], cmap='seismic')
    axes[4].set_title('Sandeel Labels Annotation')
    fig.colorbar(im5, ax=axes[4], pad=0.01, fraction=0.015)

    # Aspect ratios and hide spines
    for ax in axes[:-1]:  # Iterate over all but the last axis
        ax.set(aspect=0.6)
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.tick_params(labelbottom=False)  # Hide the x-axis tick labels
        ax.xaxis.set_ticks_position('none')  # Remove x-axis ticks
        ax.xaxis.grid(False)  # Disable the grid lines for x-axis

    axes[-1].set(aspect=0.6)  # Set aspect ratio for the last axis

    plt.show()
