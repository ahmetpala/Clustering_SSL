import matplotlib.pyplot as plt
from matplotlib import rcParams


def visualize_echogram_yeni(echogram, features_indices_=None, data_indices_=None, show_final=True):
    rcParams.update(
        {'axes.titlesize': 10, 'xtick.labelsize': 6, 'ytick.labelsize': 6})

    if data_indices_ is None:
        data_indices_ = []
    if features_indices_ is None:
        features_indices_ = []
    clusters_to_plot_features = sum(
        echogram.cluster_channel_features[:, :, idx] for idx in features_indices_)
    clusters_to_plot_data = sum(
        echogram.cluster_channel_data[:, :, idx] for idx in data_indices_)

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    im1 = axes[0].imshow(echogram.dat1_all[3].T[:-4, :-4])
    axes[0].set_title('Data in 200 kHz')
    fig.colorbar(im1, ax=axes[0], pad=0.01, fraction=0.015)

    if show_final:
        im2 = axes[1].imshow(clusters_to_plot_features *
                             (echogram.bottom_portion[:-4, :-4] != 1) > 0.25, cmap='seismic')
    else:
        im2 = axes[1].imshow(clusters_to_plot_features, cmap='seismic')
    axes[1].set_title('Clusters - SSL Features')
    fig.colorbar(im2, ax=axes[1], pad=0.01, fraction=0.015)

    im3 = axes[2].imshow(clusters_to_plot_data *
                         (echogram.bottom_portion[:-4, :-4] != 1), cmap='seismic')
    axes[2].set_title('Clusters - Raw Data')
    fig.colorbar(im3, ax=axes[2], pad=0.01, fraction=0.015)

    if show_final:
        im4 = axes[3].imshow(
            echogram.UNET_probabilities[:-4, :-4] > 0.8560, cmap='seismic')
    else:
        im4 = axes[3].imshow(
            echogram.UNET_probabilities[:-4, :-4], cmap='seismic')
    axes[3].set_title('UNET Predictions')
    fig.colorbar(im4, ax=axes[3], pad=0.01, fraction=0.015)

    im5 = axes[4].imshow(
        echogram.modified_labels_portion[:-4, :-4], cmap='seismic')
    axes[4].set_title('Sandeel Labels Annotation')
    fig.colorbar(im5, ax=axes[4], pad=0.01, fraction=0.015)

    for ax in axes[:-1]:
        ax.set(aspect=0.6)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelbottom=False)
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.grid(False)

    axes[-1].set(aspect=0.6)
    plt.show()
