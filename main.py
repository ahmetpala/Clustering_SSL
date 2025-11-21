from src.survey_data_processor import SurveyDataProcessor
from src.clustering import apply_kmeans
from src.utils import add_cluster_channels
from src.visualization import visualize_echogram_yeni
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

data_location = '/scratch/disk5/ahmet/data'

# Example usage
echogram = SurveyDataProcessor(
    year=2017, ping_start=1289700, range_end=328, data_location=data_location)

# Apply clustering
features = echogram.data_patch_tensor.reshape(
    echogram.data_patch_tensor.shape[0], -1)
kmeans = apply_kmeans(features, num_clusters=350)

# Visualize results
visualize_echogram_yeni(echogram)
