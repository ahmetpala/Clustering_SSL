from src.data_loader import load_data
from src.data_processing import process_data


class SurveyDataProcessor:
    def __init__(self, year, ping_start, range_end, data_location):
        self.year = year
        self.ping_start = ping_start
        self.range_end = range_end
        self.data_location = data_location

        self.survey, self.labels, self.bottom, self.prediction_UNET, self.objects = load_data(
            year, data_location)
        self.dat1_all, self.labels_portion, self.modified_labels_portion, self.bottom_portion, self.UNET_probabilities, \
            self.center_coordinates, self.data_patch_tensor = process_data(
                self.survey, self.labels, self.bottom, self.prediction_UNET, ping_start, range_end)
