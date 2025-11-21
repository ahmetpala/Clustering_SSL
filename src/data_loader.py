import xarray as xr
import pandas as pd


def load_data(year, data_location):
    year_dict = {2007: 2007205, 2008: 2008205, 2009: 2009107, 2010: 2010205, 2011: 2011206, 2013: 2013842,
                 2014: 2014807, 2015: 2015837, 2016: 2016837, 2017: 2017843, 2018: 2018823}
    code = year_dict[year]

    survey = xr.open_zarr(f'{data_location}/ACOUSTIC/GRIDDED/{code}_sv.zarr')
    labels = xr.open_zarr(
        f'{data_location}/ACOUSTIC/GRIDDED/{code}_labels.zarr')
    bottom = xr.open_zarr(
        f'{data_location}/ACOUSTIC/GRIDDED/{code}_bottom.zarr')
    prediction_UNET = xr.open_dataarray(
        f'/scratch/disk5/ahmet/data/UNET_Predictions/{code}_pred.zarr')

    objects = pd.read_csv(
        f'{data_location}/ACOUSTIC/GRIDDED/{code}_objects_parsed.csv')
    return survey, labels, bottom, prediction_UNET, objects
