from pathlib import Path
import numpy as np
from scipy.stats import iqr

from vame.logging.logger import VameLogger
from vame.io.load_poses import read_pose_estimation_file

logger_config = VameLogger(__name__)
logger = logger_config.logger

def lowconf_cleaning(
    config: dict,
    read_from_variable: str = "position_egocentric_aligned",
    save_to_variable: str = "position_processed",
) -> None:
    """
    Clean the low-confidence data points from the dataset. Processes position data by:
     - setting low-confidence points to NaN
     - interpolating NaN points

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    read_from_variable : str, optional
        Variable to read from the dataset.
    save_to_variable : str, optional
        Variable to save the cleaned data to.

    Returns
    -------
    None
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    pose_confidence = config["pose_confidence"]
    logger.info(f"Cleaning low confidence data points. Confidence threshold: {pose_confidence}")

    for session in sessions:
        logger.info(f"Session: {session}")
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Validate the existence of the variable
        if read_from_variable not in ds.variables:
            raise KeyError(
                f"Variable '{read_from_variable}' not found in dataset. Available variables: {list(ds.variables.keys())}"
            )

        position = ds[read_from_variable].values
        cleaned_position = np.empty_like(position)
        confidence = ds["confidence"].values

        perc_interp_points = np.zeros((position.shape[1], position.shape[2], position.shape[3]))
        for individual in range(position.shape[1]):
            for keypoint in range(position.shape[2]):
                conf_series = confidence[:, individual, keypoint]
                for space in range(position.shape[3]):
                    # Set low-confidence positions to NaN
                    nan_mask = conf_series < pose_confidence
                    series = np.copy(position[:, individual, keypoint, space])
                    series[nan_mask] = np.nan

                    # Update nan_mask because the series might come with NaN values previously
                    nan_mask = np.isnan(series)

                    perc_interp_points[individual, keypoint, space] = 100 * np.sum(nan_mask) / len(nan_mask)

                    # Interpolate NaN values
                    if not nan_mask.all():
                        series[nan_mask] = np.interp(
                            np.flatnonzero(nan_mask),
                            np.flatnonzero(~nan_mask),
                            series[~nan_mask],
                        )

                    # Update the position array
                    cleaned_position[:, individual, keypoint, space] = series

        # Update the dataset with the cleaned position values
        ds[save_to_variable] = (ds[read_from_variable].dims, cleaned_position)
        ds.attrs.update({"processed_confidence": True})

        ds["percentage_low_confidence"] = (["individual", "keypoint", "space"], perc_interp_points)

        # Save the cleaned dataset to file
        cleaned_file_path = Path(project_path) / "data" / "processed" / f"{session}_processed.nc"
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="scipy",
        )

def outlier_cleaning(
    config: dict,
    read_from_variable: str = "position_processed",
    save_to_variable: str = "position_processed",
) -> None:
    """
    Clean the outliers from the dataset. Processes position data by:
     - setting outlier points to NaN
     - interpolating NaN points

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    read_from_variable : str, optional
        Variable to read from the dataset.
    save_to_variable : str, optional
        Variable to save the cleaned data to.

    Returns
    -------
    None
    """
    logger.info("Cleaning outliers with Z-score transformation and IQR cutoff.")
    project_path = config["project_path"]
    sessions = config["session_names"]

    for session in sessions:
        logger.info(f"Session: {session}")
        # Read raw session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Validate the existence of the variable
        if read_from_variable not in ds.variables:
            raise KeyError(
                f"Variable '{read_from_variable}' not found in dataset. Available variables: {list(ds.variables.keys())}"
            )

        position = np.copy(ds[read_from_variable].values)
        cleaned_position = np.copy(position)

        perc_interp_points = np.zeros((position.shape[1], position.shape[2], position.shape[3]))

        for individual in range(position.shape[1]):
            for keypoint in range(position.shape[2]):
                for space in range(position.shape[3]):
                    series = np.copy(position[:, individual, keypoint, space])

                    # Check if all values are zero, then skip
                    if np.all(series == 0):
                        continue

                    # Calculate Z-score
                    z_series = (series - np.nanmean(series)) / np.nanstd(series)

                    # Set outlier positions to NaN, based on IQR cutoff
                    if config["robust"]:
                        iqr_factor = config["iqr_factor"]
                        iqr_val = iqr(z_series)
                        outlier_mask = np.abs(z_series) > iqr_factor * iqr_val
                        z_series[outlier_mask] = np.nan
                        perc_interp_points[individual, keypoint, space] = (
                            100 * np.sum(outlier_mask) / len(outlier_mask)
                        )

                        # Interpolate NaN values
                        if not outlier_mask.all():
                            z_series[outlier_mask] = np.interp(
                                np.flatnonzero(outlier_mask),
                                np.flatnonzero(~outlier_mask),
                                z_series[~outlier_mask],
                            )

                        # Redo the z-score to remove the bias of the now-removed outliers
                        z_series = (z_series - np.nanmean(z_series)) / np.nanstd(z_series)

                    # Update the processed position array
                    cleaned_position[:, individual, keypoint, space] = z_series

        # Update the dataset with the cleaned position values
        ds[save_to_variable] = (ds[read_from_variable].dims, cleaned_position)
        ds.attrs.update({"processed_outliers": True})

        ds["percentage_iqr_outliers"] = (["individual", "keypoint", "space"], perc_interp_points)

        # Save the cleaned dataset to file
        cleaned_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="scipy",
        )
