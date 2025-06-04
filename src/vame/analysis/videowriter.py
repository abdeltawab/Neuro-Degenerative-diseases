import os
from pathlib import Path
import numpy as np
import cv2 as cv
import tqdm
from typing import Union
import imageio

from vame.util.auxiliary import read_config
from vame.util.cli import get_sessions_from_user_input
from vame.schemas.states import (
    save_state,
    MotifVideosFunctionSchema,
    CommunityVideosFunctionSchema,
)
from vame.logging.logger import VameLogger, TqdmToLogger
from vame.schemas.project import SegmentationAlgorithms


logger_config = VameLogger(__name__)
logger = logger_config.logger


def get_cluster_label_file_path(
    path_to_file: str,
    session: str,
    n_clusters: int,
    segmentation_algorithm: SegmentationAlgorithms,
    flag: str,
) -> str:
    """
    Get the path to the cluster label file based on segmentation algorithm.
    
    Parameters
    ----------
    path_to_file : str
        Base path to the file directory
    session : str
        Session name
    n_clusters : int
        Number of clusters (used for kmeans/hmm/gmm, ignored for dbscan)
    segmentation_algorithm : SegmentationAlgorithms
        Segmentation algorithm used
    flag : str
        Flag indicating motif or community
        
    Returns
    -------
    str
        Path to the label file
    """
    if flag == "motif":
        if segmentation_algorithm == "dbscan":
            # DBSCAN uses different naming convention without n_clusters
            label_file = f"dbscan_label_{session}.npy"
        else:
            # HMM, KMeans, and GMM use n_clusters in filename
            label_file = f"{n_clusters}_{segmentation_algorithm}_label_{session}.npy"
        
        return os.path.join(path_to_file, label_file)
    
    elif flag == "community":
        # Community logic remains the same for all algorithms
        if cohort:
            return os.path.join(
                path_to_file,
                "community",
                f"cohort_community_label_{session}.npy",
            )
        else:
            return os.path.join(
                path_to_file,
                "community", 
                f"community_label_{session}.npy",
            )


def create_cluster_videos(
    config: dict,
    path_to_file: str,
    session: str,
    n_clusters: int,
    video_type: str,
    flag: str,
    segmentation_algorithm: SegmentationAlgorithms,
    cohort: bool = True,
    output_video_type: str = ".mp4",
    tqdm_logger_stream: Union[TqdmToLogger, None] = None,
) -> None:
    """
    Generate cluster videos and save them to filesystem on project folder.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    path_to_file : str
        Path to the file.
    session : str
        Name of the session.
    n_clusters : int
        Number of clusters.
    video_type : str
        Type of input video.
    flag : str
        Flag indicating the type of video (motif or community).
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm', 'kmeans', 'gmm', or 'dbscan'.
    cohort : bool, optional
        Flag indicating cohort analysis. Defaults to True.
    output_video_type : str, optional
        Type of output video. Default is '.mp4'.
    tqdm_logger_stream : TqdmToLogger, optional
        Tqdm logger stream. Default is None.

    Returns
    -------
    None
    """
    if output_video_type not in [".mp4", ".avi"]:
        raise ValueError("Output video type must be either '.avi' or '.mp4'.")

    if flag == "motif":
        logger.info("Motif videos getting created for " + session + " ...")
        label_file_path = get_cluster_label_file_path(
            path_to_file, session, n_clusters, segmentation_algorithm, flag
        )
        labels = np.load(label_file_path)
        
    elif flag == "community":
        if cohort:
            logger.info("Cohort community videos getting created for " + session + " ...")
            labels = np.load(
                os.path.join(
                    path_to_file,
                    "community",
                    "cohort_community_label_" + session + ".npy",
                )
            )
        else:
            logger.info("Community videos getting created for " + session + " ...")
            labels = np.load(
                os.path.join(
                    path_to_file,
                    "community",
                    "community_label_" + session + ".npy",
                )
            )

    video_file_path = os.path.join(
        config["project_path"],
        "data",
        "raw",
        session + video_type,
    )
    capture = cv.VideoCapture(video_file_path)
    if not capture.isOpened():
        raise ValueError(f"Video capture could not be opened. Ensure the video file is valid.\n {video_file_path}")
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = 25  # capture.get(cv.CAP_PROP_FPS)

    cluster_start = config["time_window"] / 2
    unique_labels, count_labels = np.unique(labels, return_counts=True)
    
    # Filter out noise for DBSCAN (label -1)
    if segmentation_algorithm == "dbscan":
        # Remove noise label (-1) from unique_labels
        unique_labels = unique_labels[unique_labels != -1]
        logger.info(f"DBSCAN found {len(unique_labels)} clusters (excluding noise)")

    for cluster in unique_labels:
        logger.info("Cluster: %d" % (cluster))
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
        if not cluster_lbl.size:
            logger.info("Cluster is empty")
            continue

        if flag == "motif":
            output = os.path.join(
                path_to_file,
                "cluster_videos",
                session + f"-motif_%d{output_video_type}" % cluster,
            )
        if flag == "community":
            output = os.path.join(
                path_to_file,
                "community_videos",
                session + f"-community_%d{output_video_type}" % cluster,
            )

        if output_video_type == ".avi":
            codec = cv.VideoWriter_fourcc("M", "J", "P", "G")
            video_writer = cv.VideoWriter(output, codec, fps, (int(width), int(height)))
        elif output_video_type == ".mp4":
            video_writer = imageio.get_writer(
                output,
                fps=fps,
                codec="h264",
                macro_block_size=None,
            )

        if len(cluster_lbl) < config["length_of_motif_video"]:
            vid_length = len(cluster_lbl)
        else:
            vid_length = config["length_of_motif_video"]

        for num in tqdm.tqdm(range(vid_length), file=tqdm_logger_stream):
            idx = cluster_lbl[num]
            capture.set(1, idx + cluster_start)
            ret, frame = capture.read()
            if output_video_type == ".avi":
                video_writer.write(frame)
            elif output_video_type == ".mp4":
                video_writer.append_data(frame)
        if output_video_type == ".avi":
            video_writer.release()
        elif output_video_type == ".mp4":
            video_writer.close()
    capture.release()


def get_segmentation_path(segmentation_algorithm: SegmentationAlgorithms, n_clusters: int) -> str:
    """
    Get the segmentation path based on algorithm type.
    
    Parameters
    ----------
    segmentation_algorithm : SegmentationAlgorithms
        The segmentation algorithm used
    n_clusters : int
        Number of clusters (ignored for DBSCAN)
        
    Returns
    -------
    str
        Path suffix for the algorithm
    """
    if segmentation_algorithm == "dbscan":
        return "dbscan"
    else:
        # Works for kmeans, hmm, and gmm
        return f"{segmentation_algorithm}-{n_clusters}"


@save_state(model=MotifVideosFunctionSchema)
def motif_videos(
    config: dict,
    segmentation_algorithm: SegmentationAlgorithms,
    video_type: str = ".mp4",
    output_video_type: str = ".mp4",
    save_logs: bool = False,
) -> None:
    """
    Generate motif videos and save them to filesystem.
    Fills in the values in the "motif_videos" key of the states.json file.
    Files are saved at:
    - project_name/
        - results/
            - session/
                - model_name/
                    - segmentation_algorithm-n_clusters/ (or just 'dbscan' for DBSCAN)
                        - cluster_videos/
                            - session-motif_0.mp4
                            - session-motif_1.mp4
                            - ...

    Parameters
    ----------
    config : dict
        Configuration parameters.
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm', 'kmeans', 'gmm', or 'dbscan'.
    video_type : str, optional
        Type of video. Default is '.mp4'.
    output_video_type : str, optional
        Type of output video. Default is '.mp4'.
    save_logs : bool, optional
        Save logs to filesystem. Default is False.

    Returns
    -------
    None
    """
    try:
        tqdm_logger_stream = None
        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "motif_videos.log"
            logger_config.add_file_handler(str(log_path))
            tqdm_logger_stream = TqdmToLogger(logger=logger)
        model_name = config["model_name"]
        n_clusters = config["n_clusters"]

        logger.info(f"Creating motif videos for algorithm: {segmentation_algorithm}...")

        # Get sessions
        if config["all_data"] in ["Yes", "yes"]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                cfg=config,
                action_message="write motif videos",
            )

        if segmentation_algorithm == "dbscan":
            logger.info("Creating DBSCAN cluster videos (dynamic cluster count)")
        elif segmentation_algorithm == "gmm":
            logger.info(f"Creating GMM cluster videos with {n_clusters} components")
        else:
            logger.info("Cluster size is: %d " % n_clusters)
            
        for session in sessions:
            # Get the correct segmentation path
            segmentation_path = get_segmentation_path(segmentation_algorithm, n_clusters)
            
            path_to_file = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                segmentation_path,
                "",
            )
            if not os.path.exists(os.path.join(path_to_file, "cluster_videos")):
                os.mkdir(os.path.join(path_to_file, "cluster_videos"))

            create_cluster_videos(
                config=config,
                path_to_file=path_to_file,
                session=session,
                n_clusters=n_clusters,
                video_type=video_type,
                flag="motif",
                segmentation_algorithm=segmentation_algorithm,
                output_video_type=output_video_type,
                tqdm_logger_stream=tqdm_logger_stream,
            )
        logger.info("All videos have been created!")
    except Exception as e:
        logger.exception(f"Error in motif_videos: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()


@save_state(model=CommunityVideosFunctionSchema)
def community_videos(
    config: dict,
    segmentation_algorithm: SegmentationAlgorithms,
    cohort: bool = True,
    video_type: str = ".mp4",
    save_logs: bool = False,
    output_video_type: str = ".mp4",
) -> None:
    """
    Generate community videos and save them to filesystem on project community_videos folder.
    Fills in the values in the "community_videos" key of the states.json file.
    Files are saved at:

    1. If cohort is True:
    TODO: Add cohort analysis

    2. If cohort is False:
    - project_name/
        - results/
            - file_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/ (or just 'dbscan' for DBSCAN)
                        - community_videos/
                            - file_name-community_0.mp4
                            - file_name-community_1.mp4
                            - ...

    Parameters
    ----------
    config : dict
        Configuration parameters.
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm', 'kmeans', 'gmm', or 'dbscan'.
    cohort : bool, optional
        Flag indicating cohort analysis. Defaults to True.
    video_type : str, optional
        Type of video. Default is '.mp4'.
    save_logs : bool, optional
        Save logs to filesystem. Default is False.
    output_video_type : str, optional
        Type of output video. Default is '.mp4'.

    Returns
    -------
    None
    """
    try:
        tqdm_logger_stream = None

        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "community_videos.log"
            logger_config.add_file_handler(str(log_path))
            tqdm_logger_stream = TqdmToLogger(logger=logger)
        model_name = config["model_name"]
        n_clusters = config["n_clusters"]

        # Get sessions
        if config["all_data"] in ["Yes", "yes"]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                cfg=config,
                action_message="write community videos",
            )

        if segmentation_algorithm == "dbscan":
            logger.info("Creating DBSCAN community videos (dynamic cluster count)")
        elif segmentation_algorithm == "gmm":
            logger.info(f"Creating GMM community videos with {n_clusters} components")
        else:
            logger.info("Cluster size is: %d " % n_clusters)
            
        for session in sessions:
            # Get the correct segmentation path
            segmentation_path = get_segmentation_path(segmentation_algorithm, n_clusters)
            
            path_to_file = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                segmentation_path,
                "",
            )
            if not os.path.exists(os.path.join(path_to_file, "community_videos")):
                os.mkdir(os.path.join(path_to_file, "community_videos"))

            create_cluster_videos(
                config=config,
                path_to_file=path_to_file,
                session=session,
                n_clusters=n_clusters,
                video_type=video_type,
                flag="community",
                segmentation_algorithm=segmentation_algorithm,
                cohort=cohort,
                tqdm_logger_stream=tqdm_logger_stream,
                output_video_type=output_video_type,
            )

        logger.info("All videos have been created!")

    except Exception as e:
        logger.exception(f"Error in community_videos: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()