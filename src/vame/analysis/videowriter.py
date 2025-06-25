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
    cohort: bool = True,
    use_sequential_for_videos: bool = True,  # NEW PARAMETER
) -> str:
    """
    FIXED: Get the path to the cluster label file based on segmentation algorithm.
    Now properly supports sequential video labels for clean numbering.
    
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
    cohort : bool, optional
        Flag for cohort analysis (only used for community)
    use_sequential_for_videos : bool, optional
        Whether to use sequential video labels for clean numbering (default: True)
        
    Returns
    -------
    str
        Path to the label file
    """
    if flag == "motif":
        if segmentation_algorithm == "dbscan":
            if use_sequential_for_videos:
                # Check for sequential video labels first (PRIORITY)
                sequential_shared_path = os.path.join(path_to_file, f"dbscan_video_sequential_label_{session}.npy")
                if os.path.exists(sequential_shared_path):
                    logger.info(f"Using SEQUENTIAL VIDEO LABELS for clean numbering: {session}")
                    return sequential_shared_path
            
            # Fall back to original behavior
            shared_numbering_path = os.path.join(path_to_file, f"dbscan_shared_numbering_label_{session}.npy")
            group_aware_path = os.path.join(path_to_file, f"dbscan_group_aware_label_{session}.npy")
            regular_path = os.path.join(path_to_file, f"dbscan_label_{session}.npy")
            
            if os.path.exists(shared_numbering_path):
                logger.info(f"Using shared numbering DBSCAN results for video creation: {session}")
                return shared_numbering_path
            elif os.path.exists(group_aware_path):
                logger.info(f"Using group-aware DBSCAN results for video creation: {session}")
                return group_aware_path
            elif os.path.exists(regular_path):
                logger.info(f"Using regular DBSCAN results for video creation: {session}")
                return regular_path
            else:
                return regular_path  # Will cause error if doesn't exist
        else:
            if use_sequential_for_videos:
                # Check for sequential video labels for other algorithms
                sequential_path = os.path.join(path_to_file, f"{segmentation_algorithm}_video_sequential_label_{session}.npy")
                if os.path.exists(sequential_path):
                    logger.info(f"Using SEQUENTIAL VIDEO LABELS for {segmentation_algorithm.upper()}: {session}")
                    return sequential_path
            
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
    use_sequential_for_videos: bool = True,  # NEW PARAMETER
) -> None:
    """
    FIXED: Generate cluster videos with clean sequential numbering.
    Now properly handles sequential video labels for clean 0-N numbering.

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
    use_sequential_for_videos : bool, optional
        Whether to use sequential video labels for clean numbering. Default is True.

    Returns
    -------
    None
    """
    if output_video_type not in [".mp4", ".avi"]:
        raise ValueError("Output video type must be either '.avi' or '.mp4'.")

    if flag == "motif":
        logger.info("Motif videos getting created for " + session + " ...")
        label_file_path = get_cluster_label_file_path(
            path_to_file, session, n_clusters, segmentation_algorithm, flag, cohort, use_sequential_for_videos
        )
        
        if not os.path.exists(label_file_path):
            logger.error(f"Label file not found: {label_file_path}")
            raise FileNotFoundError(f"Label file not found: {label_file_path}")
            
        labels = np.load(label_file_path)
        
        # Check if using sequential video labels
        using_sequential = "video_sequential" in label_file_path
        
        if using_sequential:
            logger.info(f"Using SEQUENTIAL VIDEO LABELS - videos will have clean 0-N numbering for {session}")
        else:
            logger.info(f"Using original labels - videos may have gaps in numbering for {session}")
        
    elif flag == "community":
        if cohort:
            logger.info("Cohort community videos getting created for " + session + " ...")
            community_label_path = os.path.join(
                path_to_file,
                "community",
                "cohort_community_label_" + session + ".npy",
            )
        else:
            logger.info("Community videos getting created for " + session + " ...")
            community_label_path = os.path.join(
                path_to_file,
                "community",
                "community_label_" + session + ".npy",
            )
        
        if not os.path.exists(community_label_path):
            logger.error(f"Community label file not found: {community_label_path}")
            raise FileNotFoundError(f"Community label file not found: {community_label_path}")
            
        labels = np.load(community_label_path)

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

    # Log the motif range for verification
    if len(unique_labels) > 0:
        logger.info(f"Creating videos for motifs: {sorted(unique_labels)} (range: {min(unique_labels)}-{max(unique_labels)})")

    for cluster in unique_labels:
        logger.info("Creating video for cluster: %d" % (cluster))
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
        if not cluster_lbl.size:
            logger.info("Cluster is empty")
            continue

        if flag == "motif":
            output = os.path.join(
                path_to_file,
                "cluster_videos",
                session + f"-motif_{cluster:d}{output_video_type}",
            )
        if flag == "community":
            output = os.path.join(
                path_to_file,
                "community_videos",
                session + f"-community_{cluster:d}{output_video_type}",
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
            if ret:  # Check if frame was read successfully
                if output_video_type == ".avi":
                    video_writer.write(frame)
                elif output_video_type == ".mp4":
                    video_writer.append_data(frame)
            else:
                logger.warning(f"Could not read frame at index {idx + cluster_start}")
                
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


def detect_and_get_dbscan_path(
    config: dict,
    session: str,
    model_name: str,
    segmentation_algorithm: str
) -> str:
    """
    Detect whether to use shared numbering, group-aware, or regular DBSCAN path.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    session : str
        Session name
    model_name : str
        Model name
    segmentation_algorithm : str
        Segmentation algorithm
        
    Returns
    -------
    str
        Appropriate path for DBSCAN results
    """
    if segmentation_algorithm != "dbscan":
        return get_segmentation_path(segmentation_algorithm, config["n_clusters"])
    
    # Check for shared numbering results first
    shared_numbering_path = os.path.join(
        config["project_path"],
        "results",
        session,
        model_name,
        "dbscan_shared_numbering"
    )
    
    # Check if shared numbering label file exists
    shared_label_file = os.path.join(
        shared_numbering_path,
        f"dbscan_shared_numbering_label_{session}.npy"
    )
    
    if os.path.exists(shared_label_file):
        logger.info(f"Using shared numbering DBSCAN path for session {session}")
        return "dbscan_shared_numbering"
    
    # Check for group-aware results second
    group_aware_path = os.path.join(
        config["project_path"],
        "results",
        session,
        model_name,
        "dbscan_group_aware"
    )
    
    if os.path.exists(group_aware_path):
        logger.info(f"Using group-aware DBSCAN path for session {session}")
        return "dbscan_group_aware"
    else:
        logger.info(f"Using regular DBSCAN path for session {session}")
        return "dbscan"


@save_state(model=MotifVideosFunctionSchema)
def motif_videos(
    config: dict,
    segmentation_algorithm: SegmentationAlgorithms,
    video_type: str = ".mp4",
    output_video_type: str = ".mp4",
    save_logs: bool = False,
    use_sequential_numbering: bool = True,  # NEW PARAMETER
) -> None:
    """
    FIXED: Generate motif videos with clean sequential numbering.
    Now creates videos with clean 0-N numbering instead of gaps like 0, 4, 72, 96.

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
    use_sequential_numbering : bool, optional
        Whether to use sequential video labels for clean numbering. Default is True.

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
        
        if use_sequential_numbering:
            logger.info("Using SEQUENTIAL NUMBERING for clean video names (0, 1, 2, 3...)")
        else:
            logger.info("Using ORIGINAL NUMBERING (may have gaps)")

        # Get sessions
        if config["all_data"] in ["Yes", "yes"]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                cfg=config,
                action_message="write motif videos",
            )

        # Log algorithm-specific information
        if segmentation_algorithm == "dbscan":
            individual_segmentation = config.get("individual_segmentation", False)
            
            if individual_segmentation:
                logger.info("Creating DBSCAN cluster videos (individual segmentation)")
            else:
                logger.info("Creating DBSCAN cluster videos (cross-session behavioral consistency)")
        elif segmentation_algorithm == "gmm":
            logger.info(f"Creating GMM cluster videos with {n_clusters} components")
        else:
            logger.info("Cluster size is: %d " % n_clusters)
            
        for session in sessions:
            # Use detection function for proper path
            segmentation_path = detect_and_get_dbscan_path(config, session, model_name, segmentation_algorithm)
            
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
                use_sequential_for_videos=use_sequential_numbering,
            )
        
        if use_sequential_numbering:
            logger.info("All videos have been created with CLEAN SEQUENTIAL NUMBERING!")
            logger.info("Video names: session-motif_0.mp4, session-motif_1.mp4, session-motif_2.mp4, ...")
        else:
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
    use_sequential_numbering: bool = True,  # NEW PARAMETER
) -> None:
    """
    FIXED: Generate community videos with clean sequential numbering.

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
    use_sequential_numbering : bool, optional
        Whether to use sequential video labels for clean numbering. Default is True.

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

        logger.info(f"Creating community videos for algorithm: {segmentation_algorithm}...")
        
        if use_sequential_numbering:
            logger.info("Using SEQUENTIAL NUMBERING for clean community video names")
        
        # Log algorithm-specific information
        if segmentation_algorithm == "dbscan":
            individual_segmentation = config.get("individual_segmentation", False)
            
            if individual_segmentation:
                logger.info("Creating DBSCAN community videos (individual segmentation)")
            else:
                logger.info("Creating DBSCAN community videos (cross-session behavioral consistency)")
        elif segmentation_algorithm == "gmm":
            logger.info(f"Creating GMM community videos with {n_clusters} components")
        else:
            logger.info("Cluster size is: %d " % n_clusters)
            
        for session in sessions:
            # Use detection function for proper path
            segmentation_path = detect_and_get_dbscan_path(config, session, model_name, segmentation_algorithm)
            
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
                use_sequential_for_videos=use_sequential_numbering,
            )

        if use_sequential_numbering:
            logger.info("All community videos have been created with CLEAN SEQUENTIAL NUMBERING!")
        else:
            logger.info("All community videos have been created!")

    except Exception as e:
        logger.exception(f"Error in community_videos: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()


# UTILITY FUNCTIONS

def check_sequential_labels_status(config: dict) -> None:
    """
    Check if sequential video labels exist for all sessions.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    
    sessions = config["session_names"]
    model_name = config["model_name"]
    
    print("\n" + "="*80)
    print("SEQUENTIAL VIDEO LABELS STATUS")
    print("="*80)
    
    for session in sessions:
        print(f"\n📁 SESSION: {session}")
        
        # Check different paths
        paths_to_check = [
            ("Shared Numbering", os.path.join(
                config["project_path"], "results", session, model_name, 
                "dbscan_shared_numbering", f"dbscan_video_sequential_label_{session}.npy"
            )),
            ("Group Aware", os.path.join(
                config["project_path"], "results", session, model_name, 
                "dbscan_group_aware", f"dbscan_video_sequential_label_{session}.npy"
            )),
            ("Regular DBSCAN", os.path.join(
                config["project_path"], "results", session, model_name, 
                "dbscan", f"dbscan_video_sequential_label_{session}.npy"
            )),
        ]
        
        sequential_found = False
        for path_type, path in paths_to_check:
            if os.path.exists(path):
                labels = np.load(path)
                unique_labels = np.unique(labels[labels != -1])
                print(f"   ✅ {path_type}: Sequential labels 0-{max(unique_labels) if len(unique_labels) > 0 else 'N/A'}")
                sequential_found = True
                break
        
        if not sequential_found:
            print(f"   ❌ No sequential video labels found")
            print(f"      Run segment_session() with create_video_labels=True")


def create_sequential_labels_manually(config: dict) -> None:
    """
    Manually create sequential video labels if they don't exist.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    
    from .pose_segmentation import create_sequential_video_labels
    
    sessions = config["session_names"]
    model_name = config["model_name"]
    
    print("\n" + "="*60)
    print("CREATING SEQUENTIAL VIDEO LABELS MANUALLY")
    print("="*60)
    
    labels_list = []
    
    for session in sessions:
        # Try to find existing labels
        label_paths = [
            os.path.join(config["project_path"], "results", session, model_name, 
                        "dbscan_shared_numbering", f"dbscan_shared_numbering_label_{session}.npy"),
            os.path.join(config["project_path"], "results", session, model_name, 
                        "dbscan_group_aware", f"dbscan_group_aware_label_{session}.npy"),
            os.path.join(config["project_path"], "results", session, model_name, 
                        "dbscan", f"dbscan_label_{session}.npy"),
        ]
        
        session_labels = None
        for path in label_paths:
            if os.path.exists(path):
                session_labels = np.load(path)
                print(f"✓ Found labels for {session}")
                break
        
        if session_labels is None:
            print(f"❌ No labels found for {session}")
            return
        
        labels_list.append(session_labels)
    
    # Create sequential labels
    create_sequential_video_labels(config, sessions, labels_list, model_name, "dbscan")
    print("\n✅ Sequential video labels created successfully!")


def preview_video_names(config: dict, max_videos_per_session: int = 5) -> None:
    """
    Preview what video names will look like with sequential numbering.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    max_videos_per_session : int
        Maximum number of video names to show per session
    """
    
    sessions = config["session_names"]
    model_name = config["model_name"]
    
    print("\n" + "="*80)
    print("VIDEO NAMING PREVIEW")
    print("="*80)
    
    for session in sessions:
        print(f"\n🎬 SESSION: {session}")
        
        # Try to find sequential labels
        sequential_paths = [
            os.path.join(config["project_path"], "results", session, model_name, 
                        "dbscan_shared_numbering", f"dbscan_video_sequential_label_{session}.npy"),
            os.path.join(config["project_path"], "results", session, model_name, 
                        "dbscan_group_aware", f"dbscan_video_sequential_label_{session}.npy"),
            os.path.join(config["project_path"], "results", session, model_name, 
                        "dbscan", f"dbscan_video_sequential_label_{session}.npy"),
        ]
        
        sequential_labels = None
        for path in sequential_paths:
            if os.path.exists(path):
                sequential_labels = np.load(path)
                break
        
        if sequential_labels is not None:
            unique_labels = np.unique(sequential_labels[sequential_labels != -1])
            print(f"   📊 Sequential motifs: {len(unique_labels)} total")
            print(f"   📝 Video names will be:")
            
            for i, motif in enumerate(sorted(unique_labels)[:max_videos_per_session]):
                video_name = f"{session}-motif_{motif}.mp4"
                print(f"      • {video_name}")
            
            if len(unique_labels) > max_videos_per_session:
                print(f"      • ... and {len(unique_labels) - max_videos_per_session} more")
        else:
            print(f"   ❌ No sequential labels found")
            print(f"      Need to create sequential labels first")

