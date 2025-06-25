import os
import scipy
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Literal

from vame.analysis.tree_hierarchy import (
    graph_to_tree,
    bag_nodes_by_cutline,
)
from vame.util.data_manipulation import consecutive
from vame.util.cli import get_sessions_from_user_input
from vame.visualization.community import draw_tree
from vame.schemas.states import save_state, CommunityFunctionSchema
from vame.schemas.project import SegmentationAlgorithms
from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


def get_adjacency_matrix(
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the adjacency matrix, transition matrix, and temporal matrix.

    Parameters
    ----------
    labels : np.ndarray
        Array of cluster labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing: adjacency matrix, transition matrix, and temporal matrix.
    """
    temp_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    adjacency_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    cntMat = np.zeros((n_clusters))
    steps = len(labels)

    for i in range(n_clusters):
        for k in range(steps - 1):
            idx = labels[k]
            if idx == i:
                idx2 = labels[k + 1]
                if idx == idx2:
                    continue
                else:
                    cntMat[idx2] = cntMat[idx2] + 1
        temp_matrix[i] = cntMat
        cntMat = np.zeros((n_clusters))

    for k in range(steps - 1):
        idx = labels[k]
        idx2 = labels[k + 1]
        if idx == idx2:
            continue
        adjacency_matrix[idx, idx2] = 1
        adjacency_matrix[idx2, idx] = 1

    transition_matrix = get_transition_matrix(temp_matrix)
    return adjacency_matrix, transition_matrix, temp_matrix


def get_transition_matrix(
    adjacency_matrix: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Compute the transition matrix from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix.
    threshold : float, optional
        Threshold for considering transitions. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        Transition matrix.
    """
    row_sum = adjacency_matrix.sum(axis=1)
    transition_matrix = adjacency_matrix / row_sum[:, np.newaxis]
    transition_matrix[transition_matrix <= threshold] = 0
    if np.any(np.isnan(transition_matrix)):
        transition_matrix = np.nan_to_num(transition_matrix)
    return transition_matrix


def fill_motifs_with_zero_counts(
    unique_motif_labels: np.ndarray,
    motif_counts: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Find motifs that never occur in the dataset, and fill the motif_counts array with zeros for those motifs.
    Example 1:
        - unique_motif_labels = [0, 1, 3, 4]
        - motif_counts = [10, 20, 30, 40],
        - n_clusters = 5
        - the function will return [10, 20, 0, 30, 40].
    Example 2:
        - unique_motif_labels = [0, 1, 3, 4]
        - motif_counts = [10, 20, 30, 40],
        - n_clusters = 6
        - the function will return [10, 20, 0, 30, 40, 0].

    Parameters
    ----------
    unique_motif_labels : np.ndarray
        Array of unique motif labels.
    motif_counts : np.ndarray
        Array of motif counts (in number of frames).
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        List of motif counts (in number of frame) with 0's for motifs that never happened.
    """
    cons = consecutive(unique_motif_labels)
    usage_list = list(motif_counts)
    if len(cons) != 1:  # if missing motif is in the middle of the list
        logger.info("Go")
        if 0 not in cons[0]:
            first_id = cons[0][0]
            for k in range(first_id):
                usage_list.insert(k, 0)

        for i in range(len(cons) - 1):
            a = cons[i + 1][0]
            b = cons[i][-1]
            d = (a - b) - 1
            for j in range(1, d + 1):
                index = cons[i][-1] + j
                usage_list.insert(index, 0)
        if len(usage_list) < n_clusters:
            usage_list.insert(n_clusters, 0)

    elif len(cons[0]) != n_clusters:  # if missing motif is at the front or end of list
        # diff = n_clusters - cons[0][-1]
        usage_list = list(motif_counts)
        if cons[0][0] != 0:  # missing motif at front of list
            usage_list.insert(0, 0)
        else:  # missing motif at end of list
            usage_list.insert(n_clusters - 1, 0)

    if len(usage_list) < n_clusters:  # if there's more than one motif missing
        for k in range(len(usage_list), n_clusters):
            usage_list.insert(k, 0)

    usage = np.array(usage_list)
    return usage


def augment_motif_timeseries(
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment motif time series by filling zero motifs.

    Parameters
    ----------
    labels : np.ndarray
        Original array of labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple with:
            - Array of labels augmented with motifs that never occurred, artificially inputed
            at the end of the original labels array
            - Indices of the motifs that never occurred.
    """
    augmented_labels = labels.copy()
    unique_motif_labels, motif_counts = np.unique(augmented_labels, return_counts=True)
    augmented_motif_counts = fill_motifs_with_zero_counts(
        unique_motif_labels=unique_motif_labels,
        motif_counts=motif_counts,
        n_clusters=n_clusters,
    )
    motifs_with_zero_counts = np.where(augmented_motif_counts == 0)[0]
    logger.info(f"Zero motifs: {motifs_with_zero_counts}")
    # TODO - this seems to be filling the labels array with random motifs that have zero counts
    # is this intended? and why?
    idx = -1
    for i in range(len(motifs_with_zero_counts)):
        for j in range(20):
            x = np.random.choice(motifs_with_zero_counts)
            augmented_labels[idx] = x
            idx -= 1
    return augmented_labels, motifs_with_zero_counts


def get_label_file_path(
    config: dict,
    session: str,
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
) -> str:
    """
    UPDATED: Get the path to the label file based on segmentation algorithm.
    Now supports group-aware DBSCAN results.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    session : str
        Session name
    model_name : str
        Model name
    n_clusters : int
        Number of clusters (ignored for DBSCAN)
    segmentation_algorithm : str
        Segmentation algorithm
        
    Returns
    -------
    str
        Path to the label file
    """
    if segmentation_algorithm == "dbscan":
        # UPDATED: Check for group-aware results first
        group_aware_path = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            "dbscan_group_aware",
            f"dbscan_group_aware_label_{session}.npy"
        )
        
        if os.path.exists(group_aware_path):
            logger.info(f"Using group-aware DBSCAN results for session {session}")
            return group_aware_path
        else:
            # Fall back to regular DBSCAN results
            path_to_dir = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                "dbscan",
                "",
            )
            file_path = os.path.join(path_to_dir, f"dbscan_label_{session}.npy")
            logger.info(f"Using regular DBSCAN results for session {session}")
            return file_path
    else:
        # HMM, KMeans, and GMM use n_clusters in path
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        file_path = os.path.join(
            path_to_dir,
            str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
        )
        return file_path


def get_motif_labels(
    config: dict,
    sessions: List[str],
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
) -> np.ndarray:
    """
    UPDATED: Get motif labels for given files.
    Now supports group-aware DBSCAN results and preserves all clusters.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    sessions : List[str]
        List of session names.
    model_name : str
        Model name.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Which segmentation algorithm to use. Options are 'hmm', 'kmeans', 'gmm', or 'dbscan'.

    Returns
    -------
    np.ndarray
        Array of community labels (integers).
    """
    # Check if group-aware results exist
    using_group_aware = False
    if segmentation_algorithm == "dbscan":
        # Check if any session has group-aware results
        for session in sessions:
            group_aware_path = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                "dbscan_group_aware",
                f"dbscan_group_aware_label_{session}.npy"
            )
            if os.path.exists(group_aware_path):
                using_group_aware = True
                logger.info("Detected group-aware DBSCAN results - using for community analysis")
                break
    
    community_label = []
    for session in sessions:
        file_path = get_label_file_path(config, session, model_name, n_clusters, segmentation_algorithm)
        
        if not os.path.exists(file_path):
            logger.error(f"Label file not found: {file_path}")
            raise FileNotFoundError(f"Label file not found: {file_path}")
            
        file_labels = np.load(file_path)
        
        # For DBSCAN, filter out noise (-1) labels
        if segmentation_algorithm == "dbscan":
            file_labels = file_labels[file_labels != -1]
        
        # Use full session data to preserve all clusters
        community_label.extend(file_labels)
    
    community_label = np.array(community_label)
    logger.info(f"Total frames used for community analysis: {len(community_label)}")
    
    if using_group_aware:
        logger.info("Community analysis using group-aware DBSCAN results - behavioral consistency preserved")
    
    return community_label


def remap_dbscan_labels(labels: np.ndarray) -> Tuple[np.ndarray, dict, int]:
    """
    Remap DBSCAN labels to sequential integers starting from 0.
    
    Parameters
    ----------
    labels : np.ndarray
        Original DBSCAN labels (can be non-sequential, e.g., [0, 48, 49, 50])
        
    Returns
    -------
    Tuple[np.ndarray, dict, int]
        - Remapped labels (sequential from 0)
        - Mapping dictionary {original_label: new_label}
        - Number of unique clusters
    """
    # Remove noise (-1) if present
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    
    # Check if labels are already sequential
    if np.array_equal(unique_labels, np.arange(len(unique_labels))):
        logger.info("DBSCAN labels are already sequential - no remapping needed")
        return labels, {i: i for i in unique_labels}, n_clusters
    
    # Create mapping from original to sequential
    label_mapping = {orig: new for new, orig in enumerate(sorted(unique_labels))}
    
    # Apply mapping
    remapped_labels = np.zeros_like(labels)
    for orig_label, new_label in label_mapping.items():
        remapped_labels[labels == orig_label] = new_label
    
    # Handle noise labels (keep as -1)
    remapped_labels[labels == -1] = -1
    
    logger.info(f"DBSCAN labels remapped: {dict(list(label_mapping.items())[:5])}{'...' if len(label_mapping) > 5 else ''}")
    
    return remapped_labels, label_mapping, n_clusters


def compute_transition_matrices(
    files: List[str],
    labels: List[np.ndarray],
    n_clusters: int,
) -> List[np.ndarray]:
    """
    Compute transition matrices for given files and labels.

    Parameters
    ----------
    files : List[str]
        List of file paths.
    labels : List[np.ndarray]
        List of label arrays.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    List[np.ndarray]:
        List of transition matrices.
    """
    transition_matrices = []
    for i, file in enumerate(files):
        adj, trans, mat = get_adjacency_matrix(labels[i], n_clusters)
        transition_matrices.append(trans)
    return transition_matrices


def create_cohort_community_bag(
    config: dict,
    motif_labels: List[np.ndarray],
    trans_mat_full: np.ndarray,
    cut_tree: int | None,
    n_clusters: int,
    segmentation_algorithm: Literal["hmm", "kmeans", "gmm", "dbscan"],
) -> list:
    """
    UPDATED: Create cohort community bag for given motif labels, transition matrix,
    cut tree, and number of clusters. Now supports group-aware DBSCAN.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    motif_labels : List[np.ndarray]
        List of motif label arrays.
    trans_mat_full : np.ndarray
        Full transition matrix.
    cut_tree : int | None
        Cut line for tree.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Which segmentation algorithm to use. Options are 'hmm', 'kmeans', 'gmm', or 'dbscan'.

    Returns
    -------
    List
        List of community bags.
    """
    communities_all = []
    unique_labels, usage_full = np.unique(motif_labels, return_counts=True)
    labels_usage = dict()
    for la, u in zip(unique_labels, usage_full):
        labels_usage[str(la)] = u / np.sum(usage_full)
    T = graph_to_tree(
        motif_usage=usage_full,
        transition_matrix=trans_mat_full,
        n_clusters=n_clusters,
        merge_sel=1,
    )
    
    # UPDATED: Create appropriate results directory
    if segmentation_algorithm == "dbscan":
        # Check if group-aware results exist
        group_aware_exists = any(
            os.path.exists(os.path.join(
                config["project_path"],
                "results",
                session,
                config["model_name"],
                "dbscan_group_aware"
            ))
            for session in config.get("session_names", [])
        )
        
        if group_aware_exists:
            results_dir = os.path.join(
                config["project_path"],
                "results",
                "community_cohort",
                "dbscan_group_aware",
            )
            logger.info("Using group-aware community analysis directory")
        else:
            results_dir = os.path.join(
                config["project_path"],
                "results",
                "community_cohort",
                "dbscan",
            )
    else:
        # Works for kmeans, hmm, and gmm
        results_dir = os.path.join(
            config["project_path"],
            "results",
            "community_cohort",
            segmentation_algorithm + "-" + str(n_clusters),
        )
    
    os.makedirs(results_dir, exist_ok=True)
    
    nx.write_graphml(T, os.path.join(results_dir, "tree.graphml"))
    draw_tree(
        T=T,
        fig_width=n_clusters,
        usage_dict=labels_usage,
        save_to_file=True,
        show_figure=False,
        results_dir=results_dir,
    )

    if cut_tree is not None:
        communities_all = bag_nodes_by_cutline(
            tree=T,
            cutline=cut_tree,
            root="Root",
        )
        logger.info("Communities bag:")
        for ci, comm in enumerate(communities_all):
            logger.info(f"Community {ci}: {comm}")
    else:
        plt.pause(0.5)
        flag_1 = "no"
        while flag_1 == "no":
            cutline = int(input("Where do you want to cut the Tree? 0/1/2/3/..."))
            community_bag = bag_nodes_by_cutline(
                tree=T,
                cutline=cutline,
                root="Root",
            )
            logger.info(community_bag)
            flag_2 = input("\nAre all motifs in the list? (yes/no/restart)")
            if flag_2 == "no":
                while flag_2 == "no":
                    add = input("Extend list or add in the end? (ext/end)")
                    if add == "ext":
                        motif_idx = int(input("Which motif number? "))
                        list_idx = int(input("At which position in the list? (pythonic indexing starts at 0) "))
                        community_bag[list_idx].append(motif_idx)
                    if add == "end":
                        motif_idx = int(input("Which motif number? "))
                        community_bag.append([motif_idx])
                        logger.info(community_bag)
                    flag_2 = input("\nAre all motifs in the list? (yes/no/restart)")
            if flag_2 == "restart":
                continue
            if flag_2 == "yes":
                communities_all = community_bag
                flag_1 = "yes"
    return communities_all


def get_cohort_community_labels(
    motif_labels: List[np.ndarray],
    cohort_community_bag: list,
    median_filter_size: int = 7,
) -> List[np.ndarray]:
    """
    Transform kmeans/hmm/gmm parameterized latent vector motifs into communities.
    Get cohort community labels for given labels, and community bags.

    Parameters
    ----------
    labels : List[np.ndarray]
        List of label arrays.
    cohort_community_bag : np.ndarray
        List of community bags. Dimensions: (n_communities, n_clusters_in_community)
    median_filter_size : int, optional
        Size of the median filter, in number of frames. Defaults to 7.

    Returns
    -------
    List[np.ndarray]
        List of cohort community labels for each file.
    """
    community_labels_all = []
    num_comm = len(cohort_community_bag)
    community_labels = np.zeros_like(motif_labels)
    for i in range(num_comm):
        clust = np.asarray(cohort_community_bag[i])
        for j in range(len(clust)):
            find_clust = np.where(motif_labels == clust[j])[0]
            community_labels[find_clust] = i
    community_labels = np.int64(scipy.signal.medfilt(community_labels, median_filter_size))
    community_labels_all.append(community_labels)
    return community_labels_all


def save_cohort_community_labels_per_file(
    config: dict,
    sessions: List[str],
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
    cohort_community_bag: list,
) -> None:
    """
    UPDATED: Save cohort community labels per file with proper DBSCAN label remapping.
    Now supports group-aware DBSCAN results.
    """
    for idx, session in enumerate(sessions):
        file_path = get_label_file_path(config, session, model_name, n_clusters, segmentation_algorithm)
        file_labels = np.load(file_path)
        
        # Check if using group-aware results
        using_group_aware = "dbscan_group_aware" in file_path
        
        # For DBSCAN, filter out noise (-1) labels before community analysis
        if segmentation_algorithm == "dbscan":
            original_labels = file_labels.copy()
            
            # For group-aware results, labels should already be properly mapped
            if using_group_aware:
                logger.info(f"DBSCAN Group-Aware Community: Using pre-mapped labels for session {session}")
                file_labels_for_community = file_labels[file_labels != -1]
            else:
                # Apply remapping for regular DBSCAN results
                non_noise_labels = file_labels[file_labels != -1]
                if len(non_noise_labels) > 0:
                    unique_labels_orig = np.unique(non_noise_labels)
                    # Check if remapping is needed (non-sequential labels)
                    if not np.array_equal(unique_labels_orig, np.arange(len(unique_labels_orig))):
                        label_mapping = {orig: new for new, orig in enumerate(sorted(unique_labels_orig))}
                        
                        # Apply remapping to non-noise labels
                        remapped_file_labels = np.copy(file_labels)
                        for orig_label, new_label in label_mapping.items():
                            remapped_file_labels[file_labels == orig_label] = new_label
                        
                        file_labels = remapped_file_labels
                        logger.info(f"DBSCAN Community: Remapped labels for session {session}")
                
                # Filter out noise after remapping
                file_labels_for_community = file_labels[file_labels != -1]
        else:
            file_labels_for_community = file_labels
            
        community_labels = get_cohort_community_labels(
            motif_labels=file_labels_for_community,
            cohort_community_bag=cohort_community_bag,
        )
        
        # UPDATED: Get the directory path for saving
        if segmentation_algorithm == "dbscan":
            if using_group_aware:
                path_to_dir = os.path.join(
                    config["project_path"],
                    "results",
                    session,
                    model_name,
                    "dbscan_group_aware",
                    "",
                )
            else:
                path_to_dir = os.path.join(
                    config["project_path"],
                    "results",
                    session,
                    model_name,
                    "dbscan",
                    "",
                )
        else:
            # Works for kmeans, hmm, and gmm
            path_to_dir = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                segmentation_algorithm + "-" + str(n_clusters),
                "",
            )
        
        if not os.path.exists(os.path.join(path_to_dir, "community")):
            os.mkdir(os.path.join(path_to_dir, "community"))
            
        # For DBSCAN, need to map back to original indices (including noise)
        if segmentation_algorithm == "dbscan":
            full_community_labels = np.full_like(original_labels, -1, dtype=np.int64)
            non_noise_mask = original_labels != -1
            full_community_labels[non_noise_mask] = community_labels[0]
            
            np.save(
                os.path.join(
                    path_to_dir,
                    "community",
                    f"cohort_community_label_{session}.npy",
                ),
                full_community_labels,
            )
        else:
            np.save(
                os.path.join(
                    path_to_dir,
                    "community",
                    f"cohort_community_label_{session}.npy",
                ),
                np.array(community_labels[0]),
            )


@save_state(model=CommunityFunctionSchema)
def community(
    config: dict,
    segmentation_algorithm: SegmentationAlgorithms,
    cohort: bool = True,
    cut_tree: int | None = None,
    save_logs: bool = False,
) -> None:
    """
    UPDATED: Perform community analysis.
    Now supports group-aware DBSCAN results for cross-group behavioral comparison.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm', 'kmeans', 'gmm', or 'dbscan'.
    cohort : bool, optional
        Flag indicating cohort analysis. Defaults to True.
    cut_tree : int, optional
        Cut line for tree. Defaults to None.
    save_logs : bool, optional
        Flag indicating whether to save logs. Defaults to False.

    Returns
    -------
    None
    """
    try:
        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "community.log"
            logger_config.add_file_handler(str(log_path))

        model_name = config["model_name"]
        n_clusters = config["n_clusters"]

        # Get sessions
        if config["all_data"] in ["Yes", "yes"]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                cfg=config,
                action_message="run community analysis",
            )

        # UPDATED: Check if group-aware results exist
        using_group_aware = False
        if segmentation_algorithm == "dbscan":
            for session in sessions:
                group_aware_path = os.path.join(
                    config["project_path"],
                    "results",
                    session,
                    model_name,
                    "dbscan_group_aware"
                )
                if os.path.exists(group_aware_path):
                    using_group_aware = True
                    logger.info("Group-aware DBSCAN results detected - community analysis will preserve behavioral consistency")
                    break

        # Run community analysis for cohort=True
        if cohort:
            # UPDATED: Create appropriate directory path
            if segmentation_algorithm == "dbscan":
                if using_group_aware:
                    path_to_dir = Path(
                        os.path.join(
                            config["project_path"],
                            "results",
                            "community_cohort",
                            "dbscan_group_aware",
                        )
                    )
                else:
                    path_to_dir = Path(
                        os.path.join(
                            config["project_path"],
                            "results",
                            "community_cohort",
                            "dbscan",
                        )
                    )
            else:
                # Works for kmeans, hmm, and gmm
                path_to_dir = Path(
                    os.path.join(
                        config["project_path"],
                        "results",
                        "community_cohort",
                        segmentation_algorithm + "-" + str(n_clusters),
                    )
                )

            if not path_to_dir.exists():
                path_to_dir.mkdir(parents=True, exist_ok=True)

            # Get motif labels
            motif_labels = get_motif_labels(
                config=config,
                sessions=sessions,
                model_name=model_name,
                n_clusters=n_clusters,
                segmentation_algorithm=segmentation_algorithm,
            )
            
            # For DBSCAN, we need to remap labels to be sequential
            if segmentation_algorithm == "dbscan":
                motif_labels, label_mapping, actual_n_clusters = remap_dbscan_labels(motif_labels)
                if using_group_aware:
                    logger.info(f"Group-aware DBSCAN: Using {len(label_mapping)} behaviorally-consistent clusters")
                else:
                    logger.info(f"Regular DBSCAN: Remapped {len(label_mapping)} clusters to sequential labels")
                logger.info(f"Original -> New mapping: {label_mapping}")
                effective_n_clusters = actual_n_clusters
            else:
                effective_n_clusters = n_clusters
            
            augmented_labels, motifs_with_zero_counts = augment_motif_timeseries(
                labels=motif_labels,
                n_clusters=effective_n_clusters,
            )
            _, trans_mat_full, _ = get_adjacency_matrix(
                labels=augmented_labels,
                n_clusters=effective_n_clusters,
            )
            cohort_community_bag = create_cohort_community_bag(
                config=config,
                motif_labels=motif_labels,
                trans_mat_full=trans_mat_full,
                cut_tree=cut_tree,
                n_clusters=effective_n_clusters,
                segmentation_algorithm=segmentation_algorithm,
            )
            community_labels_all = get_cohort_community_labels(
                motif_labels=motif_labels,
                cohort_community_bag=cohort_community_bag,
            )

            # convert cohort_community_bag to dtype object numpy array because it is an inhomogeneous list
            cohort_community_bag = np.array(cohort_community_bag, dtype=object)

            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_transition_matrix" + ".npy",
                ),
                trans_mat_full,
            )
            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_community_label" + ".npy",
                ),
                community_labels_all,
            )
            
            # UPDATED: Save algorithm-specific label file
            if segmentation_algorithm == "dbscan":
                if using_group_aware:
                    np.save(
                        os.path.join(
                            path_to_dir,
                            "cohort_dbscan_group_aware_label" + ".npy",
                        ),
                        motif_labels,
                    )
                else:
                    np.save(
                        os.path.join(
                            path_to_dir,
                            "cohort_dbscan_label" + ".npy",
                        ),
                        motif_labels,
                    )
            else:
                # Works for kmeans, hmm, and gmm
                np.save(
                    os.path.join(
                        path_to_dir,
                        "cohort_" + segmentation_algorithm + "_label" + ".npy",
                    ),
                    motif_labels,
                )
                
            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_community_bag" + ".npy",
                ),
                cohort_community_bag,
            )
            with open(os.path.join(path_to_dir, "hierarchy" + ".pkl"), "wb") as fp:  # Pickling
                pickle.dump(cohort_community_bag, fp)

            # UPDATED: Save DBSCAN label mapping if applicable
            if segmentation_algorithm == "dbscan":
                import json
                mapping_file_name = "dbscan_group_aware_label_mapping.json" if using_group_aware else "dbscan_label_mapping.json"
                mapping_file = os.path.join(path_to_dir, mapping_file_name)
                with open(mapping_file, 'w') as f:
                    # Convert numpy int keys to regular int for JSON serialization
                    json_mapping = {int(k): int(v) for k, v in label_mapping.items()}
                    json.dump({
                        "original_to_sequential": json_mapping,
                        "sequential_to_original": {int(v): int(k) for k, v in label_mapping.items()},
                        "n_clusters": actual_n_clusters,
                        "group_aware": using_group_aware,
                        "behavioral_consistency": "preserved" if using_group_aware else "standard"
                    }, f, indent=2)
                logger.info(f"DBSCAN label mapping saved to: {mapping_file}")

            # Save the full community labels list to each of the original video files
            # This is useful for further analysis when cohort=True
            save_cohort_community_labels_per_file(
                config=config,
                sessions=sessions,
                model_name=model_name,
                n_clusters=n_clusters,
                segmentation_algorithm=segmentation_algorithm,
                cohort_community_bag=cohort_community_bag,
            )
            
            # ADDED: Log completion message with group-aware status
            if using_group_aware:
                logger.info("Community analysis completed using group-aware DBSCAN results!")
                logger.info("Communities preserve cross-group behavioral consistency.")
                logger.info("Community motifs have the same behavioral meaning across groups.")
            else:
                logger.info("Community analysis completed using standard segmentation results.")

        # Work in Progress - cohort is False
        else:
            raise NotImplementedError("Community analysis for cohort=False is not supported yet.")

    except Exception as e:
        logger.exception(f"Error in community_analysis: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()


# ADDED: Utility function to check group-aware status
def check_group_aware_status(config: dict, segmentation_algorithm: str = "dbscan") -> dict:
    """
    Check if group-aware DBSCAN results exist and provide status information.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    segmentation_algorithm : str
        Segmentation algorithm to check
        
    Returns
    -------
    dict
        Status information about group-aware results
    """
    if segmentation_algorithm != "dbscan":
        return {
            "group_aware_available": False,
            "reason": f"{segmentation_algorithm} does not support group-aware analysis",
            "sessions_with_group_aware": [],
            "sessions_with_regular": []
        }
    
    sessions = config.get("session_names", [])
    model_name = config.get("model_name", "")
    project_path = config.get("project_path", "")
    
    sessions_with_group_aware = []
    sessions_with_regular = []
    
    for session in sessions:
        # Check for group-aware results
        group_aware_path = os.path.join(
            project_path,
            "results",
            session,
            model_name,
            "dbscan_group_aware",
            f"dbscan_group_aware_label_{session}.npy"
        )
        
        # Check for regular results
        regular_path = os.path.join(
            project_path,
            "results",
            session,
            model_name,
            "dbscan",
            f"dbscan_label_{session}.npy"
        )
        
        if os.path.exists(group_aware_path):
            sessions_with_group_aware.append(session)
        elif os.path.exists(regular_path):
            sessions_with_regular.append(session)
    
    group_aware_available = len(sessions_with_group_aware) > 0
    
    status = {
        "group_aware_available": group_aware_available,
        "sessions_with_group_aware": sessions_with_group_aware,
        "sessions_with_regular": sessions_with_regular,
        "total_sessions": len(sessions),
        "coverage": len(sessions_with_group_aware) / len(sessions) if sessions else 0
    }
    
    if group_aware_available:
        status["reason"] = f"Group-aware results found for {len(sessions_with_group_aware)}/{len(sessions)} sessions"
        
        # Check if behavioral mapping exists
        mapping_path = os.path.join(
            project_path,
            "results",
            "group_comparison",
            "dbscan_group_aware",
            "behavioral_mapping.json"
        )
        status["behavioral_mapping_available"] = os.path.exists(mapping_path)
        
    else:
        status["reason"] = "No group-aware DBSCAN results found"
    
    return status


# ADDED: Utility function to print group-aware status
def print_group_aware_status(config: dict, segmentation_algorithm: str = "dbscan") -> None:
    """
    Print a formatted status report about group-aware results.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    segmentation_algorithm : str
        Segmentation algorithm to check
    """
    status = check_group_aware_status(config, segmentation_algorithm)
    
    print("\n" + "="*60)
    print("GROUP-AWARE SEGMENTATION STATUS")
    print("="*60)
    
    print(f"Algorithm: {segmentation_algorithm.upper()}")
    print(f"Group-aware available: {'Yes' if status['group_aware_available'] else 'No'}")
    print(f"Reason: {status['reason']}")
    
    if status['group_aware_available']:
        print(f"Coverage: {status['coverage']:.1%} ({len(status['sessions_with_group_aware'])}/{status['total_sessions']} sessions)")
        
        if status.get('behavioral_mapping_available', False):
            print("Behavioral mapping: Available")
        else:
            print("Behavioral mapping: Not found")
        
        print("\nSessions with group-aware results:")
        for session in status['sessions_with_group_aware']:
            print(f"  ✓ {session}")
        
        if status['sessions_with_regular']:
            print("\nSessions with regular results only:")
            for session in status['sessions_with_regular']:
                print(f"  • {session}")
    
    print("\nRecommendations:")
    if status['group_aware_available']:
        if status['coverage'] == 1.0:
            print("  ✓ All sessions have group-aware results - ready for cross-group analysis")
        else:
            print("  ⚠ Partial coverage - consider running group-aware segmentation for all sessions")
    else:
        print("  → Run segment_session() with group_labels parameter to enable group-aware analysis")
        print("  → This will ensure behavioral consistency across groups")
    
    print("="*60)