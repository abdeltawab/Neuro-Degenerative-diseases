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
    Get the path to the label file based on segmentation algorithm.
    
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
        # DBSCAN uses different path structure
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            "dbscan",
            "",
        )
        file_path = os.path.join(path_to_dir, f"dbscan_label_{session}.npy")
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
    Get motif labels for given files.

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
    # TODO  - this is limiting the number of frames to the minimum number of frames in all files
    # Is this intended behavior? and why?
    shapes = []
    for session in sessions:
        file_path = get_label_file_path(config, session, model_name, n_clusters, segmentation_algorithm)
        file_labels = np.load(file_path)
        
        # For DBSCAN, filter out noise (-1) labels
        if segmentation_algorithm == "dbscan":
            file_labels = file_labels[file_labels != -1]
        
        shape = len(file_labels)
        shapes.append(shape)
    shapes = np.array(shapes)
    min_frames = min(shapes)

    community_label = []
    for session in sessions:
        file_path = get_label_file_path(config, session, model_name, n_clusters, segmentation_algorithm)
        file_labels = np.load(file_path)
        
        # For DBSCAN, filter out noise (-1) labels
        if segmentation_algorithm == "dbscan":
            file_labels = file_labels[file_labels != -1]
        
        file_labels = file_labels[:min_frames]
        community_label.extend(file_labels)
    community_label = np.array(community_label)
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
    
    # Create mapping from original to sequential
    label_mapping = {orig: new for new, orig in enumerate(sorted(unique_labels))}
    
    # Apply mapping
    remapped_labels = np.zeros_like(labels)
    for orig_label, new_label in label_mapping.items():
        remapped_labels[labels == orig_label] = new_label
    
    # Handle noise labels (keep as -1)
    remapped_labels[labels == -1] = -1
    
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
    segmentation_algorithm: Literal["hmm", "kmeans", "gmm", "dbscan"],  # Added GMM
) -> list:
    """
    Create cohort community bag for given motif labels, transition matrix,
    cut tree, and number of clusters. (markov chain to tree -> community detection)

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
    
    # Create appropriate results directory
    if segmentation_algorithm == "dbscan":
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
    # nx.write_gpickle(T, 'T.gpickle')

    if cut_tree is not None:
        # communities_all = traverse_tree_cutline(T, cutline=cut_tree)
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
            # community_bag = traverse_tree_cutline(T, cutline=cutline)
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
    for idx, session in enumerate(sessions):
        file_path = get_label_file_path(config, session, model_name, n_clusters, segmentation_algorithm)
        file_labels = np.load(file_path)
        
        # For DBSCAN, filter out noise (-1) labels before community analysis
        if segmentation_algorithm == "dbscan":
            original_labels = file_labels.copy()
            file_labels = file_labels[file_labels != -1]
            
        community_labels = get_cohort_community_labels(
            motif_labels=file_labels,
            cohort_community_bag=cohort_community_bag,
        )
        
        # Get the directory path for saving
        if segmentation_algorithm == "dbscan":
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
    Perform community analysis.
    Fills in the values in the "community" key of the states.json file.
    Saves results files at:

    1. If cohort is True:
    - project_name/
        - results/
            - community_cohort/
                - segmentation_algorithm-n_clusters/ (or 'dbscan' for DBSCAN)
                    - cohort_community_bag.npy
                    - cohort_community_label.npy
                    - cohort_segmentation_algorithm_label.npy (or cohort_dbscan_label.npy)
                    - cohort_transition_matrix.npy
                    - hierarchy.pkl
            - file_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/ (or 'dbscan' for DBSCAN)
                        - community/
                            - cohort_community_label_file_name.npy

    2. If cohort is False:
    - project_name/
        - results/
            - file_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/ (or 'dbscan' for DBSCAN)
                        - community/
                            - transition_matrix_file_name.npy
                            - community_label_file_name.npy
                            - hierarchy_file_name.pkl

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

        # Run community analysis for cohort=True
        if cohort:
            # Create appropriate directory path
            if segmentation_algorithm == "dbscan":
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
                logger.info(f"DBSCAN: Remapped {len(label_mapping)} clusters to sequential labels")
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
            
            # Save algorithm-specific label file
            if segmentation_algorithm == "dbscan":
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

            # Save DBSCAN label mapping if applicable
            if segmentation_algorithm == "dbscan":
                import json
                mapping_file = os.path.join(path_to_dir, "dbscan_label_mapping.json")
                with open(mapping_file, 'w') as f:
                    # Convert numpy int keys to regular int for JSON serialization
                    json_mapping = {int(k): int(v) for k, v in label_mapping.items()}
                    json.dump({
                        "original_to_sequential": json_mapping,
                        "sequential_to_original": {int(v): int(k) for k, v in label_mapping.items()},
                        "n_clusters": actual_n_clusters
                    }, f, indent=2)
                logger.info(f"DBSCAN label mapping saved to: {mapping_file}")

            # Added by Luiz - 11/10/2024
            # Saves the full community labels list to each of the original video files
            # This is useful for further analysis when cohort=True
            save_cohort_community_labels_per_file(
                config=config,
                sessions=sessions,
                model_name=model_name,
                n_clusters=n_clusters,
                segmentation_algorithm=segmentation_algorithm,
                cohort_community_bag=cohort_community_bag,
            )

        # # Work in Progress - cohort is False
        else:
            raise NotImplementedError("Community analysis for cohort=False is not supported yet.")

    except Exception as e:
        logger.exception(f"Error in community_analysis: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()