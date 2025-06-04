import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture  # Add GMM import
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from vame.schemas.states import save_state, SegmentSessionFunctionSchema
from vame.logging.logger import VameLogger, TqdmToLogger
from vame.model.rnn_model import RNN_VAE
from vame.io.load_poses import read_pose_estimation_file
from vame.util.cli import get_sessions_from_user_input
from vame.util.model_util import load_model
from vame.preprocessing.to_model import format_xarray_for_rnn


logger_config = VameLogger(__name__)
logger = logger_config.logger


def embedd_latent_vectors(
    cfg: dict,
    sessions: List[str],
    model: RNN_VAE,
    fixed: bool,
    read_from_variable: str = "position_egocentric_aligned",  # Correct default value
    tqdm_stream: Union[TqdmToLogger, None] = None,
) -> List[np.ndarray]:
    logger.info("Entered embedd_latent_vectors function")  # Debug log
    logger.info(f"Using variable: {read_from_variable}")

    project_path = cfg["project_path"]
    temp_win = cfg["time_window"]
    num_features = cfg["num_features"]
    if not fixed:
        num_features = num_features - 3

    use_gpu = torch.cuda.is_available()

    latent_vector_files = []

    for session in sessions:
        logger.info(f"Embedding of latent vector for file {session}")
        
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)
        
        # Extract data using the provided variable
        if read_from_variable not in ds.variables:
            logger.error(f"Variable '{read_from_variable}' not found in dataset. Available variables: {list(ds.variables.keys())}")
            raise KeyError(f"Variable '{read_from_variable}' not found in dataset.")

        data = np.copy(ds[read_from_variable].values)  # This is the correct placement
        logger.info(f"Data shape for variable '{read_from_variable}': {data.shape}")

        # Format the data for the RNN model
        data = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
        )

        latent_vector_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(data.shape[1] - temp_win), file=tqdm_stream):
                data_sample_np = data[:, i : temp_win + i].T
                data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features))
                if use_gpu:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type("torch.FloatTensor").cuda())
                else:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type("torch.FloatTensor"))
                mu, _, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())

        latent_vector = np.concatenate(latent_vector_list, axis=0)
        latent_vector_files.append(latent_vector)

    return latent_vector_files


def estimate_dbscan_eps(data: np.ndarray, k: int = 4) -> float:
    """
    Estimate optimal eps parameter for DBSCAN using k-distance graph method.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for clustering
    k : int
        Number of nearest neighbors to consider (default: 4)
        
    Returns
    -------
    float
        Estimated eps value
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    # Sort distances to k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1], axis=0)
    
    # Use 75th percentile as a balanced approach
    eps_estimate = np.percentile(k_distances, 75)
    
    return eps_estimate


def tune_dbscan_parameters(data: np.ndarray, cfg: dict) -> Tuple[float, int]:
    """
    Automatically tune DBSCAN parameters for the given data.
    Uses iterative parameter testing to find optimal balance between 
    number of clusters and meaningful cluster sizes.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for clustering
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    Tuple[float, int]
        Tuned (eps, min_samples) parameters
    """
    # Get user-specified parameters or use defaults
    eps_val = cfg.get("dbscan_eps", None)
    min_samples_val = cfg.get("dbscan_min_samples", None)
    
    # If both parameters are specified, use them
    if eps_val is not None and min_samples_val is not None:
        logger.info(f"Using user-specified DBSCAN parameters: eps={eps_val}, min_samples={min_samples_val}")
        return eps_val, min_samples_val
    
    # Auto-tune parameters with iterative approach
    n_samples, n_features = data.shape
    
    # If only one parameter is specified, estimate the other
    if eps_val is None and min_samples_val is not None:
        eps_val = estimate_dbscan_eps(data) * 0.6  # More aggressive
        logger.info(f"Auto-estimated eps for given min_samples: {eps_val:.4f}")
        return eps_val, min_samples_val
    
    if eps_val is not None and min_samples_val is None:
        min_samples_val = max(5, min(12, int(0.002 * n_samples)))
        logger.info(f"Auto-estimated min_samples for given eps: {min_samples_val}")
        return eps_val, min_samples_val
    
    # Full auto-tuning: test multiple parameter combinations
    base_eps = estimate_dbscan_eps(data)
    logger.info(f"Base eps estimate: {base_eps:.4f}")
    
    # Define parameter ranges to test - more aggressive for more clusters
    eps_candidates = [
        base_eps * 0.3,   # Very aggressive
        base_eps * 0.4,   # Aggressive  
        base_eps * 0.5,   # Moderate-aggressive
        base_eps * 0.6,   # Moderate
        base_eps * 0.7,   # Conservative
    ]
    
    min_samples_candidates = [
        max(3, int(0.0005 * n_samples)),   # Very permissive
        max(5, int(0.001 * n_samples)),    # Permissive
        max(8, int(0.0015 * n_samples)),   # Moderate
        max(10, int(0.002 * n_samples)),   # Conservative
    ]
    
    # Cap min_samples at reasonable values
    min_samples_candidates = [min(ms, 20) for ms in min_samples_candidates]
    
    logger.info(f"Testing eps candidates: {[f'{e:.3f}' for e in eps_candidates]}")
    logger.info(f"Testing min_samples candidates: {min_samples_candidates}")
    
    best_eps = eps_candidates[2]  # Default to moderate-aggressive
    best_min_samples = min_samples_candidates[1]  # Default to permissive
    best_score = 0
    
    logger.info("Testing DBSCAN parameter combinations...")
    
    # Test combinations and score them
    for i, eps in enumerate(eps_candidates):
        for j, min_samp in enumerate(min_samples_candidates):
            # Quick test with a sample of data for speed
            sample_size = min(3000, n_samples)  # Smaller sample for speed
            np.random.seed(42)  # Consistent sampling
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_data = data[sample_indices]
            
            try:
                dbscan_test = DBSCAN(eps=eps, min_samples=min_samp)
                test_labels = dbscan_test.fit_predict(sample_data)
                
                # Calculate metrics
                unique_labels = np.unique(test_labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                n_noise = np.sum(test_labels == -1)
                noise_ratio = n_noise / len(test_labels) if len(test_labels) > 0 else 1.0
                
                # More aggressive scoring - heavily favor more clusters
                if n_clusters >= 2:  # Must have at least 2 clusters
                    # Base score from number of clusters
                    cluster_score = n_clusters * 2.0
                    
                    # Noise penalty (less harsh than before)
                    noise_penalty = 0
                    if noise_ratio > 0.7:  # Only penalize if >70% noise
                        noise_penalty = (noise_ratio - 0.7) * 3
                    
                    # Bonus for reasonable noise levels (10-50%)
                    if 0.1 <= noise_ratio <= 0.5:
                        cluster_score *= 1.2
                    
                    score = cluster_score - noise_penalty
                else:
                    score = 0  # No score for <2 clusters
                
                logger.info(f"  eps={eps:.3f}, min_samples={min_samp}: "
                           f"{n_clusters} clusters, {noise_ratio:.1%} noise, score={score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samp
                    logger.info(f"    *** New best combination! ***")
                    
            except Exception as e:
                logger.info(f"  eps={eps:.3f}, min_samples={min_samp}: Failed - {str(e)}")
    
    # Final safety check - if we didn't find good parameters, use aggressive defaults
    if best_score == 0:
        logger.warning("No good parameter combination found, using aggressive defaults")
        best_eps = base_eps * 0.4
        best_min_samples = max(5, min(10, int(0.001 * n_samples)))
    
    logger.info(f"Final selected parameters: eps={best_eps:.4f}, min_samples={best_min_samples}")
    logger.info(f"Best score achieved: {best_score:.2f}")
    
    return best_eps, best_min_samples


def get_motif_usage(
    session_labels: np.ndarray,
    n_clusters: int = None,
) -> np.ndarray:
    """
    Count motif usage from session label array.

    Parameters
    ----------
    session_labels : np.ndarray
        Array of session labels.
    n_clusters : int, optional
        Number of clusters. For KMeans, GMM, and HMM, this should be set to get fixed-length output.
        For DBSCAN, leave as None to infer cluster count dynamically (excluding noise -1).

    Returns
    -------
    np.ndarray
        Motif usage counts. Length = n_clusters for fixed methods, or dynamic for DBSCAN.
    """
    unique_labels = np.unique(session_labels)

    # Handle DBSCAN case: exclude noise (-1) and use dynamic output
    if n_clusters is None:
        if -1 in unique_labels:
            logger.info("DBSCAN: Noise label (-1) detected. Ignoring in motif usage count.")
            unique_labels = unique_labels[unique_labels != -1]

        motif_usage = np.zeros(len(unique_labels), dtype=int)
        for i, cluster in enumerate(sorted(unique_labels)):
            motif_usage[i] = np.sum(session_labels == cluster)

        return motif_usage

    # Fixed-length output for KMeans, GMM, or HMM
    motif_usage = np.zeros(n_clusters, dtype=int)
    for label in unique_labels:
        if label >= 0 and label < n_clusters:
            motif_usage[label] = np.sum(session_labels == label)

    # Warn about any unused motifs
    unused = np.setdiff1d(np.arange(n_clusters), unique_labels)
    if unused.size > 0:
        logger.info(f"Warning: The following motifs are unused: {unused}")

    return motif_usage


def same_segmentation(
    cfg: dict,
    sessions: List[str],
    latent_vectors: List[np.ndarray],
    n_clusters: int,
    segmentation_algorithm: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Apply the same segmentation (shared clustering) to all sessions using the specified algorithm.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    latent_vectors : List[np.ndarray]
        List of latent vector arrays per session.
    n_clusters : int
        Number of clusters (only used for KMeans, GMM, and HMM).
    segmentation_algorithm : str
        One of: "kmeans", "gmm", "hmm", or "dbscan".

    Returns
    -------
    Tuple of:
        - labels: List of np.ndarray of predicted motif labels per session.
        - cluster_centers: List of cluster centers (KMeans and GMM only).
        - motif_usages: List of motif usage arrays per session.
    """
    labels = []
    cluster_centers = []
    motif_usages = []

    # Concatenate latent vectors from all sessions
    latent_vector_cat = np.concatenate(latent_vectors, axis=0)

    if segmentation_algorithm == "kmeans":
        logger.info(f"Using KMeans with {n_clusters} clusters.")
        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            random_state=cfg.get("random_state_kmeans", 42),
            n_init=cfg.get("n_init_kmeans", 20),
        ).fit(latent_vector_cat)
        combined_labels = kmeans.labels_
        clust_center = kmeans.cluster_centers_

    elif segmentation_algorithm == "gmm":
        logger.info(f"Using Gaussian Mixture Model with {n_clusters} components.")
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=cfg.get("gmm_covariance_type", "full"),
            max_iter=cfg.get("gmm_max_iter", 100),
            n_init=cfg.get("gmm_n_init", 1),
            init_params=cfg.get("gmm_init_params", "kmeans"),
            random_state=cfg.get("gmm_random_state", 42),
        ).fit(latent_vector_cat)
        combined_labels = gmm.predict(latent_vector_cat)
        clust_center = gmm.means_

    elif segmentation_algorithm == "hmm":
        logger.info(f"Using HMM with {n_clusters} states.")
        hmm_model = hmm.GaussianHMM(
            n_components=n_clusters,
            covariance_type="full",
            n_iter=100,
        )
        hmm_model.fit(latent_vector_cat)
        combined_labels = hmm_model.predict(latent_vector_cat)
        clust_center = None  # HMM doesn't use cluster centers

    elif segmentation_algorithm == "dbscan":
        # Normalize data for DBSCAN
        scaler = StandardScaler()
        latent_vector_scaled = scaler.fit_transform(latent_vector_cat)
        
        # Tune DBSCAN parameters
        eps_val, min_samples_val = tune_dbscan_parameters(latent_vector_scaled, cfg)
        
        logger.info(f"Using DBSCAN (eps={eps_val:.4f}, min_samples={min_samples_val})")
        dbscan_model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        combined_labels = dbscan_model.fit_predict(latent_vector_scaled)
        clust_center = None  # DBSCAN does not produce cluster centers

        # Check clustering results
        unique_labels = np.unique(combined_labels)
        n_clusters_found = len(unique_labels[unique_labels != -1])  # Exclude noise
        n_noise = np.sum(combined_labels == -1)
        
        logger.info(f"DBSCAN results: {n_clusters_found} clusters found, {n_noise} noise points "
                   f"({100*n_noise/len(combined_labels):.1f}% of data)")
        
        if n_clusters_found == 0:
            logger.warning("DBSCAN found no clusters! All points classified as noise. "
                          "Consider adjusting eps and min_samples parameters.")
        elif n_noise > 0.5 * len(combined_labels):
            logger.warning(f"DBSCAN classified {100*n_noise/len(combined_labels):.1f}% of points as noise. "
                          "Consider adjusting parameters if this seems too high.")

    else:
        raise ValueError(f"Unknown segmentation algorithm: {segmentation_algorithm}")

    # Distribute combined labels back to each session
    idx = 0
    for i, session in enumerate(sessions):
        session_len = latent_vectors[i].shape[0]
        session_labels = combined_labels[idx: idx + session_len]
        labels.append(session_labels)

        if segmentation_algorithm in ["kmeans", "gmm"]:
            cluster_centers.append(clust_center)
        else:
            cluster_centers.append(None)

        # Motif usage: fixed length for kmeans/gmm/hmm, dynamic for dbscan
        usage = get_motif_usage(
            session_labels,
            None if segmentation_algorithm == "dbscan" else n_clusters
        )
        motif_usages.append(usage)

        idx += session_len

    return labels, cluster_centers, motif_usages


def individual_segmentation(
    cfg: dict,
    sessions: List[str],
    latent_vectors: List[np.ndarray],
    n_clusters: int,
    segmentation_algorithm: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    labels = []
    cluster_centers = []
    motif_usages = []
    
    for i, session in enumerate(sessions):
        data = latent_vectors[i]
        logger.info(f"Processing session '{session}' with {segmentation_algorithm.upper()} segmentation")
        
        if segmentation_algorithm == "kmeans":
            # KMeans clustering on this session's latent vectors
            kmeans = KMeans(
                init="k-means++",
                n_clusters=n_clusters,
                random_state=cfg.get("random_state_kmeans", 42),
                n_init=cfg.get("n_init_kmeans", 20),
            ).fit(data)
            session_labels = kmeans.labels_
            cluster_centers.append(kmeans.cluster_centers_)
            
        elif segmentation_algorithm == "gmm":
            # Train a separate GMM on this session's latent vectors
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type=cfg.get("gmm_covariance_type", "full"),
                max_iter=cfg.get("gmm_max_iter", 100),
                n_init=cfg.get("gmm_n_init", 1),
                init_params=cfg.get("gmm_init_params", "kmeans"),
                random_state=cfg.get("gmm_random_state", 42),
            ).fit(data)
            session_labels = gmm.predict(data)
            cluster_centers.append(gmm.means_)  # Store component means
            
        elif segmentation_algorithm == "hmm":
            # Train a separate HMM on this session's latent vectors
            hmm_model = hmm.GaussianHMM(
                n_components=n_clusters,
                covariance_type="full",
                n_iter=100,
            )
            hmm_model.fit(data)
            session_labels = hmm_model.predict(data)
            cluster_centers.append(None)  # No cluster centers for HMM
            
        elif segmentation_algorithm == "dbscan":
            # Normalize data for DBSCAN
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Tune DBSCAN parameters for this session
            eps_val, min_samples_val = tune_dbscan_parameters(data_scaled, cfg)
            
            dbscan_model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
            session_labels = dbscan_model.fit_predict(data_scaled)
            cluster_centers.append(None)  # No cluster centers for DBSCAN
            
            # Log results for this session
            unique_labels = np.unique(session_labels)
            n_clusters_found = len(unique_labels[unique_labels != -1])
            n_noise = np.sum(session_labels == -1)
            
            logger.info(f"Session '{session}' DBSCAN results: {n_clusters_found} clusters, "
                       f"{n_noise} noise points ({100*n_noise/len(session_labels):.1f}%)")
            
        else:
            raise ValueError(f"Unknown segmentation algorithm: {segmentation_algorithm}")

        labels.append(session_labels)
        
        # Compute motif usage (fixed length for kmeans/gmm/hmm, dynamic for DBSCAN)
        motif_usage = get_motif_usage(
            session_labels, 
            None if segmentation_algorithm == "dbscan" else n_clusters
        )
        motif_usages.append(motif_usage)
        
    return labels, cluster_centers, motif_usages


@save_state(model=SegmentSessionFunctionSchema)
def segment_session(config: dict, save_logs: bool = False) -> None:
    project_path = Path(config["project_path"]).resolve()
    try:
        tqdm_stream = None
        if save_logs:
            log_path = project_path / "logs" / "pose_segmentation.log"
            logger_config.add_file_handler(str(log_path))
            tqdm_stream = TqdmToLogger(logger)

        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        fixed = config["egocentric_data"]
        segmentation_algorithms = config["segmentation_algorithms"]
        ind_seg = config["individual_segmentation"]

        logger.info("Pose segmentation for VAME model: %s \n", model_name)
        logger.info(f"Segmentation algorithms: {segmentation_algorithms}")

        for seg in segmentation_algorithms:
            segmentation_path = seg + ("-" + str(n_clusters) if seg != "dbscan" else "")
            logger.info(f"Running pose segmentation using {seg} algorithm...")

            # Ensure per-session directory exists
            for session in config["session_names"]:
                os.makedirs(
                    os.path.join(project_path, "results", session, model_name),
                    exist_ok=True,
                )

            sessions = (
                config["session_names"]
                if config["all_data"].lower() == "yes"
                else get_sessions_from_user_input(cfg=config, action_message="run segmentation")
            )

            if torch.cuda.is_available():
                logger.info("Using CUDA on: %s", torch.cuda.get_device_name(0))
            else:
                logger.info("CUDA is not available. Using CPU.")
                torch.device("cpu")

            result_dir = os.path.join(project_path, "results", sessions[0], model_name, segmentation_path)

            if not os.path.exists(result_dir):
                new = True
                model = load_model(config, model_name, fixed)
                latent_vectors = embedd_latent_vectors(
                    cfg=config,
                    sessions=sessions,
                    model=model,
                    fixed=fixed,
                    read_from_variable=config["read_from_variable"],
                    tqdm_stream=tqdm_stream,
                )

                if ind_seg:
                    if seg == "dbscan":
                        logger.info("Apply individual segmentation for each session using DBSCAN")
                    elif seg == "hmm":
                        logger.info(f"Apply individual segmentation for each session with HMM ({n_clusters} states)")
                    elif seg == "gmm":
                        logger.info(f"Apply individual segmentation for each session with GMM ({n_clusters} components)")
                    else:
                        logger.info(f"Apply individual segmentation for each session with {n_clusters} clusters")

                    labels, cluster_center, motif_usages = individual_segmentation(
                        cfg=config,
                        sessions=sessions,
                        latent_vectors=latent_vectors,
                        n_clusters=n_clusters,
                        segmentation_algorithm=seg,
                    )
                else:
                    if seg == "dbscan":
                        logger.info("Apply the same segmentation for all sessions using DBSCAN")
                    elif seg == "hmm":
                        logger.info(f"Apply the same segmentation for all sessions with HMM ({n_clusters} states)")
                    elif seg == "gmm":
                        logger.info(f"Apply the same segmentation for all sessions with GMM ({n_clusters} components)")
                    else:
                        logger.info(f"Apply the same segmentation for all sessions with {n_clusters} clusters")

                    labels, cluster_center, motif_usages = same_segmentation(
                        cfg=config,
                        sessions=sessions,
                        latent_vectors=latent_vectors,
                        n_clusters=n_clusters,
                        segmentation_algorithm=seg,
                    )
            else:
                if seg == "dbscan":
                    logger.info(f"Segmentation with DBSCAN already exists for model {model_name}")
                elif seg == "hmm":
                    logger.info(f"Segmentation with HMM ({n_clusters} states) already exists for model {model_name}")
                elif seg == "gmm":
                    logger.info(f"Segmentation with GMM ({n_clusters} components) already exists for model {model_name}")
                else:
                    logger.info(f"Segmentation with {n_clusters} k-means clusters already exists for model {model_name}")

                flag = "yes"
                if os.path.exists(result_dir):
                    flag = input(
                        "WARNING: A segmentation for the chosen model and cluster size already exists!\n"
                        "Do you want to continue? A new segmentation will be computed! (yes/no) "
                    )

                if flag.lower() == "yes":
                    new = True
                    latent_vectors = []
                    for session in sessions:
                        path = os.path.join(
                            project_path, "results", session, model_name, segmentation_path
                        )
                        latent_vectors.append(np.load(os.path.join(path, f"latent_vector_{session}.npy")))

                    if ind_seg:
                        if seg == "dbscan":
                            logger.info("Apply individual segmentation for each session using DBSCAN")
                        elif seg == "hmm":
                            logger.info(f"Apply individual segmentation for each session with HMM ({n_clusters} states)")
                        elif seg == "gmm":
                            logger.info(f"Apply individual segmentation for each session with GMM ({n_clusters} components)")
                        else:
                            logger.info(f"Apply individual segmentation for each session with {n_clusters} clusters")

                        labels, cluster_center, motif_usages = individual_segmentation(
                            cfg=config,
                            sessions=sessions,
                            latent_vectors=latent_vectors,
                            n_clusters=n_clusters,
                            segmentation_algorithm=seg,
                        )
                    else:
                        if seg == "dbscan":
                            logger.info("Apply the same segmentation for all sessions using DBSCAN")
                        elif seg == "hmm":
                            logger.info(f"Apply the same segmentation for all sessions with HMM ({n_clusters} states)")
                        elif seg == "gmm":
                            logger.info(f"Apply the same segmentation for all sessions with GMM ({n_clusters} components)")
                        else:
                            logger.info(f"Apply the same segmentation for all sessions with {n_clusters} clusters")

                        labels, cluster_center, motif_usages = same_segmentation(
                            cfg=config,
                            sessions=sessions,
                            latent_vectors=latent_vectors,
                            n_clusters=n_clusters,
                            segmentation_algorithm=seg,
                        )
                else:
                    logger.info("No new segmentation has been calculated.")
                    new = False

            if new:
                for idx, session in enumerate(sessions):
                    save_dir = os.path.join(project_path, "results", session, model_name, segmentation_path)
                    os.makedirs(save_dir, exist_ok=True)

                    label_filename = f"{'' if seg == 'dbscan' else str(n_clusters) + '_'}{seg}_label_{session}.npy"
                    np.save(os.path.join(save_dir, label_filename), labels[idx])

                    if seg in ["kmeans", "gmm"]:
                        np.save(os.path.join(save_dir, f"cluster_center_{session}.npy"), cluster_center[idx])

                    np.save(os.path.join(save_dir, f"latent_vector_{session}.npy"), latent_vectors[idx])
                    np.save(os.path.join(save_dir, f"motif_usage_{session}.npy"), motif_usages[idx])

                logger.info("Segmentation completed. You can now run vame.motif_videos() to visualize motifs.")

    except Exception as e:
        logger.exception(f"An error occurred during pose segmentation: {e}")
    finally:
        logger_config.remove_file_handler()