import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture  # Add GMM import
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

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


def calculate_cluster_features(latent_data: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """
    Calculate representative features for each cluster for remapping.
    
    Parameters
    ----------
    latent_data : np.ndarray
        Latent space data
    cluster_labels : np.ndarray
        Cluster labels for the data
        
    Returns
    -------
    dict
        Dictionary mapping cluster ID to feature dictionary
    """
    cluster_features = {}
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
    
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_data = latent_data[cluster_mask]
        
        if len(cluster_data) > 0:
            features = {
                'centroid': np.mean(cluster_data, axis=0),
                'std': np.std(cluster_data, axis=0),
                'size': len(cluster_data),
                'variance': np.var(cluster_data, axis=0),
                'median': np.median(cluster_data, axis=0)
            }
            cluster_features[cluster] = features
    
    return cluster_features


def calculate_cluster_similarity(features1: dict, features2: dict) -> float:
    """
    Calculate similarity between two cluster feature sets.
    
    Parameters
    ----------
    features1, features2 : dict
        Feature dictionaries for two clusters
        
    Returns
    -------
    float
        Similarity score between 0 and 1
    """
    # Compare centroids (main behavioral pattern)
    centroid_sim = cosine_similarity([features1['centroid']], [features2['centroid']])[0, 0]
    
    # Compare variability patterns
    std_sim = cosine_similarity([features1['std']], [features2['std']])[0, 0]
    
    # Compare median patterns
    median_sim = cosine_similarity([features1['median']], [features2['median']])[0, 0]
    
    # Size similarity (normalized)
    size_sim = 1 - abs(features1['size'] - features2['size']) / max(features1['size'], features2['size'])
    
    # Weighted combination - prioritize behavioral patterns over size
    total_similarity = 0.5 * centroid_sim + 0.2 * std_sim + 0.2 * median_sim + 0.1 * size_sim
    return max(0, total_similarity)  # Ensure non-negative


def find_cluster_mapping(
    reference_features: dict, 
    target_features: dict,
    similarity_threshold: float = 0.7
) -> dict:
    """
    FIXED: Find cluster mapping for shared numbering system.
    Similar behaviors get the same number, unique behaviors get new numbers.
    
    Parameters
    ----------
    reference_features : dict
        Features from reference session/group
    target_features : dict
        Features from target session/group to be remapped
    similarity_threshold : float
        Minimum similarity for shared numbering (default: 0.7)
        
    Returns
    -------
    dict
        Mapping from target cluster IDs to final cluster IDs
    """
    ref_clusters = list(reference_features.keys())
    target_clusters = list(target_features.keys())
    
    if not ref_clusters or not target_clusters:
        logger.warning("Empty cluster set found during remapping")
        return {}
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(target_clusters), len(ref_clusters)))
    
    for i, target_cluster in enumerate(target_clusters):
        for j, ref_cluster in enumerate(ref_clusters):
            similarity_matrix[i, j] = calculate_cluster_similarity(
                target_features[target_cluster], 
                reference_features[ref_cluster]
            )
    
    # FIXED: Find matches above threshold using greedy assignment
    mapping = {}
    used_ref_clusters = set()
    
    # Sort target-reference pairs by similarity (highest first)
    pairs = []
    for i, target_cluster in enumerate(target_clusters):
        for j, ref_cluster in enumerate(ref_clusters):
            pairs.append((target_cluster, ref_cluster, similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
    
    # Greedy assignment: best matches first, only if above threshold
    for target_cluster, ref_cluster, similarity in pairs:
        if similarity >= similarity_threshold:
            if ref_cluster not in used_ref_clusters and target_cluster not in mapping:
                # This target behavior is similar enough to reference behavior
                mapping[target_cluster] = ref_cluster  # Use reference number
                used_ref_clusters.add(ref_cluster)
                
                logger.info(f"Shared behavior: target cluster {target_cluster} → "
                           f"reference cluster {ref_cluster} (similarity: {similarity:.3f})")
    
    # FIXED: Assign unique numbers to unmapped target clusters
    next_available_id = max(ref_clusters) + 1 if ref_clusters else 0
    
    for target_cluster in target_clusters:
        if target_cluster not in mapping:
            # This is a unique behavior - gets its own new number
            mapping[target_cluster] = next_available_id
            logger.info(f"Unique behavior: target cluster {target_cluster} → "
                       f"new cluster {next_available_id} (unique to target)")
            next_available_id += 1
    
    return mapping


def apply_cluster_remapping(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Apply cluster remapping to labels array.
    
    Parameters
    ----------
    labels : np.ndarray
        Original cluster labels
    mapping : dict
        Mapping from original to new cluster IDs
        
    Returns
    -------
    np.ndarray
        Remapped labels
    """
    remapped_labels = np.copy(labels)
    
    for original_cluster, new_cluster in mapping.items():
        remapped_labels[labels == original_cluster] = new_cluster
    
    return remapped_labels


def remap_dbscan_sessions_for_comparison(
    sessions: List[str], 
    latent_vectors: List[np.ndarray], 
    labels: List[np.ndarray],
    similarity_threshold: float = 0.7
) -> List[np.ndarray]:
    """
    FIXED: Remap DBSCAN clusters across sessions using shared numbering.
    Uses the first session as reference and remaps others to match.
    
    Parameters
    ----------
    sessions : List[str]
        Session names
    latent_vectors : List[np.ndarray]
        Latent vectors for each session
    labels : List[np.ndarray]
        Original DBSCAN labels for each session
    similarity_threshold : float
        Minimum similarity for shared numbering
        
    Returns
    -------
    List[np.ndarray]
        Remapped labels for each session
    """
    if len(sessions) <= 1:
        return labels  # No remapping needed for single session
    
    logger.info("Starting DBSCAN shared numbering for cross-session comparison...")
    logger.info(f"Using similarity threshold: {similarity_threshold}")
    
    # Use first session as reference
    reference_session = sessions[0]
    reference_features = calculate_cluster_features(latent_vectors[0], labels[0])
    
    logger.info(f"Using session '{reference_session}' as reference with "
               f"{len(reference_features)} clusters")
    
    remapped_labels = [labels[0]]  # Reference session stays unchanged
    
    # Remap each subsequent session to match reference
    for i in range(1, len(sessions)):
        session_name = sessions[i]
        session_features = calculate_cluster_features(latent_vectors[i], labels[i])
        
        logger.info(f"Creating shared numbering for session '{session_name}' "
                   f"({len(session_features)} clusters) to match reference...")
        
        if not session_features:
            logger.warning(f"No valid clusters found in session '{session_name}', keeping original labels")
            remapped_labels.append(labels[i])
            continue
        
        # Find shared numbering mapping
        mapping = find_cluster_mapping(reference_features, session_features, similarity_threshold)
        
        if not mapping:
            logger.warning(f"Could not create mapping for session '{session_name}', keeping original labels")
            remapped_labels.append(labels[i])
            continue
        
        # Apply remapping
        session_remapped = apply_cluster_remapping(labels[i], mapping)
        remapped_labels.append(session_remapped)
        
        # Log mapping summary
        unique_original = len(np.unique(labels[i][labels[i] != -1]))
        unique_remapped = len(np.unique(session_remapped[session_remapped != -1]))
        shared_count = len([v for v in mapping.values() if v in reference_features.keys()])
        unique_count = len(mapping) - shared_count
        
        logger.info(f"Session '{session_name}': {unique_original} → {unique_remapped} motifs "
                   f"({shared_count} shared, {unique_count} unique)")
    
    logger.info("DBSCAN shared numbering completed!")
    return remapped_labels


def group_aware_dbscan_segmentation(
    cfg: dict,
    sessions: List[str],
    group_labels: List[int],
    latent_vectors: List[np.ndarray],
    similarity_threshold: float = 0.7,
) -> Tuple[List[np.ndarray], Dict]:
    """
    FIXED: Enhanced DBSCAN segmentation with group-aware shared numbering.
    
    This approach ensures that motif numbers represent the same behaviors
    across groups for meaningful comparison. Similar behaviors get the same
    motif numbers, unique behaviors get their own numbers.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    sessions : List[str]
        Session names
    group_labels : List[int]
        Group assignment for each session (e.g., [1, 1, 2, 2])
    latent_vectors : List[np.ndarray]
        Latent vectors for each session
    similarity_threshold : float
        Minimum similarity for shared numbering (default: 0.7)
        
    Returns
    -------
    Tuple[List[np.ndarray], Dict]
        - Remapped labels for each session
        - Mapping information dictionary
    """
    
    # Step 1: Separate sessions by group
    unique_groups = list(set(group_labels))
    group_indices = {group: [i for i, g in enumerate(group_labels) if g == group] for group in unique_groups}
    
    logger.info(f"Groups found: {unique_groups}")
    logger.info(f"Using shared numbering with similarity threshold: {similarity_threshold}")
    for group, indices in group_indices.items():
        group_sessions = [sessions[i] for i in indices]
        logger.info(f"Group {group} sessions: {group_sessions}")
    
    # Step 2: Apply clustering within each group separately
    group_results = {}
    for group, indices in group_indices.items():
        group_sessions = [sessions[i] for i in indices]
        group_vectors = [latent_vectors[i] for i in indices]
        
        logger.info(f"Clustering Group {group}...")
        group_labels_result, group_features = cluster_group_sessions(
            group_vectors, group_sessions, cfg, f"Group{group}"
        )
        
        group_results[group] = {
            'labels': group_labels_result,
            'features': group_features,
            'sessions': group_sessions,
            'indices': indices
        }
    
    # Step 3: FIXED - Create cross-group shared numbering (use first group as reference)
    reference_group = unique_groups[0]
    reference_features = group_results[reference_group]['features']
    
    final_labels = [None] * len(sessions)
    mapping_info = {
        'group_mappings': {}, 
        'reference_group': reference_group,
        'similarity_threshold': similarity_threshold
    }
    
    # Set reference group labels (keep original numbers)
    for i, session_idx in enumerate(group_results[reference_group]['indices']):
        final_labels[session_idx] = group_results[reference_group]['labels'][i]
    
    # Map other groups to reference using shared numbering
    for group in unique_groups[1:]:
        logger.info(f"Creating shared numbering: Group {group} → Group {reference_group}")
        
        target_features = group_results[group]['features']
        # FIXED: Use shared numbering mapping with similarity threshold
        behavioral_mapping = find_cluster_mapping(reference_features, target_features, similarity_threshold)
        
        # Apply mapping to this group's sessions
        mapped_labels = []
        for session_labels in group_results[group]['labels']:
            mapped_session_labels = apply_cluster_remapping(session_labels, behavioral_mapping)
            mapped_labels.append(mapped_session_labels)
        
        # Store results
        for i, session_idx in enumerate(group_results[group]['indices']):
            final_labels[session_idx] = mapped_labels[i]
        
        mapping_info['group_mappings'][group] = behavioral_mapping
    
    return final_labels, mapping_info


def cluster_group_sessions(
    group_vectors: List[np.ndarray],
    group_sessions: List[str],
    cfg: dict,
    group_name: str
) -> Tuple[List[np.ndarray], Dict]:
    """
    Apply DBSCAN clustering within a group of sessions.
    """
    # Concatenate all sessions in the group
    group_data = np.concatenate(group_vectors, axis=0)
    
    # Normalize data
    scaler = StandardScaler()
    group_data_scaled = scaler.fit_transform(group_data)
    
    # Tune DBSCAN parameters for this group
    eps_val, min_samples_val = tune_dbscan_parameters(group_data_scaled, cfg)
    
    logger.info(f"{group_name}: Using eps={eps_val:.4f}, min_samples={min_samples_val}")
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    combined_labels = dbscan.fit_predict(group_data_scaled)
    
    # Check results
    unique_labels = np.unique(combined_labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(combined_labels == -1)
    
    logger.info(f"{group_name}: Found {n_clusters} clusters, {n_noise} noise points "
               f"({100*n_noise/len(combined_labels):.1f}%)")
    
    # Split labels back to individual sessions
    session_labels = []
    idx = 0
    for session_data in group_vectors:
        session_len = session_data.shape[0]
        session_label = combined_labels[idx:idx + session_len]
        session_labels.append(session_label)
        idx += session_len
    
    # Calculate cluster features for mapping
    cluster_features = calculate_cluster_features(group_data, combined_labels)
    
    return session_labels, cluster_features


def validate_cross_group_mapping(
    final_labels: List[np.ndarray],
    group_labels: List[int],
    sessions: List[str]
) -> None:
    """
    ENHANCED: Validate that the cross-group mapping preserved behavioral meaning.
    """
    logger.info("="*60)
    logger.info("SHARED NUMBERING VALIDATION")
    logger.info("="*60)
    
    unique_groups = list(set(group_labels))
    group_indices = {group: [i for i, g in enumerate(group_labels) if g == group] for group in unique_groups}
    
    # Get all unique motifs across all groups
    all_motifs = set()
    group_motifs = {}
    
    for group in unique_groups:
        group_motif_set = set()
        for idx in group_indices[group]:
            labels = final_labels[idx]
            unique_motifs = np.unique(labels[labels != -1])
            group_motif_set.update(unique_motifs)
        group_motifs[group] = group_motif_set
        all_motifs.update(group_motif_set)
    
    logger.info(f"Total unique motifs across all groups: {len(all_motifs)}")
    
    # Identify shared vs unique motifs
    if len(unique_groups) == 2:
        group1, group2 = unique_groups
        shared_motifs = group_motifs[group1] & group_motifs[group2]
        group1_unique = group_motifs[group1] - group_motifs[group2]
        group2_unique = group_motifs[group2] - group_motifs[group1]
        
        logger.info(f"\n📊 MOTIF DISTRIBUTION:")
        logger.info(f"Shared motifs (same behavior): {sorted(shared_motifs)} ({len(shared_motifs)} total)")
        logger.info(f"Group {group1} unique motifs: {sorted(group1_unique)} ({len(group1_unique)} total)")
        logger.info(f"Group {group2} unique motifs: {sorted(group2_unique)} ({len(group2_unique)} total)")
        
        overlap_pct = len(shared_motifs) / len(all_motifs) * 100 if all_motifs else 0
        logger.info(f"\nBehavioral overlap: {overlap_pct:.1f}%")
        
        if overlap_pct > 70:
            logger.info("→ High overlap: Groups have very similar behavioral repertoires")
        elif overlap_pct > 40:
            logger.info("→ Moderate overlap: Groups share core behaviors but have differences")
        else:
            logger.info("→ Low overlap: Groups have substantially different behaviors")
    
    # Compare motif usage for shared behaviors
    logger.info(f"\n🔗 SHARED BEHAVIOR ANALYSIS:")
    shared_motifs = set(group_motifs[unique_groups[0]])
    for group in unique_groups[1:]:
        shared_motifs &= group_motifs[group]
    
    if shared_motifs:
        logger.info("Motif | " + " | ".join([f"Group{g}_Count" for g in unique_groups]))
        logger.info("-" * (40 + 12 * len(unique_groups)))
        
        for motif in sorted(shared_motifs):
            group_counts = {}
            for group in unique_groups:
                total_count = 0
                for idx in group_indices[group]:
                    labels = final_labels[idx]
                    total_count += np.sum(labels == motif)
                group_counts[group] = total_count
            
            count_str = " | ".join([f"{group_counts[g]:10d}" for g in unique_groups])
            logger.info(f"{motif:5d} | {count_str}")
    else:
        logger.info("No shared behaviors found between groups")
    
    logger.info("\n✅ Validation complete! Ready for direct motif comparison.")


def save_group_aware_results(
    config: dict,
    sessions: List[str],
    final_labels: List[np.ndarray],
    mapping_info: Dict,
    group_labels: List[int]
) -> None:
    """
    ENHANCED: Save the group-aware segmentation results with shared numbering.
    """
    project_path = Path(config["project_path"])
    model_name = config["model_name"]
    
    # Create results directory
    results_dir = project_path / "results"
    
    for idx, session in enumerate(sessions):
        session_dir = results_dir / session / model_name / "dbscan_shared_numbering"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save labels with shared numbering
        np.save(session_dir / f"dbscan_shared_numbering_label_{session}.npy", final_labels[idx])
        
        logger.info(f"Saved shared numbering labels for session: {session}")
    
    # Save mapping information
    mapping_dir = results_dir / "group_comparison" / "dbscan_shared_numbering"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    
    # Save behavioral mapping with detailed analysis
    import json
    with open(mapping_dir / "shared_numbering_mapping.json", 'w') as f:
        json_mapping = {}
        for group, mapping in mapping_info['group_mappings'].items():
            json_mapping[str(group)] = {str(k): int(v) for k, v in mapping.items()}
        
        json.dump({
            'group_mappings': json_mapping,
            'reference_group': int(mapping_info['reference_group']),
            'similarity_threshold': mapping_info['similarity_threshold'],
            'group_labels': group_labels,
            'sessions': sessions,
            'mapping_type': 'shared_numbering'
        }, f, indent=2)
    
    logger.info("Saved shared numbering mapping information")


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
    similarity_threshold: float = 0.7,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    ENHANCED: Apply the same segmentation (shared clustering) to all sessions using the specified algorithm.
    For DBSCAN with multiple sessions, applies shared numbering for cross-session comparison.

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
    similarity_threshold : float
        Minimum similarity for shared numbering (DBSCAN only)

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
    session_labels_list = []
    for i, session in enumerate(sessions):
        session_len = latent_vectors[i].shape[0]
        session_labels = combined_labels[idx: idx + session_len]
        session_labels_list.append(session_labels)
        idx += session_len

    # FIXED: Apply shared numbering for DBSCAN with multiple sessions
    if segmentation_algorithm == "dbscan" and len(sessions) > 1:
        enable_remapping = cfg.get("dbscan_enable_remapping", True)
        if enable_remapping:
            logger.info("Applying DBSCAN shared numbering for cross-session comparison...")
            session_labels_list = remap_dbscan_sessions_for_comparison(
                sessions, latent_vectors, session_labels_list, similarity_threshold
            )
        else:
            logger.info("DBSCAN remapping disabled in config")

    # Finalize results
    for i, session in enumerate(sessions):
        labels.append(session_labels_list[i])

        if segmentation_algorithm in ["kmeans", "gmm"]:
            cluster_centers.append(clust_center)
        else:
            cluster_centers.append(None)

        # Motif usage: fixed length for kmeans/gmm/hmm, dynamic for dbscan
        usage = get_motif_usage(
            session_labels_list[i],
            None if segmentation_algorithm == "dbscan" else n_clusters
        )
        motif_usages.append(usage)

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
def segment_session(
    config: dict, 
    group_labels: List[int] = None, 
    similarity_threshold: float = 0.7,
    save_logs: bool = False
) -> None:
    """
    ENHANCED: segment_session function with shared numbering support.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    group_labels : List[int], optional
        Group assignment for each session (e.g., [1, 1, 2, 2] for 2 groups of 2 sessions each)
        If provided, will use shared numbering DBSCAN segmentation for cross-group comparison
    similarity_threshold : float
        Minimum similarity for shared numbering (0.0-1.0)
        Higher = stricter (fewer shared motifs, more unique motifs)
        Lower = more permissive (more shared motifs, fewer unique motifs)
    save_logs : bool, optional
        Whether to save logs to file
    """
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
        
        # Log shared numbering mode if enabled
        if group_labels is not None:
            logger.info(f"Shared numbering mode enabled with groups: {group_labels}")
            logger.info(f"Similarity threshold: {similarity_threshold}")

        for seg in segmentation_algorithms:
            segmentation_path = seg + ("-" + str(n_clusters) if seg != "dbscan" else "")
            
            # Modify path for shared numbering mode
            if group_labels is not None and seg == "dbscan":
                segmentation_path = "dbscan_shared_numbering"
                logger.info(f"Running SHARED NUMBERING pose segmentation using {seg} algorithm...")
            else:
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
                    read_from_variable=config.get("read_from_variable", "position_egocentric_aligned"),
                    tqdm_stream=tqdm_stream,
                )

                # SHARED NUMBERING PROCESSING
                if group_labels is not None and seg == "dbscan":
                    logger.info("Using shared numbering DBSCAN segmentation for cross-group comparison...")
                    
                    labels, mapping_info = group_aware_dbscan_segmentation(
                        cfg=config,
                        sessions=sessions,
                        group_labels=group_labels,
                        latent_vectors=latent_vectors,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Validate the mapping
                    validate_cross_group_mapping(labels, group_labels, sessions)
                    
                    # Create dummy cluster_center and motif_usages for compatibility
                    cluster_center = [None] * len(sessions)
                    motif_usages = []
                    for session_labels in labels:
                        usage = get_motif_usage(session_labels, None)  # None for DBSCAN
                        motif_usages.append(usage)
                
                # ORIGINAL PROCESSING
                else:
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
                            similarity_threshold=similarity_threshold,
                        )
            else:
                # Handle existing results
                if group_labels is not None and seg == "dbscan":
                    logger.info(f"Shared numbering DBSCAN segmentation already exists for model {model_name}")
                elif seg == "dbscan":
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
                        # Handle different path structures
                        if group_labels is not None and seg == "dbscan":
                            path = os.path.join(project_path, "results", session, model_name, "dbscan_shared_numbering")
                        else:
                            path = os.path.join(project_path, "results", session, model_name, segmentation_path)
                        latent_vectors.append(np.load(os.path.join(path, f"latent_vector_{session}.npy")))

                    # SHARED NUMBERING PROCESSING FOR EXISTING DATA
                    if group_labels is not None and seg == "dbscan":
                        logger.info("Using shared numbering DBSCAN segmentation for cross-group comparison...")
                        
                        labels, mapping_info = group_aware_dbscan_segmentation(
                            cfg=config,
                            sessions=sessions,
                            group_labels=group_labels,
                            latent_vectors=latent_vectors,
                            similarity_threshold=similarity_threshold
                        )
                        
                        validate_cross_group_mapping(labels, group_labels, sessions)
                        
                        cluster_center = [None] * len(sessions)
                        motif_usages = []
                        for session_labels in labels:
                            usage = get_motif_usage(session_labels, None)
                            motif_usages.append(usage)
                    
                    # ORIGINAL PROCESSING FOR EXISTING DATA
                    else:
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
                                similarity_threshold=similarity_threshold,
                            )
                else:
                    logger.info("No new segmentation has been calculated.")
                    new = False

            if new:
                for idx, session in enumerate(sessions):
                    save_dir = os.path.join(project_path, "results", session, model_name, segmentation_path)
                    os.makedirs(save_dir, exist_ok=True)

                    # MODIFIED SAVE LOGIC FOR SHARED NUMBERING RESULTS
                    if group_labels is not None and seg == "dbscan":
                        label_filename = f"dbscan_shared_numbering_label_{session}.npy"
                    else:
                        label_filename = f"{'' if seg == 'dbscan' else str(n_clusters) + '_'}{seg}_label_{session}.npy"
                    
                    np.save(os.path.join(save_dir, label_filename), labels[idx])

                    if seg in ["kmeans", "gmm"]:
                        np.save(os.path.join(save_dir, f"cluster_center_{session}.npy"), cluster_center[idx])

                    np.save(os.path.join(save_dir, f"latent_vector_{session}.npy"), latent_vectors[idx])
                    np.save(os.path.join(save_dir, f"motif_usage_{session}.npy"), motif_usages[idx])

                # SAVE SHARED NUMBERING MAPPING INFO
                if group_labels is not None and seg == "dbscan":
                    save_group_aware_results(config, sessions, labels, mapping_info, group_labels)

                logger.info("Segmentation completed. You can now run vame.motif_videos() to visualize motifs.")

    except Exception as e:
        logger.exception(f"An error occurred during pose segmentation: {e}")
    finally:
        logger_config.remove_file_handler()


def analyze_motif_frequencies(config: dict, sessions: List[str], segmentation_algorithm: str = "dbscan") -> dict:
    """
    Analyze and compare motif frequencies across sessions after shared numbering.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    sessions : List[str]
        List of session names
    segmentation_algorithm : str
        Segmentation algorithm used
        
    Returns
    -------
    dict
        Analysis results with frequencies and comparisons
    """
    project_path = Path(config["project_path"])
    model_name = config["model_name"]
    n_clusters = config["n_clusters"]
    
    # Build path based on algorithm
    if segmentation_algorithm == "dbscan":
        # Check for shared numbering results first
        shared_numbering_path = project_path / "results" / sessions[0] / model_name / "dbscan_shared_numbering"
        if shared_numbering_path.exists():
            segmentation_path = "dbscan_shared_numbering"
            label_prefix = "dbscan_shared_numbering_label"
            logger.info("Using shared numbering DBSCAN results for analysis")
        else:
            segmentation_path = "dbscan"
            label_prefix = "dbscan_label"
            logger.info("Using regular DBSCAN results for analysis")
    else:
        segmentation_path = f"{segmentation_algorithm}-{n_clusters}"
        label_prefix = f"{n_clusters}_{segmentation_algorithm}_label"
    
    results = {
        'sessions': sessions,
        'motif_counts': {},
        'motif_comparisons': {},
        'summary': {}
    }
    
    all_labels = []
    
    # Load labels for each session
    for session in sessions:
        label_path = project_path / "results" / session / model_name / segmentation_path / f"{label_prefix}_{session}.npy"
        
        if label_path.exists():
            labels = np.load(label_path)
            all_labels.append(labels)
            
            # Calculate counts for this session
            unique_labels = np.unique(labels[labels != -1])  # Exclude noise
            
            session_counts = {}
            for motif in unique_labels:
                count = np.sum(labels == motif)
                session_counts[int(motif)] = int(count)
            
            results['motif_counts'][session] = session_counts
            logger.info(f"Session '{session}': {len(unique_labels)} motifs analyzed")
        else:
            logger.warning(f"Labels not found for session '{session}' at {label_path}")
    
    # Cross-session comparison
    if len(all_labels) > 1:
        all_motifs = set()
        for session in results['motif_counts']:
            all_motifs.update(results['motif_counts'][session].keys())
        
        for motif in sorted(all_motifs):
            motif_comparison = {}
            for session in sessions:
                if session in results['motif_counts']:
                    count = results['motif_counts'][session].get(motif, 0)
                    motif_comparison[session] = count
                else:
                    motif_comparison[session] = 0
            
            # Calculate statistics
            counts = list(motif_comparison.values())
            results['motif_comparisons'][motif] = {
                'counts': motif_comparison,
                'total': sum(counts),
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts)),
                'max_session': max(motif_comparison, key=motif_comparison.get),
                'min_session': min(motif_comparison, key=motif_comparison.get)
            }
    
    # Generate summary
    total_motifs = len(results['motif_comparisons'])
    results['summary'] = {
        'total_motifs_found': total_motifs,
        'sessions_analyzed': len([s for s in sessions if s in results['motif_counts']]),
        'shared_numbering_used': "shared_numbering" in segmentation_path
    }
    
    return results


def print_motif_analysis_report(analysis_results: dict) -> None:
    """
    Print a formatted report of motif analysis results for shared numbering.
    
    Parameters
    ----------
    analysis_results : dict
        Results from analyze_motif_frequencies function
    """
    print("\n" + "="*80)
    print("MOTIF USAGE ANALYSIS REPORT (SHARED NUMBERING)")
    print("="*80)
    
    summary = analysis_results['summary']
    print(f"Total motifs found: {summary['total_motifs_found']}")
    print(f"Sessions analyzed: {summary['sessions_analyzed']}")
    print(f"Shared numbering used: {'Yes' if summary['shared_numbering_used'] else 'No'}")
    
    if summary['shared_numbering_used']:
        print("✅ Same motif numbers = same behaviors across sessions/groups")
    
    print("\n" + "-"*60)
    print("MOTIF USAGE COMPARISON (COUNTS)")
    print("-"*60)
    
    # Sort motifs by total usage (most used first)
    motifs_by_usage = sorted(
        analysis_results['motif_comparisons'].items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )
    
    for motif, data in motifs_by_usage:
        print(f"\nMotif {motif}:")
        print(f"  Total usage: {data['total']} events")
        print(f"  Mean per session: {data['mean']:.1f} ± {data['std']:.1f}")
        print(f"  Highest in: {data['max_session']} ({data['counts'][data['max_session']]} events)")
        print(f"  Lowest in: {data['min_session']} ({data['counts'][data['min_session']]} events)")
        
        # Show usage across all sessions
        session_usage = []
        for session, count in data['counts'].items():
            session_usage.append(f"{session}: {count}")
        print(f"  Usage breakdown: {', '.join(session_usage)}")
        
        # Show interpretation
        max_count = data['counts'][data['max_session']]
        min_count = data['counts'][data['min_session']]
        if max_count > 2 * min_count and data['std'] > data['mean'] * 0.3:
            print(f"  → Motif {motif} shows high variability across sessions")
        elif data['std'] < data['mean'] * 0.2:
            print(f"  → Motif {motif} is consistently used across sessions")
    
    print("\n" + "-"*60)
    print("SESSION-WISE BREAKDOWN")
    print("-"*60)
    
    for session in analysis_results['sessions']:
        if session in analysis_results['motif_counts']:
            session_data = analysis_results['motif_counts'][session]
            total_count = sum(session_data.values())
            print(f"\n{session}:")
            print(f"  Total behavioral events: {total_count}")
            print(f"  Motifs present: {len(session_data)}")
            
            # Show top 3 most frequent motifs
            sorted_motifs = sorted(session_data.items(), key=lambda x: x[1], reverse=True)
            print("  Top motifs:")
            for motif, count in sorted_motifs[:3]:
                percentage = (count / total_count) * 100 if total_count > 0 else 0
                print(f"    Motif {motif}: {count} events ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("ANALYSIS GUIDE:")
    print("• Same motif numbers across sessions = same behaviors")
    print("• Different motif numbers = different behaviors")
    print("• Use counts for statistical comparisons (t-tests, etc.)")
    print("• Zero counts indicate behavior absent in that session/group")
    print("="*80)


def print_comparison_guide_for_groups(
    control_sessions: List[str], 
    treatment_sessions: List[str],
    analysis_results: dict
) -> None:
    """
    Print a guide for comparing motif usage between control and treatment groups.
    
    Parameters
    ----------
    control_sessions : List[str]
        List of control session names
    treatment_sessions : List[str]
        List of treatment session names
    analysis_results : dict
        Results from analyze_motif_frequencies function
    """
    print("\n" + "="*80)
    print("GROUP COMPARISON GUIDE")
    print("="*80)
    
    # Calculate group totals
    control_totals = {}
    treatment_totals = {}
    
    all_motifs = set()
    
    # Aggregate counts per group
    for motif, data in analysis_results['motif_comparisons'].items():
        all_motifs.add(motif)
        
        control_total = sum(data['counts'].get(session, 0) for session in control_sessions)
        treatment_total = sum(data['counts'].get(session, 0) for session in treatment_sessions)
        
        control_totals[motif] = control_total
        treatment_totals[motif] = treatment_total
    
    print(f"Control sessions: {control_sessions}")
    print(f"Treatment sessions: {treatment_sessions}")
    
    print(f"\n📊 BEHAVIORAL COMPARISON:")
    print("Motif | Control_Total | Treatment_Total | Difference | Status")
    print("-" * 70)
    
    for motif in sorted(all_motifs):
        control_count = control_totals[motif]
        treatment_count = treatment_totals[motif]
        
        if control_count > 0 and treatment_count > 0:
            diff = treatment_count - control_count
            diff_pct = (diff / control_count) * 100
            status = "Both groups"
            if abs(diff_pct) > 50:
                status += f" (±{abs(diff_pct):.0f}%)"
        elif control_count > 0:
            diff = -control_count
            status = "Control only"
        elif treatment_count > 0:
            diff = treatment_count
            status = "Treatment only"
        else:
            diff = 0
            status = "Neither"
        
        print(f"{motif:5d} | {control_count:12d} | {treatment_count:14d} | {diff:10d} | {status}")
    
    print(f"\n🔍 SUGGESTED STATISTICAL TESTS:")
    print("For shared behaviors (both groups):")
    print("  • t-test: compare mean counts between groups")
    print("  • Mann-Whitney U: non-parametric alternative")
    print("  • Effect size: Cohen's d for practical significance")
    
    print("\nFor unique behaviors (one group only):")
    print("  • Presence/absence analysis")
    print("  • Fisher's exact test for categorical data")
    
    print(f"\n📈 EFFECT SIZE EXAMPLES:")
    shared_motifs = [m for m in all_motifs if control_totals[m] > 0 and treatment_totals[m] > 0]
    
    for motif in sorted(shared_motifs)[:3]:  # Show first 3 as examples
        control_count = control_totals[motif]
        treatment_count = treatment_totals[motif]
        
        # Simple effect size calculation
        pooled_mean = (control_count + treatment_count) / 2
        effect_size = abs(treatment_count - control_count) / pooled_mean if pooled_mean > 0 else 0
        
        if effect_size > 0.8:
            effect_desc = "Large effect"
        elif effect_size > 0.5:
            effect_desc = "Medium effect"
        elif effect_size > 0.2:
            effect_desc = "Small effect"
        else:
            effect_desc = "Minimal effect"
        
        print(f"  Motif {motif}: {effect_desc} (effect size: {effect_size:.2f})")
    
    print("="*80)


# UTILITY FUNCTIONS FOR EASY USAGE

def run_shared_numbering_analysis(
    config: dict,
    control_sessions: List[str],
    treatment_sessions: List[str],
    similarity_threshold: float = 0.7
) -> dict:
    """
    Complete pipeline for shared numbering analysis between two groups.
    
    Parameters
    ----------
    config : dict
        VAME configuration dictionary
    control_sessions : List[str]
        List of control session names
    treatment_sessions : List[str]
        List of treatment session names
    similarity_threshold : float
        Similarity threshold for shared numbering
        
    Returns
    -------
    dict
        Complete analysis results
    """
    # Prepare group labels
    all_sessions = control_sessions + treatment_sessions
    group_labels = [1] * len(control_sessions) + [2] * len(treatment_sessions)
    
    # Update config
    config["session_names"] = all_sessions
    
    logger.info("="*60)
    logger.info("RUNNING SHARED NUMBERING ANALYSIS")
    logger.info("="*60)
    logger.info(f"Control sessions: {control_sessions}")
    logger.info(f"Treatment sessions: {treatment_sessions}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    
    # Run segmentation with shared numbering
    segment_session(
        config=config,
        group_labels=group_labels,
        similarity_threshold=similarity_threshold
    )
    
    # Analyze results
    analysis_results = analyze_motif_frequencies(config, all_sessions, "dbscan")
    
    # Print reports
    print_motif_analysis_report(analysis_results)
    print_comparison_guide_for_groups(control_sessions, treatment_sessions, analysis_results)
    
    return analysis_results


def load_motif_usage_arrays(
    config: dict,
    sessions: List[str],
    use_shared_numbering: bool = True
) -> List[np.ndarray]:
    """
    Load motif usage arrays from saved results.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    sessions : List[str]
        List of session names
    use_shared_numbering : bool
        Whether to load shared numbering results
        
    Returns
    -------
    List[np.ndarray]
        List of motif usage arrays for each session
    """
    project_path = Path(config["project_path"])
    model_name = config["model_name"]
    
    if use_shared_numbering:
        segmentation_path = "dbscan_shared_numbering"
    else:
        segmentation_path = "dbscan"
    
    motif_usage_arrays = []
    
    for session in sessions:
        usage_path = project_path / "results" / session / model_name / segmentation_path / f"motif_usage_{session}.npy"
        
        if usage_path.exists():
            usage_array = np.load(usage_path)
            motif_usage_arrays.append(usage_array)
            logger.info(f"Loaded motif usage for {session}: {len(usage_array)} motifs")
        else:
            logger.error(f"Motif usage file not found: {usage_path}")
            raise FileNotFoundError(f"Motif usage file not found: {usage_path}")
    
    return motif_usage_arrays


# EXAMPLE USAGE FUNCTIONS

def example_drug_study_analysis():
    """
    Example of how to use the shared numbering system for a drug study.
    """
    # Example configuration (replace with your actual config)
    config = {
        "project_path": "/path/to/your/vame/project",
        "model_name": "your_model_name",
        "n_clusters": 10,  # Not used for DBSCAN
        "segmentation_algorithms": ["dbscan"],
        "individual_segmentation": False,
        "all_data": "yes",
        "egocentric_data": True,
        # Add other required config parameters
    }
    
    # Define your sessions
    control_sessions = ["control_1", "control_2", "control_3"]
    treatment_sessions = ["drug_1", "drug_2", "drug_3"]
    
    # Run complete analysis
    analysis_results = run_shared_numbering_analysis(
        config=config,
        control_sessions=control_sessions,
        treatment_sessions=treatment_sessions,
        similarity_threshold=0.7  # Adjust as needed
    )
    
    # Load motif usage arrays for further statistical analysis
    all_sessions = control_sessions + treatment_sessions
    motif_usage_arrays = load_motif_usage_arrays(config, all_sessions)
    
    # Now you can do statistical tests
    control_usage = motif_usage_arrays[:len(control_sessions)]
    treatment_usage = motif_usage_arrays[len(control_sessions):]
    
    print(f"Control group motif usage arrays: {len(control_usage)} sessions")
    print(f"Treatment group motif usage arrays: {len(treatment_usage)} sessions")
    print("Ready for statistical analysis!")
    
    return analysis_results, control_usage, treatment_usage


def example_statistical_comparison(
    control_usage: List[np.ndarray],
    treatment_usage: List[np.ndarray]
) -> None:
    """
    Example of statistical comparison using the motif usage arrays.
    
    Parameters
    ----------
    control_usage : List[np.ndarray]
        Motif usage arrays for control group
    treatment_usage : List[np.ndarray]
        Motif usage arrays for treatment group
    """
    from scipy import stats
    
    # Ensure all arrays have the same length (pad with zeros if needed)
    max_motifs = max(
        max(len(arr) for arr in control_usage),
        max(len(arr) for arr in treatment_usage)
    )
    
    # Pad arrays to same length
    control_padded = []
    for arr in control_usage:
        padded = np.zeros(max_motifs, dtype=int)
        padded[:len(arr)] = arr
        control_padded.append(padded)
    
    treatment_padded = []
    for arr in treatment_usage:
        padded = np.zeros(max_motifs, dtype=int)
        padded[:len(arr)] = arr
        treatment_padded.append(padded)
    
    # Convert to numpy arrays
    control_matrix = np.array(control_padded)  # Shape: (n_control_sessions, n_motifs)
    treatment_matrix = np.array(treatment_padded)  # Shape: (n_treatment_sessions, n_motifs)
    
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON RESULTS")
    print("="*60)
    
    # Perform t-tests for each motif
    for motif_idx in range(max_motifs):
        control_counts = control_matrix[:, motif_idx]
        treatment_counts = treatment_matrix[:, motif_idx]
        
        # Only test motifs that appear in at least one group
        if np.sum(control_counts) > 0 or np.sum(treatment_counts) > 0:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(control_counts, treatment_counts)
            
            # Calculate means
            control_mean = np.mean(control_counts)
            treatment_mean = np.mean(treatment_counts)
            
            # Calculate effect size (Cohen's d)import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture  # Add GMM import
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

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


def calculate_cluster_features(latent_data: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """
    Calculate representative features for each cluster for remapping.
    
    Parameters
    ----------
    latent_data : np.ndarray
        Latent space data
    cluster_labels : np.ndarray
        Cluster labels for the data
        
    Returns
    -------
    dict
        Dictionary mapping cluster ID to feature dictionary
    """
    cluster_features = {}
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
    
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_data = latent_data[cluster_mask]
        
        if len(cluster_data) > 0:
            features = {
                'centroid': np.mean(cluster_data, axis=0),
                'std': np.std(cluster_data, axis=0),
                'size': len(cluster_data),
                'variance': np.var(cluster_data, axis=0),
                'median': np.median(cluster_data, axis=0)
            }
            cluster_features[cluster] = features
    
    return cluster_features


def calculate_cluster_similarity(features1: dict, features2: dict) -> float:
    """
    Calculate similarity between two cluster feature sets.
    
    Parameters
    ----------
    features1, features2 : dict
        Feature dictionaries for two clusters
        
    Returns
    -------
    float
        Similarity score between 0 and 1
    """
    # Compare centroids (main behavioral pattern)
    centroid_sim = cosine_similarity([features1['centroid']], [features2['centroid']])[0, 0]
    
    # Compare variability patterns
    std_sim = cosine_similarity([features1['std']], [features2['std']])[0, 0]
    
    # Compare median patterns
    median_sim = cosine_similarity([features1['median']], [features2['median']])[0, 0]
    
    # Size similarity (normalized)
    size_sim = 1 - abs(features1['size'] - features2['size']) / max(features1['size'], features2['size'])
    
    # Weighted combination - prioritize behavioral patterns over size
    total_similarity = 0.5 * centroid_sim + 0.2 * std_sim + 0.2 * median_sim + 0.1 * size_sim
    return max(0, total_similarity)  # Ensure non-negative


def find_cluster_mapping(
    reference_features: dict, 
    target_features: dict,
    similarity_threshold: float = 0.7
) -> dict:
    """
    FIXED: Find cluster mapping for shared numbering system.
    Similar behaviors get the same number, unique behaviors get new numbers.
    
    Parameters
    ----------
    reference_features : dict
        Features from reference session/group
    target_features : dict
        Features from target session/group to be remapped
    similarity_threshold : float
        Minimum similarity for shared numbering (default: 0.7)
        
    Returns
    -------
    dict
        Mapping from target cluster IDs to final cluster IDs
    """
    ref_clusters = list(reference_features.keys())
    target_clusters = list(target_features.keys())
    
    if not ref_clusters or not target_clusters:
        logger.warning("Empty cluster set found during remapping")
        return {}
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(target_clusters), len(ref_clusters)))
    
    for i, target_cluster in enumerate(target_clusters):
        for j, ref_cluster in enumerate(ref_clusters):
            similarity_matrix[i, j] = calculate_cluster_similarity(
                target_features[target_cluster], 
                reference_features[ref_cluster]
            )
    
    # FIXED: Find matches above threshold using greedy assignment
    mapping = {}
    used_ref_clusters = set()
    
    # Sort target-reference pairs by similarity (highest first)
    pairs = []
    for i, target_cluster in enumerate(target_clusters):
        for j, ref_cluster in enumerate(ref_clusters):
            pairs.append((target_cluster, ref_cluster, similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
    
    # Greedy assignment: best matches first, only if above threshold
    for target_cluster, ref_cluster, similarity in pairs:
        if similarity >= similarity_threshold:
            if ref_cluster not in used_ref_clusters and target_cluster not in mapping:
                # This target behavior is similar enough to reference behavior
                mapping[target_cluster] = ref_cluster  # Use reference number
                used_ref_clusters.add(ref_cluster)
                
                logger.info(f"Shared behavior: target cluster {target_cluster} → "
                           f"reference cluster {ref_cluster} (similarity: {similarity:.3f})")
    
    # FIXED: Assign unique numbers to unmapped target clusters
    next_available_id = max(ref_clusters) + 1 if ref_clusters else 0
    
    for target_cluster in target_clusters:
        if target_cluster not in mapping:
            # This is a unique behavior - gets its own new number
            mapping[target_cluster] = next_available_id
            logger.info(f"Unique behavior: target cluster {target_cluster} → "
                       f"new cluster {next_available_id} (unique to target)")
            next_available_id += 1
    
    return mapping


def apply_cluster_remapping(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Apply cluster remapping to labels array.
    
    Parameters
    ----------
    labels : np.ndarray
        Original cluster labels
    mapping : dict
        Mapping from original to new cluster IDs
        
    Returns
    -------
    np.ndarray
        Remapped labels
    """
    remapped_labels = np.copy(labels)
    
    for original_cluster, new_cluster in mapping.items():
        remapped_labels[labels == original_cluster] = new_cluster
    
    return remapped_labels


def remap_dbscan_sessions_for_comparison(
    sessions: List[str], 
    latent_vectors: List[np.ndarray], 
    labels: List[np.ndarray],
    similarity_threshold: float = 0.7
) -> List[np.ndarray]:
    """
    FIXED: Remap DBSCAN clusters across sessions using shared numbering.
    Uses the first session as reference and remaps others to match.
    
    Parameters
    ----------
    sessions : List[str]
        Session names
    latent_vectors : List[np.ndarray]
        Latent vectors for each session
    labels : List[np.ndarray]
        Original DBSCAN labels for each session
    similarity_threshold : float
        Minimum similarity for shared numbering
        
    Returns
    -------
    List[np.ndarray]
        Remapped labels for each session
    """
    if len(sessions) <= 1:
        return labels  # No remapping needed for single session
    
    logger.info("Starting DBSCAN shared numbering for cross-session comparison...")
    logger.info(f"Using similarity threshold: {similarity_threshold}")
    
    # Use first session as reference
    reference_session = sessions[0]
    reference_features = calculate_cluster_features(latent_vectors[0], labels[0])
    
    logger.info(f"Using session '{reference_session}' as reference with "
               f"{len(reference_features)} clusters")
    
    remapped_labels = [labels[0]]  # Reference session stays unchanged
    
    # Remap each subsequent session to match reference
    for i in range(1, len(sessions)):
        session_name = sessions[i]
        session_features = calculate_cluster_features(latent_vectors[i], labels[i])
        
        logger.info(f"Creating shared numbering for session '{session_name}' "
                   f"({len(session_features)} clusters) to match reference...")
        
        if not session_features:
            logger.warning(f"No valid clusters found in session '{session_name}', keeping original labels")
            remapped_labels.append(labels[i])
            continue
        
        # Find shared numbering mapping
        mapping = find_cluster_mapping(reference_features, session_features, similarity_threshold)
        
        if not mapping:
            logger.warning(f"Could not create mapping for session '{session_name}', keeping original labels")
            remapped_labels.append(labels[i])
            continue
        
        # Apply remapping
        session_remapped = apply_cluster_remapping(labels[i], mapping)
        remapped_labels.append(session_remapped)
        
        # Log mapping summary
        unique_original = len(np.unique(labels[i][labels[i] != -1]))
        unique_remapped = len(np.unique(session_remapped[session_remapped != -1]))
        shared_count = len([v for v in mapping.values() if v in reference_features.keys()])
        unique_count = len(mapping) - shared_count
        
        logger.info(f"Session '{session_name}': {unique_original} → {unique_remapped} motifs "
                   f"({shared_count} shared, {unique_count} unique)")
    
    logger.info("DBSCAN shared numbering completed!")
    return remapped_labels


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
    unique_labels, usage_full = np.unique(session_labels, return_counts=True)

    # Handle DBSCAN case: exclude noise (-1) and use dynamic output
    if n_clusters is None:
        if -1 in unique_labels:
            logger.info("DBSCAN: Noise label (-1) detected. Ignoring in motif usage count.")
            unique_labels = unique_labels[unique_labels != -1]
            # Remove count for noise label
            noise_index = np.where(np.unique(session_labels) == -1)[0]
            if len(noise_index) > 0:
                usage_full = np.delete(usage_full, noise_index[0])

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


# NEW FUNCTION: Create sequential video labels
def create_sequential_video_labels(
    config: dict,
    sessions: List[str],
    labels: List[np.ndarray],
    model_name: str,
    segmentation_algorithm: str = "dbscan"
) -> None:
    """
    Create sequential video labels (0, 1, 2, 3...) from shared numbering results.
    This solves the issue of having motif numbers like 0, 4, 6, 72, 96.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    sessions : List[str]
        List of session names
    labels : List[np.ndarray]
        List of label arrays for each session
    model_name : str
        Model name
    segmentation_algorithm : str
        Segmentation algorithm
    """
    
    logger.info("Creating sequential video labels for clean video numbering...")
    
    for i, session in enumerate(sessions):
        session_labels = labels[i]
        
        # Get unique labels (excluding noise -1)
        unique_labels = np.unique(session_labels[session_labels != -1])
        
        if len(unique_labels) == 0:
            logger.warning(f"No valid clusters found for session {session}")
            continue
        
        # Create sequential mapping: original -> 0, 1, 2, 3...
        sequential_mapping = {orig: new for new, orig in enumerate(sorted(unique_labels))}
        
        # Apply sequential mapping
        sequential_labels = np.copy(session_labels)
        for orig_label, new_label in sequential_mapping.items():
            sequential_labels[session_labels == orig_label] = new_label
        
        # Determine save path based on algorithm
        if segmentation_algorithm == "dbscan":
            # Check if shared numbering was used
            shared_numbering_path = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                "dbscan_shared_numbering"
            )
            
            if os.path.exists(shared_numbering_path):
                save_path = shared_numbering_path
                video_label_filename = f"dbscan_video_sequential_label_{session}.npy"
            else:
                save_path = os.path.join(
                    config["project_path"],
                    "results",
                    session,
                    model_name,
                    "dbscan"
                )
                video_label_filename = f"dbscan_video_sequential_label_{session}.npy"
        else:
            save_path = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                f"{segmentation_algorithm}-{config['n_clusters']}"
            )
            video_label_filename = f"{segmentation_algorithm}_video_sequential_label_{session}.npy"
        
        # Save sequential labels for video creation
        sequential_file_path = os.path.join(save_path, video_label_filename)
        np.save(sequential_file_path, sequential_labels)
        
        # Save mapping information
        import json
        mapping_file = os.path.join(save_path, f"video_sequential_mapping_{session}.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                "original_to_sequential": {int(k): int(v) for k, v in sequential_mapping.items()},
                "sequential_to_original": {int(v): int(k) for k, v in sequential_mapping.items()},
                "n_clusters": len(unique_labels),
                "session": session,
                "purpose": "Sequential numbering for clean video creation"
            }, f, indent=2)
        
        logger.info(f"Session {session}: Created sequential labels 0-{len(unique_labels)-1} "
                   f"from original range {min(unique_labels)}-{max(unique_labels)}")


def same_segmentation(
    cfg: dict,
    sessions: List[str],
    latent_vectors: List[np.ndarray],
    n_clusters: int,
    segmentation_algorithm: str,
    similarity_threshold: float = 0.7,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    ENHANCED: Apply the same segmentation (shared clustering) to all sessions using the specified algorithm.
    For DBSCAN with multiple sessions, applies shared numbering for cross-session comparison.

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
    similarity_threshold : float
        Minimum similarity for shared numbering (DBSCAN only)

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
    session_labels_list = []
    for i, session in enumerate(sessions):
        session_len = latent_vectors[i].shape[0]
        session_labels = combined_labels[idx: idx + session_len]
        session_labels_list.append(session_labels)
        idx += session_len

    # FIXED: Apply shared numbering for DBSCAN with multiple sessions
    if segmentation_algorithm == "dbscan" and len(sessions) > 1:
        enable_remapping = cfg.get("dbscan_enable_remapping", True)
        if enable_remapping:
            logger.info("Applying DBSCAN shared numbering for cross-session comparison...")
            session_labels_list = remap_dbscan_sessions_for_comparison(
                sessions, latent_vectors, session_labels_list, similarity_threshold
            )
        else:
            logger.info("DBSCAN remapping disabled in config")

    # Finalize results
    for i, session in enumerate(sessions):
        labels.append(session_labels_list[i])

        if segmentation_algorithm in ["kmeans", "gmm"]:
            cluster_centers.append(clust_center)
        else:
            cluster_centers.append(None)

        # Motif usage: fixed length for kmeans/gmm/hmm, dynamic for dbscan
        usage = get_motif_usage(
            session_labels_list[i],
            None if segmentation_algorithm == "dbscan" else n_clusters
        )
        motif_usages.append(usage)

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
def segment_session(
    config: dict, 
    group_labels: List[int] = None, 
    similarity_threshold: float = 0.7,
    save_logs: bool = False,
    create_video_labels: bool = True  # NEW PARAMETER
) -> None:
    """
    ENHANCED: segment_session function with shared numbering support and video label creation.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    group_labels : List[int], optional
        Group assignment for each session (e.g., [1, 1, 2, 2] for 2 groups of 2 sessions each)
        If provided, will use shared numbering DBSCAN segmentation for cross-group comparison
    similarity_threshold : float
        Minimum similarity for shared numbering (0.0-1.0)
        Higher = stricter (fewer shared motifs, more unique motifs)
        Lower = more permissive (more shared motifs, fewer unique motifs)
    save_logs : bool, optional
        Whether to save logs to file
    create_video_labels : bool, optional
        Whether to create sequential video labels for clean video numbering (default: True)
    """
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
        
        # Log shared numbering mode if enabled
        if group_labels is not None:
            logger.info(f"Shared numbering mode enabled with groups: {group_labels}")
            logger.info(f"Similarity threshold: {similarity_threshold}")

        for seg in segmentation_algorithms:
            segmentation_path = seg + ("-" + str(n_clusters) if seg != "dbscan" else "")
            
            # Modify path for shared numbering mode
            if group_labels is not None and seg == "dbscan":
                segmentation_path = "dbscan_shared_numbering"
                logger.info(f"Running SHARED NUMBERING pose segmentation using {seg} algorithm...")
            else:
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
                    read_from_variable=config.get("read_from_variable", "position_egocentric_aligned"),
                    tqdm_stream=tqdm_stream,
                )

                # SHARED NUMBERING PROCESSING
                if group_labels is not None and seg == "dbscan":
                    logger.info("Using shared numbering DBSCAN segmentation for cross-group comparison...")
                    
                    # Apply group-aware processing would go here
                    # For now, use the standard shared numbering approach
                    if ind_seg:
                        labels, cluster_center, motif_usages = individual_segmentation(
                            cfg=config,
                            sessions=sessions,
                            latent_vectors=latent_vectors,
                            n_clusters=n_clusters,
                            segmentation_algorithm=seg,
                        )
                    else:
                        labels, cluster_center, motif_usages = same_segmentation(
                            cfg=config,
                            sessions=sessions,
                            latent_vectors=latent_vectors,
                            n_clusters=n_clusters,
                            segmentation_algorithm=seg,
                            similarity_threshold=similarity_threshold,
                        )
                
                # ORIGINAL PROCESSING
                else:
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
                            similarity_threshold=similarity_threshold,
                        )
            else:
                # Handle existing results
                if group_labels is not None and seg == "dbscan":
                    logger.info(f"Shared numbering DBSCAN segmentation already exists for model {model_name}")
                elif seg == "dbscan":
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
                        # Handle different path structures
                        if group_labels is not None and seg == "dbscan":
                            path = os.path.join(project_path, "results", session, model_name, "dbscan_shared_numbering")
                        else:
                            path = os.path.join(project_path, "results", session, model_name, segmentation_path)
                        latent_vectors.append(np.load(os.path.join(path, f"latent_vector_{session}.npy")))

                    # SHARED NUMBERING PROCESSING FOR EXISTING DATA
                    if group_labels is not None and seg == "dbscan":
                        logger.info("Using shared numbering DBSCAN segmentation for cross-group comparison...")
                        
                        if ind_seg:
                            labels, cluster_center, motif_usages = individual_segmentation(
                                cfg=config,
                                sessions=sessions,
                                latent_vectors=latent_vectors,
                                n_clusters=n_clusters,
                                segmentation_algorithm=seg,
                            )
                        else:
                            labels, cluster_center, motif_usages = same_segmentation(
                                cfg=config,
                                sessions=sessions,
                                latent_vectors=latent_vectors,
                                n_clusters=n_clusters,
                                segmentation_algorithm=seg,
                                similarity_threshold=similarity_threshold,
                            )
                    
                    # ORIGINAL PROCESSING FOR EXISTING DATA
                    else:
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
                                similarity_threshold=similarity_threshold,
                            )
                else:
                    logger.info("No new segmentation has been calculated.")
                    new = False

            if new:
                for idx, session in enumerate(sessions):
                    save_dir = os.path.join(project_path, "results", session, model_name, segmentation_path)
                    os.makedirs(save_dir, exist_ok=True)

                    # MODIFIED SAVE LOGIC FOR SHARED NUMBERING RESULTS
                    if group_labels is not None and seg == "dbscan":
                        label_filename = f"dbscan_shared_numbering_label_{session}.npy"
                    else:
                        label_filename = f"{'' if seg == 'dbscan' else str(n_clusters) + '_'}{seg}_label_{session}.npy"
                    
                    np.save(os.path.join(save_dir, label_filename), labels[idx])

                    if seg in ["kmeans", "gmm"]:
                        np.save(os.path.join(save_dir, f"cluster_center_{session}.npy"), cluster_center[idx])

                    np.save(os.path.join(save_dir, f"latent_vector_{session}.npy"), latent_vectors[idx])
                    np.save(os.path.join(save_dir, f"motif_usage_{session}.npy"), motif_usages[idx])

                # NEW: Create sequential video labels for clean numbering
                if create_video_labels:
                    logger.info("Creating sequential video labels for clean video numbering...")
                    create_sequential_video_labels(config, sessions, labels, model_name, seg)

                logger.info("Segmentation completed. You can now run vame.motif_videos() to visualize motifs.")

    except Exception as e:
        logger.exception(f"An error occurred during pose segmentation: {e}")
    finally:
        logger_config.remove_file_handler()


# UTILITY FUNCTIONS FOR VIDEO CREATION

def get_video_label_file_path(
    config: dict,
    session: str,
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
    use_sequential: bool = True
) -> str:
    """
    Get the path to video label files - either sequential (clean numbering) or original.
    
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
    use_sequential : bool
        Whether to use sequential video labels (default: True)
        
    Returns
    -------
    str
        Path to the label file
    """
    
    if segmentation_algorithm == "dbscan":
        # Check for shared numbering results first
        shared_numbering_path = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            "dbscan_shared_numbering"
        )
        
        if os.path.exists(shared_numbering_path):
            base_path = shared_numbering_path
            original_filename = f"dbscan_shared_numbering_label_{session}.npy"
            sequential_filename = f"dbscan_video_sequential_label_{session}.npy"
        else:
            base_path = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                "dbscan"
            )
            original_filename = f"dbscan_label_{session}.npy"
            sequential_filename = f"dbscan_video_sequential_label_{session}.npy"
    else:
        base_path = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            f"{segmentation_algorithm}-{n_clusters}"
        )
        original_filename = f"{n_clusters}_{segmentation_algorithm}_label_{session}.npy"
        sequential_filename = f"{segmentation_algorithm}_video_sequential_label_{session}.npy"
    
    if use_sequential:
        sequential_path = os.path.join(base_path, sequential_filename)
        if os.path.exists(sequential_path):
            logger.info(f"Using SEQUENTIAL video labels for clean numbering: {session}")
            return sequential_path
        else:
            logger.warning(f"Sequential video labels not found, using original: {session}")
            return os.path.join(base_path, original_filename)
    else:
        return os.path.join(base_path, original_filename)


def analyze_motif_distribution(config: dict, sessions: List[str] = None) -> None:
    """
    Analyze and display motif distribution across sessions.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    sessions : List[str], optional
        List of sessions to analyze (default: all sessions)
    """
    
    if sessions is None:
        sessions = config["session_names"]
    
    model_name = config["model_name"]
    
    print("\n" + "="*80)
    print("MOTIF DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for session in sessions:
        print(f"\n📊 SESSION: {session}")
        
        # Check for different types of labels
        label_types = []
        
        # Check for sequential video labels
        sequential_path = get_video_label_file_path(config, session, model_name, 15, "dbscan", use_sequential=True)
        if "video_sequential" in sequential_path and os.path.exists(sequential_path):
            labels = np.load(sequential_path)
            unique_labels = np.unique(labels[labels != -1])
            label_types.append(("Sequential (Video)", unique_labels))
        
        # Check for original labels
        original_path = get_video_label_file_path(config, session, model_name, 15, "dbscan", use_sequential=False)
        if os.path.exists(original_path):
            labels = np.load(original_path)
            unique_labels = np.unique(labels[labels != -1])
            label_types.append(("Original", unique_labels))
        
        for label_type, unique_labels in label_types:
            print(f"   {label_type}:")
            print(f"     Motifs: {sorted(unique_labels)} ({len(unique_labels)} total)")
            if len(unique_labels) > 0:
                print(f"     Range: {min(unique_labels)} to {max(unique_labels)}")
                
                # Check for gaps in numbering
                expected_range = set(range(min(unique_labels), max(unique_labels) + 1))
                actual_set = set(unique_labels)
                gaps = sorted(expected_range - actual_set)
                if gaps:
                    print(f"     Gaps: {gaps}")
                else:
                    print(f"     Numbering: Sequential ✓")

