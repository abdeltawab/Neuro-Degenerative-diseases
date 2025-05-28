---
sidebar_label: pose_segmentation
title: analysis.pose_segmentation
---

#### logger\_config

#### logger

#### embedd\_latent\_vectors

```python
def embedd_latent_vectors(
        cfg: dict,
        sessions: List[str],
        model: RNN_VAE,
        fixed: bool,
        read_from_variable: str = "position_egocentric_aligned",
        tqdm_stream: Union[TqdmToLogger, None] = None) -> List[np.ndarray]
```

#### estimate\_dbscan\_eps

```python
def estimate_dbscan_eps(data: np.ndarray, k: int = 4) -> float
```

Estimate optimal eps parameter for DBSCAN using k-distance graph method.

**Parameters**

* **data** (`np.ndarray`): Input data for clustering
* **k** (`int`): Number of nearest neighbors to consider (default: 4)

**Returns**

* `float`: Estimated eps value

#### tune\_dbscan\_parameters

```python
def tune_dbscan_parameters(data: np.ndarray, cfg: dict) -> Tuple[float, int]
```

Automatically tune DBSCAN parameters for the given data.
Uses iterative parameter testing to find optimal balance between 
number of clusters and meaningful cluster sizes.

**Parameters**

* **data** (`np.ndarray`): Input data for clustering
* **cfg** (`dict`): Configuration dictionary

**Returns**

* `Tuple[float, int]`: Tuned (eps, min_samples) parameters

#### get\_motif\_usage

```python
def get_motif_usage(session_labels: np.ndarray,
                    n_clusters: int = None) -> np.ndarray
```

Count motif usage from session label array.

**Parameters**

* **session_labels** (`np.ndarray`): Array of session labels.
* **n_clusters** (`int, optional`): Number of clusters. For KMeans and HMM, this should be set to get fixed-length output.
For DBSCAN, leave as None to infer cluster count dynamically (excluding noise -1).

**Returns**

* `np.ndarray`: Motif usage counts. Length = n_clusters for fixed methods, or dynamic for DBSCAN.

#### same\_segmentation

```python
def same_segmentation(
    cfg: dict, sessions: List[str], latent_vectors: List[np.ndarray],
    n_clusters: int, segmentation_algorithm: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
```

Apply the same segmentation (shared clustering) to all sessions using the specified algorithm.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **latent_vectors** (`List[np.ndarray]`): List of latent vector arrays per session.
* **n_clusters** (`int`): Number of clusters (only used for KMeans and HMM).
* **segmentation_algorithm** (`str`): One of: &quot;kmeans&quot;, &quot;hmm&quot;, or &quot;dbscan&quot;.

**Returns**

* `Tuple of:`: - labels: List of np.ndarray of predicted motif labels per session.
- cluster_centers: List of cluster centers (KMeans only).
- motif_usages: List of motif usage arrays per session.

#### individual\_segmentation

```python
def individual_segmentation(
    cfg: dict, sessions: List[str], latent_vectors: List[np.ndarray],
    n_clusters: int, segmentation_algorithm: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
```

#### segment\_session

```python
@save_state(model=SegmentSessionFunctionSchema)
def segment_session(config: dict, save_logs: bool = False) -> None
```

