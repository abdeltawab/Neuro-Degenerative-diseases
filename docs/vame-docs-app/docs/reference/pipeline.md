---
sidebar_label: pipeline
title: pipeline
---

#### logger\_config

#### logger

## VAMEPipeline Objects

```python
class VAMEPipeline()
```

VAME pipeline class.

#### \_\_init\_\_

```python
def __init__(project_name: str,
             videos: List[str],
             poses_estimations: List[str],
             source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
             working_directory: str = ".",
             video_type: str = ".mp4",
             fps: int | None = None,
             copy_videos: bool = False,
             paths_to_pose_nwb_series_data: Optional[str] = None,
             config_kwargs: Optional[dict] = None) -> None
```

Initializes the VAME pipeline.

**Parameters**

* **project_name** (`str`): Project name.
* **videos** (`List[str]`): List of video files.
* **poses_estimations** (`List[str]`): List of pose estimation files.
* **source_software** (`Literal["DeepLabCut", "SLEAP", "LightningPose"]`): Source software used for pose estimation.
* **working_directory** (`str, optional`): Working directory, by default &quot;.&quot;.
* **video_type** (`str, optional`): Video file type, by default &quot;.mp4&quot;.
* **fps** (`int, optional`): Sampling rate of the videos. If not passed, it will be estimated from the video file. By default None.
* **copy_videos** (`bool, optional`): Copy videos, by default False.
* **paths_to_pose_nwb_series_data** (`Optional[str], optional`): Path to pose NWB series data, by default None.
* **config_kwargs** (`Optional[dict], optional`): Additional configuration keyword arguments, by default None.

**Returns**

* `None`

#### get\_states

```python
def get_states(summary: bool = True) -> dict
```

Returns the pipeline states.

**Returns**

* `dict`: Pipeline states.

#### get\_sessions

```python
def get_sessions() -> List[str]
```

Returns a list of session names.

**Returns**

* `List[str]`: Session names.

#### get\_raw\_datasets

```python
def get_raw_datasets() -> xr.Dataset
```

Returns a xarray dataset which combines all the raw data from the project.

**Returns**

* **dss** (`xarray.Dataset`): Combined raw dataset.

#### preprocessing

```python
def preprocessing(centered_reference_keypoint: str = "snout",
                  orientation_reference_keypoint: str = "tailbase") -> None
```

Preprocesses the data.

**Parameters**

* **centered_reference_keypoint** (`str, optional`): Key point to center the data, by default &quot;snout&quot;.
* **orientation_reference_keypoint** (`str, optional`): Key point to orient the data, by default &quot;tailbase&quot;.

**Returns**

* `None`

#### create\_training\_set

```python
def create_training_set() -> None
```

Creates the training set.

**Returns**

* `None`

#### train\_model

```python
def train_model() -> None
```

Trains the model.

**Returns**

* `None`

#### evaluate\_model

```python
def evaluate_model() -> None
```

Evaluates the model.

**Returns**

* `None`

#### run\_segmentation

```python
def run_segmentation() -> None
```

Runs the pose estimation segmentation into motifs.

**Returns**

* `None`

#### run\_community\_clustering

```python
def run_community_clustering() -> None
```

Runs the community clustering.

**Returns**

* `None`

#### generate\_motif\_videos

```python
def generate_motif_videos(
        video_type: str = ".mp4",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm") -> None
```

Generates motif videos.

**Parameters**

* **video_type** (`str, optional`): Video type, by default &quot;.mp4&quot;.
* **segmentation_algorithm** (`Literal["hmm", "kmeans"], optional`): Segmentation algorithm, by default &quot;hmm&quot;.

**Returns**

* `None`

#### generate\_community\_videos

```python
def generate_community_videos(
        video_type: str = ".mp4",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm") -> None
```

Generates community videos.

**Parameters**

* **video_type** (`str, optional`): Video type, by default &quot;.mp4&quot;.
* **segmentation_algorithm** (`Literal["hmm", "kmeans"], optional`): Segmentation algorithm, by default &quot;hmm&quot;.

**Returns**

* `None`

#### generate\_videos

```python
def generate_videos(
        video_type: str = ".mp4",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm") -> None
```

Generates motif and community videos.

**Parameters**

* **video_type** (`str, optional`): Video type, by default &quot;.mp4&quot;.
* **segmentation_algorithm** (`Literal["hmm", "kmeans"], optional`): Segmentation algorithm, by default &quot;hmm&quot;.

**Returns**

* `None`

#### visualize\_preprocessing

```python
def visualize_preprocessing(scatter: bool = True,
                            timeseries: bool = True,
                            show_figure: bool = False,
                            save_to_file: bool = True) -> None
```

Visualizes the preprocessing results.

**Parameters**

* **scatter** (`bool, optional`): Visualize scatter plot, by default True.
* **timeseries** (`bool, optional`): Visualize timeseries plot, by default True.
* **show_figure** (`bool, optional`): Show the figure, by default False.
* **save_to_file** (`bool, optional`): Save the figure to file, by default True.

**Returns**

* `None`

#### visualize\_model\_losses

```python
def visualize_model_losses(save_to_file: bool = True,
                           show_figure: bool = True) -> None
```

Visualizes the model losses.

**Parameters**

* **save_to_file** (`bool, optional`): Save the figure to file, by default False.
* **show_figure** (`bool, optional`): Show the figure, by default True.

**Returns**

* `None`

#### visualize\_motif\_tree

```python
def visualize_motif_tree(
        segmentation_algorithm: Literal["hmm", "kmeans"]) -> None
```

Visualizes the motif tree.

**Parameters**

* **segmentation_algorithm** (`Literal["hmm", "kmeans"]`): Segmentation algorithm.

**Returns**

* `None`

#### visualize\_umap

```python
def visualize_umap(label: Literal["community", "motif"] = "community",
                   segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
                   show_figure: bool = False,
                   save_to_file: bool = True) -> None
```

Visualizes the UMAP plot.

**Parameters**

* **label** (`Literal["community", "motif"], optional`): Label to visualize, by default &quot;community&quot;.
* **segmentation_algorithm** (`Literal["hmm", "kmeans"], optional`): Segmentation algorithm, by default &quot;hmm&quot;.

**Returns**

* `None`

#### report

```python
def report(segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm") -> None
```

Generates the project report.

**Parameters**

* **segmentation_algorithm** (`Literal["hmm", "kmeans"], optional`): Segmentation algorithm, by default &quot;hmm&quot;.

**Returns**

* `None`

#### run\_pipeline

```python
def run_pipeline(from_step: int = 0, preprocessing_kwargs: dict = {}) -> None
```

Runs the pipeline.

**Parameters**

* **from_step** (`int, optional`): Start from step, by default 0.
* **preprocessing_kwargs** (`dict, optional`): Preprocessing keyword arguments, by default {}.

**Returns**

* `None`

#### unique\_in\_order

```python
def unique_in_order(sequence)
```

