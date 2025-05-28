---
sidebar_label: cleaning
title: preprocessing.cleaning
---

#### logger\_config

#### logger

#### lowconf\_cleaning

```python
def lowconf_cleaning(config: dict,
                     read_from_variable: str = "position_egocentric_aligned",
                     save_to_variable: str = "position_processed") -> None
```

Clean the low-confidence data points from the dataset. Processes position data by:
 - setting low-confidence points to NaN
 - interpolating NaN points

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **read_from_variable** (`str, optional`): Variable to read from the dataset.
* **save_to_variable** (`str, optional`): Variable to save the cleaned data to.

**Returns**

* `None`

#### outlier\_cleaning

```python
def outlier_cleaning(config: dict,
                     read_from_variable: str = "position_processed",
                     save_to_variable: str = "position_processed") -> None
```

Clean the outliers from the dataset. Processes position data by:
 - setting outlier points to NaN
 - interpolating NaN points

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **read_from_variable** (`str, optional`): Variable to read from the dataset.
* **save_to_variable** (`str, optional`): Variable to save the cleaned data to.

**Returns**

* `None`

