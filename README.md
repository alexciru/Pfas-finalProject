Pfas-finalProject
==============================

final project of pfas
------------
Install dependecies:  
`pip install requirements.txt` or `pip install requirements_mac.txt` for macOS


Download Data from DVC:  

`dvc pull data` - Will download all the data (both video seq and dataset)
if wanted to download only the video sequence execute    
`dvc pull data/video`
Might require log in to the Google Account  

In order to upload new data files into the cloud execute commands:
`dvc add data/`
`dvc push`

this will update the .dvc file REMEMBER TO PUSH THE NEW .DVC FILE INTO GIT AFTER PUSIING DATA !!!

-------
## Dataset   
The data used for the project is a subset of the (COCO dataset)[https://cocodataset.org/#home].
The subset was created using only person, car and bicycle labels.  
To generate the dataset you can use src/data/make_dataset.py. 
The number of samples , labels and output path can be selected as arguments.

If created a new dataset remember to uploaded to DVC

-------

## Model training  
We used the subset to train a model. The data need to be converted to TFRecord format. 

-------

## Video Sequence:   
The Sequences are not videos but a set of frames .
1. Seq_1: Simple seq with ground truth and no oculusion
2. Seq_2: Seq with oclusion and ground truth
3. Seq_3: Seq with oclusion no ground truth

<details open>
<summary>Details</summary>
In the sequence folder you will find:

- Three recorded sequences of pedestrian, cyclists and cars.
- A sequence for camera calibration.

## Data format

Each sequence folder contains the following structure:
```
seq_<n>/
|
|____image02/
|    |____data/
|         |_ <image_seq_no>.png
|    |_ timestamps.txt
|
|
|____image03/
|    |____data/
|         |_ <image_seq_no>.png
|    |_ timestamps.txt
```

- image_02/data/ contains the left color camera sequence images (png)
- image_02/timestamps.txt timestamps for each left image, the first line being for the first image, the second line for the second, and so on.
- image_03/data/ contains the right color camera sequence images  (png)
- image_03/timestamps.txt timestamps for each right image
- labels.txt contains the ground truth labels that can be used to evaluate your solution. **Note that we do not provide the labels for sequence 03.**

The labels files contain the following information. All values (numerical or strings) are separated via spaces, each row corresponds to one object. The 17 columns represent:

|Values  |  Name    |  Description
|--------|----------|--------------------------------------------------------
|   1    | `frame`    |  Frame within the sequence where the object appearers
|   1    | `track id` |  Unique tracking id of this object within this sequence
|   1    | `type`     |  Describes the type of object: `Car`,`Pedestrian`, `Cyclist`
|   1    | `truncated`|  Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries.
|   1    | `occluded` | Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
|   1    | `alpha`    | Observation angle of object, ranging $[-\pi;\pi]$
|   **4**    | `bbox`     | 2D bounding box of object in the **Rectified image** (0-based index): contains left, top, right, bottom pixel coordinates
|   **3**   |`dimensions`| 3D object dimensions: height, width, length (in meters)
|  **3**    | `location` | 3D object location $x,y,z$ in camera coordinates (in meters)
|   1    |`rotation_y`| Rotation $r_y$ around Y-axis in camera coordinates $[-\pi;\pi]$
|   1    | `score`   | Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.

**Please note that the 2D bounding box provided is given with respect to rectified image coordinates.** 

</details>

------------  
## Goals:

- Calibrate and rectify the stereo input.
- Process the input images to detect objects in the environment and track them in 3D,
even under occlusions.
- Train a machine learning system that can classify unseen images into the 3 classes
(pedestrians, cyclists, and cars) based either on 2D or 3D data.
o Use the web or/and capture your own images to create your training set. The
image sequences 1 and 2 provided with the project will constitute your
validation set and the sequence 3 your testing set
Project Organization  
------------
File structure

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
