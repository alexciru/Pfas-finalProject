
## REQUIREMENTS (clone deep_sort and get pre-generated detections)
* git clone https://github.com/nwojke/deep_sort.git from inside notebooks/tracking directory
* download pre-generated detections and networks dirs from [here](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp), again to the notebooks/tracking directory
* unzip them

If something doesn't work, see instructions [here](https://github.com/nwojke/deep_sort) for deepsort

* run `make setup deepsort` from notebooks/tracking directory

## Depencies
* should be in requirements_mac.txt and requirements_win.txt in root directory (pfas-finalProject)
* if something is broken, see REQUIRED CODE CHANGES and also notebooks/tracking/IGNORE_requirements_mac.txt which is just added for sanity purposes
and is last known working version of requirements (on macOS)

## REQUIRED CODE CHANGES IN DEEP_SORT LIBRARY!!
Sorry, I know this is very annoying, but code does not work otherwise.

in deep_sort/deep_sort/linear_assignment.py:
* replace `from sklearn.utils.linear_assignment_ import linear_assignment` 
with `from scipy.optimize import linear_sum_assignment as linear_assignment`
* under the line (58) `indices = linear_assignment(cost_matrix)` add (has to do with this [issue](https://github.com/yehengchen/Object-Detection-and-Tracking/issues/92)):
    ```
    row, col = indices
    indices = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1)
    ```

in deep_sort/tools/generate_detections.py:
* replace `import tensorflow as tf` with `import tensorflow.compat.v1 as tf`
* add `tf.disable_v2_behavior()` right after it
