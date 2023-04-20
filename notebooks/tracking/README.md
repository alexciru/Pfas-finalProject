
## SETUP
* git clone https://github.com/nwojke/deep_sort.git from inside notebooks/tracking directory
* download pre-generated detections and networks dirs from [here](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp), again to the notebooks/tracking directory
* unzip them
* if something doesn't work, see instructions [here](https://github.com/nwojke/deep_sort) for deepsort
* run `make setup-deepsort` from notebooks/tracking directory (replaces some files with our custom ones, needed to make code run)

## Dependencies
* should be in requirements_mac.txt and requirements.txt in root directory (pfas-finalProject)
* if something is broken, see REQUIRED CODE CHANGES and also notebooks/tracking/IGNORE_requirements_mac.txt which is just added for sanity purposes
and is last known working version of requirements (on macOS)
