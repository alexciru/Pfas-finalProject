Note: object_detector.py should be made a Notebook if possible but I ran into dependency issues with it.

## Usage
Use Makefile to run things if you want

### Training
`make train` which uses detector_train.py

### Running on our data
`make run` which uses detector.py
Currently data path is hardcoded in detector.py, so will need to manually update.

## Requirements
Currently for mac (see requirements.txt). For non-macs, remove "macos" in tensorflow-mac and delete tensorflow-metal.