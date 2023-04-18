Note: object_detector.py should be made a Notebook if possible but I ran into dependency issues with it.

## Usage
Use Makefile to run things if you want

### Training
`make train` which uses detector_train.py

### Running on our data
`make run` which uses detector.py
Currently data path is hardcoded in detector.py, so will need to manually update.

## Requirements
python==3.10.9

Currently for mac (see requirements.txt). For non-macs, remove "macos" in tensorflow-mac and delete tensorflow-metal.

## Docs
Yolov8: https://docs.ultralytics.com/

## Mac M1/M2 setup
* have to get tensorflow-macos set up by following: https://developer.apple.com/metal/tensorflow-plugin/
* really make sure to run the step `source ~/miniconda/bin/activate` and keep it activated
before making virtual env
* make sure it's the right architecture (arm64) by doing:
```
>>> import platform
>>> platform.platform()
'macOS-12.6.3-arm64-arm-64bit'
```
source: https://stackoverflow.com/questions/70562033/tensorflow-deps-packagesnotfounderror
* then make your virtual env, i.e. python3 -m venv myvenv
* then, make sure to install right versions of tensorflow-macos and tensorflow-metal:
tensorflow-macos==2.9 and tensorflow-metal==0.5.0
unless they fixed the issues with earlier versions, see details on issue here: https://developer.apple.com/forums/thread/721619

* if you run into numpy error, probs just need to ugprade it with pip https://github.com/freqtrade/freqtrade/issues/4281

* ignore the warnings about NUMA

* run python testtf.py per apple docs in tensorflow-macos setup to get output:
```
782/782 [==============================] - 87s 105ms/step - loss: 5.1127 - accuracy: 0.0347
Epoch 2/5
782/782 [==============================] - 80s 102ms/step - loss: 4.5293 - accuracy: 0.0511
Epoch 3/5
782/782 [==============================] - 80s 103ms/step - loss: 4.4699 - accuracy: 0.0581
Epoch 4/5
782/782 [==============================] - 84s 107ms/step - loss: 4.1426 - accuracy: 0.0762
Epoch 5/5
782/782 [==============================] - 85s 108ms/step - loss: 3.9236 - accuracy: 0.1092
```

* `pip install cmake`
* `pip install horovod`