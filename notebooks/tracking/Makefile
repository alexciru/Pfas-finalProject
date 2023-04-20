ROOT:=$(git rev-parse --git-dir)

# TODO: add script to clone deep_sort, get detections and network weights from google drive + unzip them
setup-deepsort:
	rm -rf deep_sort/deep_sort/linear_assignment.py deep_sort/tools/generate_detections.py
	cp custom_deep_sort/deep_sort/linear_assignment.py deep_sort/deep_sort/
	cp custom_deep_sort/tools/generate_detections.py deep_sort/tools

track:
	cd $(ROOT) && python3 notebooks/tracking/tracker.py
