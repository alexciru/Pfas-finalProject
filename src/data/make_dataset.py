import fiftyone.zoo as foz

# This script downloads the COCO dataset and stores it in the data/external directory
# Can be specify the number of total images to download
# and the classes to download in this case we want [person", "car", "bicycle"]
# 
# We donnt need validation set because we can use the sequence

print("starting creating dataset")

# To download the COCO dataset for only the "person" and "car" classes
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person", "car", "bicycle"],
    max_samples=1000,
    dataset_name="my-coco-test",
    dataset_dir="data/external/coco-2017",
)

print(dataset)
# store dataset in disk 