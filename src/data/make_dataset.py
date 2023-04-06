import fiftyone.zoo as foz

# To download the COCO dataset for only the "person" and "car" classes
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections", "segmentations"],
    classes=["person", "car", "bicycle"],
    max_samples=50,
    dataset_name="my-coco-dataset",
    dataset_dir="/data/external/coco-2017",
)

print(dataset)
# store dataset in disk 