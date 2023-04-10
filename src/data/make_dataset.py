import fiftyone.zoo as foz
import fiftyone as fo
import click

import json
# This script downloads the COCO dataset and stores it in the data/external directory
# Can be specify the number of total images to download
# and the classes to download in this case we want [person", "car", "bicycle"]
# 
# We donnt need validation set because we can use the sequence

#@click.command()
#@click.argument("output_dir",default="data/external/coco-2017", type=click.Path())
#@click.argument("samples", default=1000, help="number of samples to download")
def main():
    print("starting creating dataset")

    # To download the COCO dataset for only the "person" and "car" classes
    # get only bounding boxes
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        classes=["person", "car", "bicycle"],
        split="train",
        label_types= "detections",
        max_samples= 1500,
        dataset_name="my-coco-test",
        dataset_dir= "data/external/myCoco"
    )



if __name__ == "__main__":
    main()