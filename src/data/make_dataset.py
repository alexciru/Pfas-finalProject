import fiftyone.zoo as foz
import click
# This script downloads the COCO dataset and stores it in the data/external directory
# Can be specify the number of total images to download
# and the classes to download in this case we want [person", "car", "bicycle"]
# 
# We donnt need validation set because we can use the sequence

@click.command()
@click.argument("output_dir",default="data/external/coco-2017", type=click.Path())
@click.argument("samples", default=1000, help="number of samples to download")
def main(output_dir: str, samples: int):
    print("starting creating dataset")

    # To download the COCO dataset for only the "person" and "car" classes
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=["person", "car", "bicycle"],
        max_samples= samples,
        dataset_name="my-coco-test",
        dataset_dir= output_dir,
    )

    print(dataset)
    # store dataset in disk 


if __name__ == "__main__":
    main()