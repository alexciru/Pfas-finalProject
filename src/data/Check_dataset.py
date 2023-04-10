from pycocotools.coco import COCO
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import click
# load local anotation file with realtive path
# get local path


def main():
    categories = ['person', 'car', 'bicycle']


    dataDir= r".\data\external\myCoco"
    dataType= r"val2017"
    annFile= "{}\\raw\\instances_{}.json".format(dataDir,dataType)
    annFile= "{}\\train\\labels.json".format(dataDir)
    coco=COCO(annFile)


    # Get list of category_ids, here [2] for bicycle
    category_ids = coco.getCatIds(categories)
    print(category_ids)

    # Get list of image_ids which contain some/all of the category_ids
    image_ids = coco.getImgIds(catIds=category_ids)
    image_id = random.choice(image_ids)

    print("Image Id: ", image_id)
    # get the anotation from the image
    annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=[1])


    # get bounding boxes
    anns = coco.loadAnns(annotation_ids)


    # Get list of image_ids which contain bicycles
    #image_ids = coco.getImgIds(catIds=[1])
    #print(len(image_ids))
    # Get annotation based on the category and imag

    annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=[1, 2, 3])
    print("anotations")
    print(annotation_ids)
    anns = coco.loadAnns(annotation_ids)

    # load img
    images_path = "./data/external/myCoco/train/data/"
    image_name = str(image_id).zfill(12)+".jpg" # Image names are 12 characters long
    image = Image.open(images_path+image_name)
    
    fig, ax = plt.subplots()
    
    # Draw boxes and add label to each box
    for ann in anns:
        box = ann['bbox']
        bb = patches.Rectangle((box[0],box[1]), box[2],box[3], linewidth=1, edgecolor="blue", facecolor="none")
        ax.add_patch(bb)
    
    # show
    ax.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
