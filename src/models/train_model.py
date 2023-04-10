
#%%
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision
from CocoDataset import CocoDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import yaml


#Not working yet


def main():
    # load yolov5s
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)

    # model = get_model_instance_segmentation(3)
    
    print("Loading dataset...")
    dataset = CocoDataset(root = "data/external/myCoco/train/data/",
                        annotation = "data/external/myCoco/train/labels.json",
                        transforms = transforms.ToTensor())
 

  
    classes = [ 'bicyle',
                'person',
                'car']

    data = dict(
        train =  'data/external/Yolov5/train/',
        val   =  'data/external/Yolov5/valid/',
        nc    =  3,
        names = classes
        )

    with open('./yolov5/vinbigdata.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        
    f = open('./yolov5/data.yaml', 'r')
    print('\nyaml:')
    print(f.read())



    # data loader
    batch_size = 1
    num_workers = 1

    trainloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=False
                                            )

    # test forward pass
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    
    n_epochs = 10
    batch_size = 1
    learning_rate = 0.001

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    steps = 0
    losses = []
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        for images, annotation in trainloader:
            print(annotation.keys())

            #img = transforms.ToPILImage()(images[0])
            optimizer.zero_grad()

            output = model(images, annotation)

            loss = criterion(output, annotation)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        else:
            losses.append(running_loss/len(trainloader))
            steps += 1
            print(f"Training loss: {running_loss/len(trainloader)}")

    steps = [i for i in range(steps)]

    # Use the plot function to draw a line plot
    plt.plot(steps, losses)

    # Add a title and axis labels
    plt.title("Training Loss vs Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")

    # Save the plot
    plt.savefig("reports/figures/lossV1.png")

    torch.save(model.state_dict(), 'models/trained_modelV1.pt')



def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



if __name__ == '__main__':
    main()