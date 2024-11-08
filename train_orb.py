from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import copy
import os
from datetime import datetime
from dataclasses import dataclass

Categories = ["Bus", "motorcycle", "Truck", "Car"]
ImageSize = (128, 64)
GridOptimize = False


@dataclass
class TrainingArgs:
    device: torch.device
    input_size: int = 3 * ImageSize[0] * ImageSize[1]
    c: float = 0.001
    lr: float = 0.1
    batch_size: int = 32
    num_epochs: int = 25
    num_classes: int = 4
    datadir: str = "dataset/"
    nbins: int = 9
    pool: int = 8


class MotorVehiclesDataset(torch.utils.data.Dataset):
    """Motor Vehicles dataset."""

    def __init__(self, root_dir, grayscale=False, image_transform: nn.Module | None = None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.grayscale = grayscale
        self.dataset = []
        for i in Categories:
            # print(f"loading... category : {i}")
            path = os.path.join(root_dir, i)
            for img in os.listdir(path):
                if not img.startswith("."):
                    self.dataset.append(
                        (os.path.join(root_dir, i, img), Categories.index(i))
                    )
            # print(f"loaded category:{i} successfully")
        print("Loaded dataset successfully")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_array = Image.open(self.dataset[idx][0])  # type: ignore
        if self.image_transform:
            img_array = img_array.convert("L").resize(ImageSize)
            to_tensor = transforms.ToTensor()
            img_array = torch.flatten(to_tensor(img_array))
        else:
            img_array = img_array.convert("RGB").resize(ImageSize)
            to_tensor = transforms.ToTensor()
            img_array = torch.flatten(to_tensor(img_array))

        sample = (img_array, self.dataset[idx][1])  # type: ignore

        return sample


class SVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()  # Call the init function of nn.Module
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


class FAST(nn.Module):
    def __init__(self, n=12, ):
        super(FAST, self).__init__()  # Call the init function of nn.Module

    def forward(self, x):
        kernel = torch.tensor([[1,2,1], [2,4,2],[1,2,1]])/ 16
        img = torch.conv2d(x, kernel)

        

        return out

def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, other_args
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(other_args.num_epochs):

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for i, (images, labels) in enumerate(dataloaders[phase]):
                # Reshape images to (batch_size, input_size) and then move to device
                images = images.to(other_args.device)
                labels = labels.to(other_args.device)

                # Forward pass - track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)  # data loss

                    # Add regularization i.e.  Full loss = data loss + regularization loss
                    weight = model.fc.weight.squeeze()
                    loss += other_args.c * torch.sum(torch.abs(weight))
                    # backward + optimize only if in training phase
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Collect statistics
                running_loss += loss.item() * images.size(
                    0
                )  # images.size(0) is batch size.
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double(  # type:ignore
            ) / dataset_sizes[phase]

            print(
                "Epoch [{}/{}], {} Loss: {:.4f} Acc: {:.4f}".format(
                    epoch + 1,
                    other_args.num_epochs,
                    phase,
                    epoch_loss,
                    epoch_acc * 100.0,
                )
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print("Best val Acc in percentage: {:.4f}".format(best_acc * 100.0))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return (model, best_acc)


def main():
    """
    Main function to run the linear SVM using hinge loss or logistic regression using softmax loss (cross-entropy loss)
    implemented using PyTorch.
    """
    args = TrainingArgs(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(args)

    args.input_size = int(
        args.nbins * ImageSize[0] / args.pool * ImageSize[1] / args.pool
    )

    dataset = MotorVehiclesDataset(args.datadir, grayscale=True)

    generator1 = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [320, 80], generator=generator1
    )

    dataset_sizes = dict()
    dataset_sizes["train"] = len(train_dataset)
    dataset_sizes["val"] = len(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=True
    )

    dataloaders = dict()
    dataloaders["train"] = train_loader
    dataloaders["val"] = val_loader

    model = SVM(args.input_size, args.num_classes)
    model.to(args.device)

    criterion = nn.MultiMarginLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if GridOptimize:
        print("Training with grid optimization")
        best_model_state = {}
        best_acc = 0

        for i in range(20):
            args.c = 2 ** (-12)
            args.lr = 2 ** (-8 + 0.25 * i)
            (model, last_acc) = train_model(
                model,
                dataloaders,
                dataset_sizes,
                criterion,
                optimizer,
                exp_lr_scheduler,
                args,
            )
            if last_acc > best_acc:
                best_acc = last_acc
                best_model_state = model.state_dict()

        torch.save(best_model_state, f"model/{datetime.now()}model.pth")

        print("OVERALL best val Acc in percentage: {:.4f}".format(
            best_acc * 100.0))
    else:
        (model, last_acc) = train_model(
            model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer,
            exp_lr_scheduler,
            args,
        )

        torch.save(model.state_dict(), f"model/{datetime.now()}-model.pth")


# Execute from the interpreter
if __name__ == "__main__":
    main()
