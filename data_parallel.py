from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch import Tensor
from typing import Iterator, Tuple
import torchmetrics
from torch.nn.parallel import DataParallel
from single_gpu import TrainerSingle
def prepare_const() -> dict:
    """Data and model directory + Training hyperparameters"""
    data_root = Path("data")
    trained_models = Path("trained_models")

    if not data_root.exists():
        data_root.mkdir()

    if not trained_models.exists():
        trained_models.mkdir()

    const = dict(
        data_root=data_root,
        trained_models=trained_models,
        total_epochs=15,
        batch_size=128,
        lr=0.1,  # learning rate
        momentum=0.9,
        lr_step_size=5,
        save_every=3,
    )

    return const


def cifar_model() -> nn.Module:
    model = resnet34(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def cifar_dataset(data_root: Path) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768),
            ),
        ]
    )

    trainset = CIFAR10(root=data_root, train=True, transform=transform, download=True)
    testset = CIFAR10(root=data_root, train=False, transform=transform, download=True)

    return trainset, testset


def cifar_dataloader_single(
    trainset: Dataset, testset: Dataset, bs: int
) -> Tuple[DataLoader, DataLoader]:
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)

    return trainloader, testloader

class TrainerDP(TrainerSingle):
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
    ):
        self.gpu_id = "cuda"
        super().__init__(self.gpu_id, model, trainloader, testloader) 
        
        self.model = DataParallel(self.model)     
    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        model_path = self.const["trained_models"] / f"CIFAR10_dp_epoch{epoch}.pt"
        torch.save(ckp, model_path)


def main_dp(final_model_path: str):
    const = prepare_const()
    train_dataset, test_dataset = cifar_dataset(const["data_root"])
    train_dataloader, test_dataloader = cifar_dataloader_single(
        train_dataset, test_dataset, const["batch_size"]
    )
    model = cifar_model()
    trainer = TrainerDP(
        model=model,
        trainloader=train_dataloader,
        testloader=test_dataloader,
    )
    trainer.train(const["total_epochs"])
    trainer.test(final_model_path)


if __name__ == "__main__":
    final_model_path = Path("./trained_models/CIFAR10_dp_epoch14.pt")
    main_dp(final_model_path)


###
###
"""
run the following command to train the model on multiple GPUs:
!nvidia-smi

$ CUDA_VISIBLE_DEVICES=6,7 python main.py 

"""