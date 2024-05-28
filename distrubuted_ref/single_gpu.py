import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
    


class Trainer:
    def __init__(self, 
                    model: torch.nn.Module,
                    train_data: DataLoader,
                    optimizer,
                    gpu_id: int,
                    save_every: int ) -> None:
            
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        checkpoint = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(checkpoint, PATH)
        print(f"Epoch {epoch} | Training snapshot saved at {PATH}")

    
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    train_data = DataLoader(CustomDataset(100), batch_size=10, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, train_data, optimizer


def main(save_every: int, total_epochs: int):
    model, train_data, optimizer = load_train_objs()
    trainer = Trainer(model, train_data, optimizer, 0, save_every)
    trainer.train(total_epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)


# Run the script

# !python single_gpu.py 10 2 --batch_size 10
    

