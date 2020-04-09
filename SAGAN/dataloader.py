import torch
import torchvision.datasets as dsets
from torchvision import transforms

class Data_Loader():
    def __init__(self, image_path, image_size, batch_size, shuf=True):
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.trans = transforms.Compose(
            [
                transforms.Resize((self.imsize, self.imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

            ]
        )
    
    def load_celeb(self):
        dataset = dsets.ImageFolder(self.path, transform=self.trans)
        return dataset

    def loader(self):
        dataset = self.load_celeb()

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=self.shuf,
            num_workers=0,
            drop_last=True
        )

        return loader
