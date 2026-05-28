from torch.utils.data import Dataset
import os
from PIL import Image

class ImageFolderDataset(Dataset):
  def __init__(self, root, transform=None):
    super(ImageFolderDataset).__init__()
    
    self.root = root
    self.transform = transform
    self.files = list(os.listdir(root))
    self.files = [file for file in self.files if file.endswith(".png", ".jpg", ".jpeg")]
    
  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, index):
    image_path = os.path.join(self.root, self.files[index])
    image = Image.open(image_path)
    
    if self.transform:
      image = self.transform(image)
    
    return image
    
    