import argparse
import torch
from pathlib import Path
from utility.utils import ImageFolderDataset, get_transform
from torch.utils.data import DataLoader
from utility.models import *
import torch.optim as optim

def parse_arguments():
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--content_dir", type=str, default=r"C:\Users\jiyat\Desktop\Projects\NST Project\content_data", help="Location of  content dataset")
  
  parser.add_argument("--style_dir", type=str, default=r"C:\Users\jiyat\Desktop\Projects\NST Project\style_data", help="Location of style dataset")
  
  parser.add_argument("--vgg", type=str, default=r"C:\Users\jiyat\Desktop\Projects\NST Project\utility\vgg_normalised.pth", help="Location of pre-trained VGG")  # giving path bcoz using pretrained model.
  
  parser.add_argument("--experiment", type=str, default="experiment1")  #everytime an experiment runs here it will get saved. 
  
  parser.add_argument("--final_size", type=int, default=256, help="Size of final image")
  parser.add_argument("--content_size", type=int, default=512, help="Size of content image")
  parser.add_argument("--style_size", type=int, default=512, help="Size of style image")
  parser.add_argument("--crop", action="store_true", help="Crop Image", default=True)
  parser.add_argument("--batch_size", type=int, help="batch Size", default=4)
  parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
  parser.add_argument("--lr_decay", type=float, default=5e-5, help="Learning rate decay")
  
  return parser.parse_args()
  
def main():
  args = parse_arguments()
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  save_dir = Path("experiment") / args.experiment
  save_dir.mkdir(exist_ok=True, parents=True)
  
  with open(save_dir / 'args.txt', 'w') as args_file:
    for key, value in vars(args).items():
      args_file.write(f'{key} : {value}\n')  
      
  content_transform = get_transform(args.content_size, args.crop, args.final_size)
  style_transform = get_transform(args.style_size, args.crop, args.final_size)
      
  content_dataset = ImageFolderDataset(args.content_dir, content_transform)
  style_dataset = ImageFolderDataset(args.style_dir, style_transform)
  
  content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
  style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
  
  print("Number of batches in content dataloader", len(content_dataloader))
  print("Number of batches in style dataloader", len(style_dataloader))
  
  encoder = VGGEncoder(args.vgg).to(device)
  decoder = Decoder().to(device)
  
  
  optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
  schedular = optim.lr_scheduler.LambdaLR(
    optimizer = optimizer,
    lr_lambda = lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
  )
  
  print("training...")
  
  
  
      
  
  

if __name__ == "__main__":
  main()