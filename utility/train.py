import argparse
import torch
from pathlib import Path

def parse_arguments():
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--content_dir", type=str, default=r"C:\Users\jiyat\Desktop\Projects\NST Project\content_data", help="Location of  content dataset")
  
  parser.add_argument("--style_dir", type=str, default=r"C:\Users\jiyat\Desktop\Projects\NST Project\style_data", help="Location of style dataset")
  
  parser.add_argument("--vgg", type=str, default=r"C:\Users\jiyat\Desktop\Projects\NST Project\vgg_normalised.pth", help="Location of pre-trained VGG")  # giving path bcoz using pretrained model.
  
  parser.add_argument("--experiment", type=str, default="experiment1")  #everytime an experiment runs here it will get saved. 
  
  return parser.parse_args()
  
def main():
  args = parse_arguments()
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  save_dir = Path("experiment") / args.experiment
  save_dir.mkdir(exist_ok=True, parents=True)
  
  with open(save_dir / 'args.txt', 'w') as args_file:
    for key, value in vars(args).items():
      args_file.write(f'{key} : {value}\n')  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

if __name__ == "__main__":
  main()