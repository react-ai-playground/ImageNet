import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
# pip install torch torchvision pillow pathlib 
from outputs import outputs

# For 0-999, define indeces as keys and outputs as values
labels = {i:j for j,i in enumerate(outputs)}
names = list(outputs.values())

# Transformations
TRANSFORMS_COMMON = [
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
TRANSFORMS_TRAIN = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    *TRANSFORMS_COMMON])
TRANSFORMS_TEST = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    *TRANSFORMS_COMMON])


# Define the dataset -> Currently using training set
class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        assert split in ('train', 'val', 'test'), f"Invalid split `{split}`"
        assert (Path(root_dir) / split).is_dir(), f"Data for {split} set does not exist yet, run `download.py` to fetch it."
        self.file_paths = []
        self.labels = []
        self.transform = TRANSFORMS_TRAIN if split == 'train' else TRANSFORMS_TEST

        # Loop through root directory of images
        for file in (Path(root_dir) / split).iterdir():
            if file.name.lower().endswith(".jpeg"):
                # Store paths and labels at same index, where each label = text assignment
                self.file_paths.append(str(file))
                file_id = file.name.split('_')[-1].split('.')[0]
                self.labels.append(-1 if split == 'test' else labels[file_id])

        print(f"Initialized {split} set with {len(self.labels)} samples")

    # Return length of file_paths
    def __len__(self):
        return len(self.file_paths)
    
    # Return an image object and label 
    def __getitem__(self, idx):
        img_name = self.file_paths[idx]
        image = Image.open(img_name).convert('RGB') # Open the image and convert to RGB mode using PIL
        image = self.transform(image) # Transform the image 
        label = self.labels[idx]
        return image, label
    

if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader

    RED = '\u001b[31m'
    GREEN = "\u001b[32m"
    RESET = "\033[m"

    print("Initializing datasets...")
    
    # Set the root directory for the data -> in this case data is not local, it's on a USB
    # data_dir = Path(__file__).parent / 'data'
    data_dir  = "/Volumes/Training Data/data"
   
    # train_dataset = ImageNetDataset(data_dir, split='train')
    val_dataset = ImageNetDataset(data_dir, split='val')
    test_dataset = ImageNetDataset(data_dir, split='test')

    # Benchmark -> Display progress bar to see time it takes to iterate over the dataset 200 times
    print("Benchmarking...")
    from tqdm import trange  #trange is just like range, but with a progress bar
    for i in trange(200):
        val_dataset[i]

    # Test the data pipeline 
    print("Testing the data pipeline with pre-trained model...")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
    loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    # Data loader returns tuple of arrays of length batch size, one of images, and one of labels 
    images, labels = next(iter(loader))
    results = model(images)
    predicted = torch.argmax(results, dim=1) # Take max value in output vector as guess 

    for gt, pred in zip(labels, predicted):
        print(f'True / Pred: {GREEN}{names[gt]}{RESET} / {GREEN if pred == gt else RED}{names[pred]}{RESET}')
