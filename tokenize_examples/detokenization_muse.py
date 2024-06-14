import os
import numpy as np
import einops
import base64
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import json
import wandb
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from muse import VQGANModel
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import argparse


def save_temp_images(tensors, temp_dir='./temp_images'):
    """
    Save a batch of images stored in a PyTorch tensor to a temporary directory.

    Args:
    - tensors (torch.Tensor): A batch of images in the format (batch size, channels, height, width).
    - temp_dir (str): Path to the temporary directory where images will be saved.
    """
    # Create a temporary directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # grid_img = vutils.make_grid(tensors, n=16, padding=0)
    # vutils.save_image(grid_img, 'vqlm/muse/vis_reconstruction_tokens/concatenated_image.png')

    # Save each image in the batch to the temporary directory
    for i, tensor in enumerate(tensors):
        # Construct the file path
        file_path = os.path.join(temp_dir, f'image_{i}.png')
        # Save the image
        save_image(tensor, file_path)


def delete_temp_images(temp_dir='./temp_images'):
    """
    Delete the temporary directory and its contents.

    Args:
    - temp_dir (str): Path to the temporary directory to be deleted.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Assuming you want to split or transform the image in a way that fits the '2 * h * w' description
        # This can be an operation like doubling the image, or applying some transformation.
        # For simplicity, we'll just duplicate the image in the dataset.
        # Adjust the transformation as needed.

        return (image,)


import random

def select_random_elements(your_list, num_elements=64, seed_value=42):
    """
    Randomly selects `num_elements` from `your_list` using `seed_value` for reproducibility.

    Parameters:
    - your_list: The list to select elements from.
    - num_elements: The number of elements to select. Default is 64.
    - seed_value: The seed value for the random number generator to ensure reproducibility. Default is 42.

    Returns:
    - A list containing the randomly selected elements.
    """
    random.seed(seed_value)  # Set the seed for reproducibility
    selected_elements = random.sample(your_list, num_elements)  # Randomly select elements
    return selected_elements

def save_tokens(trecons_name, idx, re_constructed):
    tensor_img = re_constructed.cpu().numpy()
    images = (tensor_img * 255).transpose(0, 2, 3, 1).astype(np.uint8)

    # Concatenate images horizontally
    concatenated_image = np.hstack(images)

    # Convert the NumPy array to a PIL Image and save it
    os.makedirs(trecons_name, exist_ok=True)
    img = Image.fromarray(concatenated_image)
    img.save(os.path.join(trecons_name, '{}.png'.format(idx)))


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add arguments
    parser.add_argument('--dataset', default='i1k_edge', type=str, help='An input name')

    # Parse the arguments
    args = parser.parse_args()

    # Load the pre-trained vq model from the hub
    net = VQGANModel.from_pretrained("vqlm/muse/ckpts/laion").cuda()
    net.eval()

    idx = 1
    dataset = args.dataset
    with open('./lvm/tokenized_muse/{}.jsonl'.format(dataset), 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())['tokens']
                # Process the JSON object
            except:
                continue

            decoded_bytes = base64.b64decode(json_obj )
            array_dtype = np.int32
            array_shape = (-1, 256)
            tokens = np.frombuffer(decoded_bytes, dtype=array_dtype).reshape(array_shape)
            tokens = torch.tensor(tokens).cuda()


            with torch.no_grad():

                # detokenized
                re_constructed = net.decode_code(tokens)

                plt.figure(figsize=(12, 12))
                for i in range(re_constructed.shape[0]):
                    recon_img = torch.clamp(re_constructed[i],
                        0.0, 1.0
                    )
                    plt.subplot(4, 4, i + 1)
                    plt.imshow((((recon_img).permute(1, 2, 0).detach().cpu().numpy() * 255)).astype(np.int32))
                    plt.grid(False)
                    plt.axis('off')
                save_root = './lvm/other_folder/vis_reconstruction_tokens_check_final/{}'.format(dataset)
                os.makedirs(save_root, exist_ok=True)
                plt.savefig('{}/{}.png'.format(save_root, idx))

                idx += 1
                if idx >= 16:
                    break

