import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import glob

def generate_random_palette():
    # Generate a random palette with RGB values
    return np.array([np.random.choice(range(256), size=3) for _ in range(256)], dtype=np.uint8)

# Function to apply the color mapping to a grayscale image
def apply_color_palette(image_path, output_path, palette):
    # Load the image and convert it to a NumPy array
    grayscale_image = Image.open(image_path).convert('L')
    grayscale_array = np.array(grayscale_image, dtype=np.int32)

    # Apply the palette using advanced indexing
    color_mapped_array = palette[grayscale_array]

    # Convert back to an image and save or display
    color_mapped_image = Image.fromarray(color_mapped_array, 'RGB')
    color_mapped_image.save(output_path)


if __name__ == '__main__':
    # Example usage
    # Generate a random color palette
    # palette = generate_random_palette()
    # np.save('./color_map.npy', palette)

    palette = np.load('./vqlm/muse/organize_dataset/i1k/color_map.npy')

    s = 1

    root = '/scratch/partial_datasets/lvm/dataset/prismer_i1k/new/seg_coco/train'

    images = glob.glob(root + '/*/*.png')

    for img in tqdm(images):
        cate_path = img.split('/')[-2]
        save_root = os.path.join('/scratch/partial_datasets/lvm/dataset/prismer_i1k/new_mapped_color/seg_coco_colored/train', cate_path)
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, os.path.basename(img))
        apply_color_palette(img, save_path, palette=palette)
