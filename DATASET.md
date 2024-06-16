
# Dataset Preparation

This section describes how to prepare your dataset by tokenizing visual data into visual sentences and then mixing and shuffling the datasets.

## Download & Prepare Dataset
We have individual scripts in `./tokenize_examples` to handle different kinds of visual sentences mentioned in the paper. Simple descriptions and instructions are listed as follows.

### Pair Datasets
For pair datasets, where the visual sentence is constructed as `[image, label, image, label, image, label...]`, we use `tokenize_examples/tokenize_paired_dataset_muse.py` to generate the visual tokens. For example, for depth maps, surface normals, edges, and segmentation, we first use [prismer](https://github.com/NVlabs/prismer) to generate pseudo labels, then use the script to generate visual sentences. Note that for segmentation, we use an additional color mapping after obtaining the pseudo labels from prismer, which can be done by running `tokenize_examples/map_color.py`.

### Video Datasets
For video datasets, the visual sentences are constructed as `[frame1, frame2, frame3, ... framex]`. One can use `tokenize_examples/tokenize_video_muse.py` to generate the visual sentences. The hyperparameter `stride` can be used to control the sampling rate of the extraction of frames from a video.

### Colorization Datasets
For colorization datasets, the visual sentences are constructed as `[gray_image, colored_image, gray_image, colored_image, ...]`. One can use `tokenize_examples/tokenize_colorization_dataset_muse.py` to generate the visual sentences. The user only needs to prepare the colored images, and the script will generate the gray counterparts.

### Inpainting Datasets
For inpainting datasets, the visual sentences are constructed as `[masked_image, image, masked_image, image, ...]`. One can use `tokenize_examples/tokenize_inpainting_dataset_muse.py` to generate the visual sentences. The user can control the masked ratio by changing `FLAGS.hole_mask_ratio`.

### Multi-Datasets
For multi-datasets, the visual sentences are constructed as `[dataset1_image, dataset1_image, dataset2_image, dataset2_image, ...]`. One can use `tokenize_examples/tokenize_multi_datasets_muse.py` to generate the visual sentences.

### Category Datasets
For category datasets, the visual sentences are constructed as `[cate1_image, cate1_image, cate2_image, cate2_image,...]`. One can use `tokenize_examples/tokenize_category_images_muse.py` to generate the visual sentences. Note that the user can use `images_per_shot` to customize the number of images for each category and `n_shots` to control the number of categories in the visual sentences.

In general, visual sentences with different logic can be achieved by combining the basic logic provided above. After you generate your visual sentences, you can always use `tokenize_examples/detokenization_muse.py` to perform a sanity check for the recovery of the visual sentences.


## Visual Sentences

- Following the concept of Visual Sentences, each image is tokenized separately, and the tokens are concatenated to form a visual sentence.

## Generating JSONL Files

- For each dataset, generate the `dataset*.jsonl` files.

### Example Code

- You can find example code for tokenizing images in the `./tokenize_examples` directory.

## Mixing and Shuffling Datasets

- After generating the JSONL files, the datasets need to be mixed and shuffled. Follow these steps:

1. Set the temporary directory and memory allocation:
    ```shell
    export TMPDIR='/global/scratch/users/yutong/data/temp'
    export MEMORY='20'
    ```

2. Navigate to your data directory:
    ```shell
    cd /path/to/your/data/
    ```

3. Mix and shuffle the datasets:
    ```shell
    cat tokenized_tasks/*.jsonl | terashuf > mix_and_shuffled/dataset.jsonl
    ```

- This will create a mixed and shuffled dataset file named `dataset.jsonl` in the `mix_and_shuffled` directory.