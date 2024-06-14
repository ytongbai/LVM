
# Dataset Preparation

This section describes how to prepare your dataset by tokenizing visual data into visual sentences and then mixing and shuffling the datasets.

## Download & Prepare Dataset

- For depth map, segmentation, surface normal, and edge, we generate with [prismer](https://github.com/NVlabs/prismer).
- For segmentation, we use the `tokenize_examples/map_color.py` for color mapping.
- For other datasets, we follow the setting described in [LVM](https://arxiv.org/abs/2312.00785).

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