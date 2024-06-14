# Large Vision Model Evaluation
This is a evaluation demo for the Large Vision Model paper. 


### Setting Up with Conda Environment
You can also set up the demo using a conda environment. First, you will need to
create a conda environment and install the dependencies:
```bash
conda env create -f environment.yml
conda activate vqlm_demo
export PYTHONPATH=/path/to/this/repo:$PYTHONPATH

```

Then you'll need to download the [VQ tokenizer checkpoint file](https://huggingface.co/Emma02/vqvae_ckpts) and put it into ./vqvae_ckpts/


## Running the Perplexity Evaluation
This repo also contains the perplexity evaluation script. You can run the following
command to evaluate the perplexity of the model:

```bash
python -m vqlm_demo.eval_perplexity \
    --input_file=path/to/input_jsonl_file \
    --input_base_dir=base/path/to/add/to/the/input \
    --checkpoint=path/to/checkpoint \
    --batch_size=4
```

This script accept a jsonl file as input. Each line of the jsonl file
representing a dictionary. Each line represents one example in the evaluation
set. The dictionary should have two key:
* input: a list of paths to the input images as **context to the model**. This list should include the few shot examples.
* target: a list of paths to the **target images** to evaluate perplexity on.

Here's an example of the json format:
```javascript
{'input': ['path/to/input1.jpg', 'path/to/input2.jpg', 'path/to/input3.jpg'],
 'target': ['path/to/target1.jpg', 'path/to/target2.jpg', 'path/to/target3.jpg']}
```

When evaluating

Ths script should run the model and compute the average perplexity on the
evaluation set.


## Running the batch generation evaluation
This repo also contains the script to batch generate images from the model. You
can run the following command to generate images from the model:

```bash
python -m vqlm_demo.batch_generation \
    --checkpoint=path/to/checkpoint \
    --input_file=path/to/input_jsonl_file \
    --input_base_dir=base/path/to/add/to/input/path/in/jsonl \
    --output_base_dir=base/path/to/add/to/output/path/in/jsonl \
    --n_new_frames=1 \
    --n_candidates=4 \
    --resize_output='original'
```

This script accept a jsonl file as input. Each line of the jsonl file
representing a dictionary. Each line represents one example in the evaluation
set. The dictionary should have two key:
* input: a list of paths to the input images as **context to the model**. This list should include the few shot examples.
* output: a string representing the output path of generated image.

Here's an example of the json format:
```javascript
{'input': ['path/to/input1.jpg', 'path/to/input2.jpg', 'path/to/input3.jpg'],
 'output': 'path/to/output.jpg'}
```