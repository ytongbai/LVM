# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Xinyang Geng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script converts LLaMA model checkpoint trained by EsayLM to the
# HuggingFace transformers LLaMA PyTorch format, which can then be loaded
# by HuggingFace transformers.

import gc
import json
import math
import os
import shutil

import numpy as np
import mlxu
import jax
import jax.numpy as jnp
import flax
from flax.traverse_util import flatten_dict
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import float_tensor_to_dtype
from EasyLM.models.llama.llama_model import LLAMA_STANDARD_CONFIGS


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    load_checkpoint='',
    model_size='vqlm_1b',
    output_dir='',
)


def match_keywords(string, positives, negatives):
    for positive in positives:
        if positive not in string:
            return False
    for negative in negatives:
        if negative in string:
            return False
    return True


def load_and_convert_checkpoint(path):
    _, flax_params = StreamingCheckpointer.load_trainstate_checkpoint(path)
    flax_params = flatten_dict(flax_params['params'], sep='.')
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ["kernel"], ["norm", 'ln_f']):
            tensor = tensor.T
        torch_params[key] = torch.tensor(
            float_tensor_to_dtype(tensor, 'fp32'), dtype=torch.float16
        )
    return torch_params


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(loaded, model_path, model_size):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = LLAMA_STANDARD_CONFIGS[model_size]

    n_layers = params["num_hidden_layers"]
    n_heads = params["num_attention_heads"]
    dim = params["hidden_size"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)


    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                loaded[f"transformer.h.{layer_i}.attention.wq.kernel"]
            ),
            f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                loaded[f"transformer.h.{layer_i}.attention.wk.kernel"]
            ),
            f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"transformer.h.{layer_i}.attention.wv.kernel"],
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"transformer.h.{layer_i}.attention.wo.kernel"],

            f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"transformer.h.{layer_i}.feed_forward.w1.kernel"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"transformer.h.{layer_i}.feed_forward.w2.kernel"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"transformer.h.{layer_i}.feed_forward.w3.kernel"],

            f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"transformer.h.{layer_i}.attention_norm.kernel"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"transformer.h.{layer_i}.ffn_norm.kernel"],

        }

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        # Unsharded
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.embedding"],
        "model.norm.weight": loaded["transformer.ln_f.kernel"],
        "lm_head.weight": loaded["lm_head.kernel"],
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        vocab_size=params["vocab_size"],
        max_position_embeddings=params["max_sequence_length"],
        hidden_size=dim,
        intermediate_size=params["intermediate_size"],
        num_attention_heads=params["num_attention_heads"],
        num_hidden_layers=params["num_hidden_layers"],
        rms_norm_eps=params["rms_norm_eps"],
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float16)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the Transformers format.")
    model.save_pretrained(model_path)
    shutil.rmtree(tmp_model_path)


def main(argv):
    assert FLAGS.load_checkpoint != "" and FLAGS.output_dir != ""
    assert FLAGS.model_size in LLAMA_STANDARD_CONFIGS

    write_model(
        load_and_convert_checkpoint(FLAGS.load_checkpoint),
        model_path=FLAGS.output_dir,
        model_size=FLAGS.model_size,
    )


if __name__ == "__main__":
    mlxu.run(main)