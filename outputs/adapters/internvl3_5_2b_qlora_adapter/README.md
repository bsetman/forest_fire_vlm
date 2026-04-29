---
library_name: peft
license: apache-2.0
base_model: OpenGVLab/InternVL3_5-2B-HF
tags:
- axolotl
- base_model:adapter:OpenGVLab/InternVL3_5-2B-HF
- lora
- transformers
datasets:
- /mnt/d/Abiyelunwen/new_fire_vlm/data/internvl35/train_internvl35_axolotl.jsonl
pipeline_tag: text-generation
model-index:
- name: mnt/d/Abiyelunwen/new_fire_vlm/outputs/internvl35_2b_hf_qlora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.16.1`
```yaml
# InternVL3.5-2B-HF QLoRA training config for Axolotl.
# Uses existing Axolotl-format JSONL files:
#   data/internvl35/train_internvl35_axolotl.jsonl
#   data/internvl35/val_internvl35_axolotl.jsonl

base_model: OpenGVLab/InternVL3_5-2B-HF
base_model_config: OpenGVLab/InternVL3_5-2B-HF
tokenizer_config: OpenGVLab/InternVL3_5-2B-HF
# Keep this field; if your Axolotl version supports it, it prevents processor_config=None.
processor_config: OpenGVLab/InternVL3_5-2B-HF
trust_remote_code: true

# Multimodal settings required by Axolotl VLM training.
processor_type: AutoProcessor
skip_prepare_dataset: true
remove_unused_columns: false
sample_packing: false

# Existing dataset files. They already use messages/content with {type: image, path: ...} and {type: text, text: ...}.
datasets:
  - path: /mnt/d/Abiyelunwen/new_fire_vlm/data/internvl35/train_internvl35_axolotl.jsonl
    type: chat_template
    chat_template: tokenizer_default
    message_property_mappings:
      role: role
      content: content

test_datasets:
  - path: /mnt/d/Abiyelunwen/new_fire_vlm/data/internvl35/val_internvl35_axolotl.jsonl
    type: chat_template
    chat_template: tokenizer_default
    message_property_mappings:
      role: role
      content: content

# Keep close to your previous Qwen3-VL LoRA settings.
sequence_len: 1024
num_epochs: 3
micro_batch_size: 1
eval_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.0001
warmup_ratio: 0.03
logging_steps: 5
eval_steps: 50
save_steps: 50
save_total_limit: 2

# Axolotl run output. Final adapter will be copied by the shell script to models/internvl35_2b_hf_qlora_adapter.
output_dir: /mnt/d/Abiyelunwen/new_fire_vlm/outputs/internvl35_2b_hf_qlora

# QLoRA / 4bit settings.
adapter: qlora
load_in_4bit: true
load_in_8bit: false
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: float16

# Match Qwen LoRA rank/alpha.
lora_r: 4
lora_alpha: 8
lora_dropout: 0.05

# Freeze vision side; train language-side LoRA only.
freeze_mm_modules: true
lora_target_modules: 'model.language_model.layers.[\d]+.(mlp|cross_attn|self_attn).(up|down|gate|q|k|v|o)_proj'

# Precision / memory.
fp16: true
bf16: false
tf32: true
gradient_checkpointing: true

# Conservative image settings for RTX 4060 Laptop 8GB.
image_size: 448
image_resize_algorithm: bilinear

# Optimizer.
optimizer: paged_adamw_8bit
lr_scheduler: cosine
weight_decay: 0.0

# Stability.
seed: 42
strict: false
generate_samples: false

```

</details><br>

# mnt/d/Abiyelunwen/new_fire_vlm/outputs/internvl35_2b_hf_qlora

This model is a fine-tuned version of [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF) on the /mnt/d/Abiyelunwen/new_fire_vlm/data/internvl35/train_internvl35_axolotl.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0568
- Ppl: 1.0585
- Memory/max Active (gib): 2.4
- Memory/max Allocated (gib): 2.4
- Memory/device Reserved (gib): 2.57

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use paged_adamw_8bit with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 5
- training_steps: 188
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Ppl     | Active (gib) | Allocated (gib) | Reserved (gib) |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:------------:|:---------------:|:--------------:|
| No log        | 0     | 0    | 4.3066          | 74.1904 | 2.07         | 2.07            | 3.19           |
| 0.0988        | 0.8   | 50   | 0.0820          | 1.0855  | 2.12         | 2.12            | 2.57           |
| 0.0627        | 1.592 | 100  | 0.0597          | 1.0615  | 2.12         | 2.12            | 2.49           |
| 0.0575        | 2.384 | 150  | 0.0571          | 1.0588  | 2.12         | 2.12            | 2.45           |
| 0.0582        | 2.992 | 188  | 0.0568          | 1.0585  | 2.4          | 2.4             | 2.57           |


### Framework versions

- PEFT 0.19.1
- Transformers 5.5.0
- Pytorch 2.8.0+cu128
- Datasets 4.5.0
- Tokenizers 0.22.2