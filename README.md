# Forest Fire VLM

A research repository for structured forest-fire image understanding with Vision-Language Models (VLMs). The project focuses on detecting fire-related visual evidence in wildfire images and producing a constrained JSON annotation that can be used for model comparison, error analysis, and thesis experiments.

The current implementation is organized around five structured fields:

```json
{
  "Smoke": "yes | no",
  "Fire": "yes | no",
  "Fire_Size": "small | large | cannot_determine | no_fire",
  "Fire_Hotspots": "one_hotspot | multiple_hotspots | cannot_determine | no_fire",
  "Description": "A short description of the fire situation in the image"
}
```
The four fields—Smoke, Fire, Fire_Size, and Fire_Hotspots—are the primary focus of this study, and Description generates rule templates based on the structured prediction results derived from these four fields.

The repository includes data-processing utilities, LoRA/QLoRA training configurations for local VLMs, local inference scripts, cloud-model inference scripts, adapter outputs, and prediction files.

---

## Project Goal

The objective of this project is to develop VLM models suitable for forest fire detection and automatic annotation tasks, and to evaluate whether modern VLMs can generate reliable structured annotations for forest fire images. Instead of free-form captions, the models are required to output only a valid JSON object with predefined fields. This makes the task easier to evaluate and more suitable for a thesis experiment.

The main research workflow is:

1. Prepare and normalize wildfire image annotations.
2. Convert the dataset into model-specific JSONL formats.
3. Fine-tune local VLMs with LoRA or QLoRA.
4. Generate predictions with zero-shot, fine-tuned, and cloud-based models.
5. Compare structured predictions against the gold labels.
6. Analyze typical errors such as false fire detection, missed smoke, uncertain fire size, and hotspot-count confusion.

---

## Repository Structure

```text
forest_fire_vlm/
├── data/
│   ├── for_results_evaluation/      # Evaluation JSONL files
│   ├── image_annotation/            # Annotation-related files
│   ├── internvl35/                  # InternVL3.5 training data
│   ├── qwen3vl/                     # Qwen3-VL training data
│   ├── test/                        # Test images or test split files
│   ├── train/                       # Training images or train split files
│   └── val/                         # Validation images or validation split files
│
├── outputs/
│   ├── adapters/
│   │   ├── internvl3_5_2b_qlora_adapter/
│   │   └── qwen3vl_2b_lora_adapter/
│   ├── predictions/
│   │   ├── gemini25flash_predictions.jsonl
│   │   ├── gpt41mini_predictions.jsonl
│   │   ├── internvl35_finetuned_predictions.jsonl
│   │   ├── internvl35_zeroshot_predictions.jsonl
│   │   ├── qwen3vl_lora_predictions.jsonl
│   │   └── qwen3vl_zeroshot_predictions.jsonl
│   └── logs/
│       ├── internvl35_train.log
│       └── qwen3vl_train.log
│
├── scripts/
│   ├── data_process/
│   │   ├── build_test_jsonl.py
│   │   ├── convert_csv_to_qwen3vl_jsonl.py
│   │   ├── convert_jsonl_from_qwen3vl_to_internvl35.py
│   │   ├── rename_images_fire.py
│   │   └── rename_images_no_fire.py
│   │
│   ├── model_train/
│   │   ├── internvl35_train/
│   │   │   ├── internvl35_2b_hf_qlora_axolotl.yml
│   │   │   └── train_internvl35_hf_qlora.sh
│   │   └── qwen3vl_train/
│   │       ├── qwen3vl_lora_sft.yml
│   │       └── train_qwen3vl_lora.sh
│   │
│   ├── results_evaluation/
│   │   ├── predict_structured_cloud.py
│   │   └── predict_structured_local.py
│   │
│   └── inference/
│       └── infer_image.py
│
├── .gitattributes
├── .gitignore
└── README.md
```

---

## Models Used

### Local models

- `Qwen/Qwen3-VL-2B-Instruct`
- `OpenGVLab/InternVL3_5-2B-HF`

The local models can be evaluated in two modes:

- zero-shot inference, without an adapter;
- fine-tuned inference, with LoRA/QLoRA adapters stored under `outputs/adapters/`.

### Cloud models

The repository also supports structured prediction through a GPTsAPI-compatible endpoint for:

- `gpt-4.1-mini`
- `gemini-2.5-flash`

Cloud-model evaluation is optional and may incur API costs.

---

## Output Schema

All models are prompted to output only a JSON object with exactly these fields:

| Field | Allowed values | Meaning |
|---|---|---|
| `Smoke` | `yes`, `no` | Whether visible smoke is present in the image. |
| `Fire` | `yes`, `no` | Whether visible flame or active fire is present. |
| `Fire_Size` | `small`, `large`, `cannot_determine`, `no_fire` | Approximate visual scale of the fire. |
| `Fire_Hotspots` | `one_hotspot`, `multiple_hotspots`, `cannot_determine`, `no_fire` | Number or distribution of visible fire hotspots. |

Logical post-processing rules used in the prompt:

1. If `Smoke = no` and `Fire = no`, then `Fire_Size = no_fire` and `Fire_Hotspots = no_fire`.
2. If `Smoke = yes` and `Fire = no`, then `Fire_Size = cannot_determine` and `Fire_Hotspots = cannot_determine`.
3. The model must output JSON only, without Markdown or explanations.

Example:

```json
{
  "Smoke": "yes",
  "Fire": "yes",
  "Fire_Size": "large",
  "Fire_Hotspots": "multiple_hotspots"
}
```

---

## Environment Setup

The repository uses three practical runtime environments. They should be treated separately because the cloud-model scripts, Qwen3-VL local scripts, and InternVL3.5 local / training scripts have different dependency requirements.

### 1. Clone the repository

```bash
git clone https://github.com/bsetman/forest_fire_vlm.git
cd forest_fire_vlm
```

### 2. Cloud-model environment

Use this environment when running cloud VLMs through `scripts/results_evaluation/predict_structured_cloud.py`. This mode does not require a local GPU or locally downloaded VLM weights. It only sends the image and prompt to the configured GPTsAPI-compatible endpoint.

```bash
conda create -n fire-vlm-cloud python=3.10 -y
conda activate fire-vlm-cloud

pip install -U pip
pip install requests pillow tqdm
```

Set the API key before running cloud inference:

```bash
export GPTSAPI_API_KEY="your_api_key_here"
```

Windows PowerShell equivalent:

```powershell
$env:GPTSAPI_API_KEY="your_api_key_here"
```

Supported cloud models in the current scripts are:

- `gpt-4.1-mini`
- `gemini-2.5-flash`

Cloud inference is mainly used as an external comparison group. It may incur API costs depending on the selected provider, model, and number of evaluated images.

### 3. Local Qwen3-VL environment

Use this environment for Qwen3-VL zero-shot inference, LoRA adapter loading, and Qwen3-VL LoRA training / evaluation.

```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl

pip install -U pip
pip install torch torchvision torchaudio
pip install transformers accelerate peft bitsandbytes pillow tqdm qwen-vl-utils
```

This environment is used with:

```text
Qwen/Qwen3-VL-2B-Instruct
scripts/model_train/qwen3vl_train/
scripts/results_evaluation/predict_structured_local.py --backend qwen3vl
scripts/inference/infer_image.py --backend qwen3vl
```

The `qwen-vl-utils` package is required for Qwen-style image preprocessing. If CUDA or 4-bit loading fails, reinstall PyTorch according to your CUDA version and verify that `torch.cuda.is_available()` returns `True`.

For Qwen3-VL training through ModelScope SWIFT, install `ms-swift` in this environment or in a dedicated training environment:

```bash
pip install ms-swift
```

### 4. Local InternVL3.5 environment

Use this environment for InternVL3.5 zero-shot inference, QLoRA adapter loading, and InternVL3.5 training / evaluation.

```bash
conda create -n internvl35 python=3.10 -y
conda activate internvl35

pip install -U pip
pip install torch torchvision torchaudio
pip install transformers accelerate peft bitsandbytes pillow tqdm
```

This environment is used with:

```text
OpenGVLab/InternVL3_5-2B-HF
scripts/model_train/internvl35_train/
scripts/results_evaluation/predict_structured_local.py --backend internvl35
scripts/inference/infer_image.py --backend internvl35
```

For InternVL3.5 training through Axolotl, install Axolotl in this environment or in a dedicated training environment:

```bash
pip install axolotl
```

Because Axolotl, Transformers, PEFT, BitsAndBytes, and CUDA versions can be sensitive to version combinations, it is recommended to keep the InternVL3.5 environment separate from the Qwen3-VL environment.

### Environment summary

| Environment | Purpose | GPU required | Main scripts |
|---|---|---:|---|
| `fire-vlm-cloud` | Cloud-model evaluation with GPT-4.1 mini and Gemini 2.5 Flash | No | `scripts/results_evaluation/predict_structured_cloud.py` |
| `qwen3vl` | Local Qwen3-VL zero-shot, LoRA inference, and training | Yes | `scripts/results_evaluation/predict_structured_local.py`, `scripts/inference/infer_image.py`, `scripts/model_train/qwen3vl_train/` |
| `internvl35` | Local InternVL3.5 zero-shot, QLoRA inference, and training | Yes | `scripts/results_evaluation/predict_structured_local.py`, `scripts/inference/infer_image.py`, `scripts/model_train/internvl35_train/` |

## Important Path Note

Some configuration files were written for a local WSL project path such as:

```text
/mnt/d/Abiyelunwen/new_fire_vlm
```

If you clone this repository to another location, update the paths in:

```text
scripts/model_train/qwen3vl_train/qwen3vl_lora_sft.yml
scripts/model_train/internvl35_train/internvl35_2b_hf_qlora_axolotl.yml
```

You may also need to update `image_path` values inside JSONL files if they point to absolute paths from the original machine.

---

## Data Preparation

The data-processing scripts are located in:

```text
scripts/data_process/
```

Typical usage includes:

```bash
# Build a unified test JSONL file
python scripts/data_process/build_test_jsonl.py

# Convert CSV annotations to Qwen3-VL JSONL format
python scripts/data_process/convert_csv_to_qwen3vl_jsonl.py

# Convert Qwen3-VL JSONL into InternVL3.5 / Axolotl format
python scripts/data_process/convert_jsonl_from_qwen3vl_to_internvl35.py
```

The exact input and output paths should be checked inside each script and adjusted to your local project directory.

---

## Training

### Qwen3-VL LoRA / QLoRA training

Configuration file:

```text
scripts/model_train/qwen3vl_train/qwen3vl_lora_sft.yml
```

Training script:

```bash
bash scripts/model_train/qwen3vl_train/train_qwen3vl_lora.sh
```

After training, the adapter should be saved and can be copied or organized under:

```text
outputs/adapters/qwen3vl_2b_lora_adapter/
```

### InternVL3.5 QLoRA training

Configuration file:

```text
scripts/model_train/internvl35_train/internvl35_2b_hf_qlora_axolotl.yml
```

Training script:

```bash
bash scripts/model_train/internvl35_train/train_internvl35_hf_qlora.sh
```

After training, the adapter should be saved and can be organized under:

```text
outputs/adapters/internvl3_5_2b_qlora_adapter/
```

---

## Local Structured Prediction

Use `predict_structured_local.py` for Qwen3-VL and InternVL3.5 inference.

### Qwen3-VL zero-shot

```bash
python scripts/results_evaluation/predict_structured_local.py \
  --backend qwen3vl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --test-jsonl data/for_results_evaluation/test_eval.jsonl \
  --out outputs/predictions/qwen3vl_zeroshot_predictions.jsonl \
  --use-4bit
```

### Qwen3-VL with LoRA adapter

```bash
python scripts/results_evaluation/predict_structured_local.py \
  --backend qwen3vl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --adapter outputs/adapters/qwen3vl_2b_lora_adapter \
  --test-jsonl data/for_results_evaluation/test_eval.jsonl \
  --out outputs/predictions/qwen3vl_lora_predictions.jsonl \
  --use-4bit
```

### InternVL3.5 zero-shot

```bash
python scripts/results_evaluation/predict_structured_local.py \
  --backend internvl35 \
  --model OpenGVLab/InternVL3_5-2B-HF \
  --test-jsonl data/for_results_evaluation/test_eval.jsonl \
  --out outputs/predictions/internvl35_zeroshot_predictions.jsonl \
  --use-4bit
```

### InternVL3.5 with QLoRA adapter

```bash
python scripts/results_evaluation/predict_structured_local.py \
  --backend internvl35 \
  --model OpenGVLab/InternVL3_5-2B-HF \
  --adapter outputs/adapters/internvl3_5_2b_qlora_adapter \
  --test-jsonl data/for_results_evaluation/test_eval.jsonl \
  --out outputs/predictions/internvl35_finetuned_predictions.jsonl \
  --use-4bit
```

If the script cannot find the images, check whether the paths inside the JSONL file are relative to the repository root or absolute WSL paths.

---

## Cloud Structured Prediction

Use `predict_structured_cloud.py` for GPTsAPI-compatible cloud inference.

Set your API key first:

```bash
export GPTSAPI_API_KEY="your_api_key_here"
```

### GPT-4.1 mini

```bash
python scripts/results_evaluation/predict_structured_cloud.py \
  --model gpt-4.1-mini \
  --test-jsonl data/for_results_evaluation/test_eval.jsonl \
  --out outputs/predictions/gpt41mini_predictions.jsonl
```

### Gemini 2.5 Flash

```bash
python scripts/results_evaluation/predict_structured_cloud.py \
  --model gemini-2.5-flash \
  --test-jsonl data/for_results_evaluation/test_eval.jsonl \
  --out outputs/predictions/gemini25flash_predictions.jsonl
```

The cloud script stores raw output, normalized prediction, JSON validity, parsing errors, elapsed time, token usage, and estimated cost when available.

---

## Single-Image Inference and Description Generation

The script `scripts/inference/infer_image.py` is used for interactive single-image inference after a local model has been loaded. It supports the same two local backends as the local evaluation script:

- `qwen3vl`
- `internvl35`

The script loads the selected base model and, if provided, a LoRA/QLoRA adapter. It then repeatedly asks the user to enter an image path and prints five final fields:

```text
Smoke
Fire
Fire_Size
Fire_Hotspots
description
```

The first four fields are generated by the VLM using the same unified prompt and label constraints as the evaluation pipeline. The `description` field is not treated as an additional free-form model output. Instead, it is generated by rule-based post-processing after the four structured fields have been parsed and normalized.

The script performs three post-processing steps:

1. It extracts a valid JSON-like object from the raw model response.
2. It normalizes aliases and repairs logically inconsistent outputs, for example:
   - if `Smoke = no` and `Fire = no`, then `Fire_Size = no_fire` and `Fire_Hotspots = no_fire`;
   - if `Smoke = yes` and `Fire = no`, then `Fire_Size = cannot_determine` and `Fire_Hotspots = cannot_determine`.
3. It builds a Russian natural-language description from the normalized prediction using predefined templates.

Example Qwen3-VL usage:

```bash
python scripts/inference/infer_image.py \
  --backend qwen3vl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --adapter outputs/adapters/qwen3vl_2b_lora_adapter \
  --load-4bit
```

Example InternVL3.5 usage:

```bash
python scripts/inference/infer_image.py \
  --backend internvl35 \
  --model OpenGVLab/InternVL3_5-2B-HF \
  --adapter outputs/adapters/internvl3_5_2b_qlora_adapter \
  --load-4bit
```

This script is mainly intended for qualitative demonstration, thesis examples, and checking individual images after training. For large-scale quantitative evaluation, use the scripts in `scripts/results_evaluation/` instead.

---

## Prediction File Format

Prediction files in `outputs/predictions/` are JSONL files. Each line usually contains:

```json
{
  "model_name": "Qwen/Qwen3-VL-2B-Instruct",
  "backend": "qwen3vl",
  "adapter": "outputs/adapters/qwen3vl_2b_lora_adapter",
  "sample_id": "test_0001",
  "image_path": "data/test/test_0001.png",
  "gold": {
    "Smoke": "yes",
    "Fire": "yes",
    "Fire_Size": "small",
    "Fire_Hotspots": "one_hotspot"
  },
  "raw_output": "{...}",
  "pred": {
    "Smoke": "yes",
    "Fire": "yes",
    "Fire_Size": "small",
    "Fire_Hotspots": "one_hotspot"
  },
  "json_valid": true,
  "parse_errors": [],
  "error": null,
  "elapsed_sec": 2.31
}
```

These files are intended for unified comparison across zero-shot, fine-tuned, and cloud-based models.

---

## Evaluation Strategy

The evaluation strategy follows the experimental design used in the thesis. All models are evaluated on the same unified test set of 300 images, using the same structured-output prompt and the same four target fields:

- `Smoke`
- `Fire`
- `Fire_Size`
- `Fire_Hotspots`

The evaluated model groups are:

- Qwen3-VL zero-shot
- Qwen3-VL fine-tuned
- InternVL3.5 zero-shot
- InternVL3.5 fine-tuned
- GPT-4.1 mini
- Gemini 2.5 Flash

For each model, predictions are generated and stored as JSONL files. The raw model output is first parsed to extract a valid JSON object. Since VLMs may sometimes generate Markdown code blocks, explanatory text, invalid JSON, or unsupported label values, the evaluation pipeline also performs field normalization and records parsing failures. Samples that cannot be parsed into a valid structured prediction are treated as structured-output failures.

The main comparison is performed between the normalized prediction field `pred` and the manually annotated reference field `gold`. No model self-evaluation is used. The manual annotations are treated as the ground-truth labels.

The evaluation treats `Smoke` and `Fire` as binary classification fields, while `Fire_Size` and `Fire_Hotspots` are treated as multi-class classification fields. The final metrics include:

- field-wise accuracy for `Smoke`;
- field-wise accuracy for `Fire`;
- field-wise accuracy for `Fire_Size`;
- field-wise accuracy for `Fire_Hotspots`;
- precision, recall, and F1-score for the binary fields `Smoke` and `Fire`;
- mean field accuracy across the four structured fields;
- exact-match accuracy across all four fields.

The exact-match metric is considered especially important because it measures whether the model correctly predicts the complete structured annotation for an image, rather than only individual fields.

---

## Adapters and Base Models

The repository stores adapter files under:

```text
outputs/adapters/
```

These adapters are not standalone full models. To run inference, you still need to download the corresponding base model from Hugging Face:

- `Qwen/Qwen3-VL-2B-Instruct`
- `OpenGVLab/InternVL3_5-2B-HF`

The adapter path should point to a directory containing files such as adapter configuration and adapter weights.

---

## Notes

- This is a research prototype for thesis experiments, not a production wildfire monitoring system.
- The model output should not be used as the only source of emergency decision-making.
- The structured labels are designed for controlled academic evaluation.
- Cloud-model predictions depend on the selected API provider, endpoint availability, pricing, and rate limits.
- Local inference requires sufficient GPU memory, especially when loading VLMs with image input.

---

## Suggested Citation / Acknowledgement

If this repository is used in a thesis or report, describe it as:

> A structured forest-fire image annotation pipeline based on Vision-Language Models, including data conversion, LoRA/QLoRA fine-tuning, local and cloud VLM inference, and field-wise evaluation of JSON-based wildfire annotations.

---

## License

No explicit license is currently specified in this repository. Please contact the repository owner before reusing the code or data outside the thesis/research context.
