# How to Use

# Install
1. Clone this repository to your local machine.
2. Install the enviroment by running
```
conda env create -f environment.yml
```


Then try:

```
mkdir -p KCom
tar -xzf "venv.tar.gz" -C "KCom"
conda activate KCom
pip install requirements.txt
```

3. Download the model from (https://huggingface.co/)

# Dataset
D4: https://github.com/BigBinnie/D4_baseline
CAMS: https://github.com/drmuskangarg/CAMS

# Main Performance
## train
run
```
torchrun --nproc_per_node=8 train.py \
  --model_name_or_path {your_model_path} \
  --dataset_name {train_dataset.json} \
  --fact \
  --label \
  --knowledge \
  --bf16 True \
  --output_dir {your_output_path} \
  --low_rank_training True \
  --num_train_epochs 3 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --r 8
```

## test
run
```
python test.py \
  --model_path {your_finetuned_model} \
  --dataset_name {DATASET_NAME} \
  --output_dir {results_output_path} \
  --test_file {test_dataset_path}
```
