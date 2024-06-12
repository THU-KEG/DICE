# DICE
DICE: Detecting In-distribution Data Contamination with LLM's Internal State
Data and Code for the paper.

## Installation

``` bash
git clone https://github.com/THU-KEG/DICE.git
cd DICE
```

## Reproducing Results

### Step 1: Fine-tune the contaminated model


Our code to fine-tune contaminated model is stored in the `OOD_test/scripts` folder. 

#### paraphrase benchmark

``` bash
python scripts/rewrite.py --dataset_name gsm8k
```
The paraphrased dataset we used in the paper is available in the `OOD_test/scripts/data` folder.

#### fine-tune 

- You can fine-tune a contaminated model as follows. Change the base model by `--model_name`.

- Change the contaminated benchmark by changing the `--train_dataset_name` and `--dataset_name`.

- The parameter `--epoch 1` represents the 2% contamination setting in the paper. Omitting it represents the 10% setting.

``` bash
cd OOD_test
CUDA_VISIBLE_DEVICES=0 python scripts/contaminated_finetune.py \
--model_name microsoft/phi-2 \
--generative_batch_size 32 \
--dataset_name gsm8k \
--train_dataset_name gsm8k \
--epochs 1
```

#### fine-tune scripts

You can also use the following script to directly reproduce the contaminated model of the main experiment in our paper.

``` bash
CUDA_VISIBLE_DEVICES=0 bash scripts/contaminated_finetune.sh
```

### Step 2: OOD Performance of contaminated models

Similar to the fine-tuning process above, you can use the following scripts to test OOD performance.

The parameter settings are the same as above. The only thing to note is that `--dataset_name` is the OOD dataset to be tested, and `--train_dataset_name` is the contaminated dataset.

``` bash
cd OOD_test
CUDA_VISIBLE_DEVICES=0 python OOD_generate_inf.py \
--model_name microsoft/phi-2 \
--generative_batch_size 32 \
--dataset_name math \
--train_dataset_name gsm8k \
--epochs 1
```

### Step 3: Locate contaminated layer

Code of this part is stored in the `Locate` folder. 

```bash
CUDA_VISIBLE_DEVICES=0 python DICE_locate.py \
--edited_model=meta-llama/Llama-2-7b-hf \
--hparams_dir=../hparams/DICE/llama-7b 
```

### Step 4: Train and test DICE detector

Code of this part is stored in the `contamination_classifier` folder. 

#### make data (hidden states of contaminated layer)

You can use the following script to get the data.

- You can fine-tune a contaminated model as follows. You can change the base model by `--model_name`.

- Change the detect benchmark by `--test_dataset`.

- `--is_contaminated` shows whether the model is contaminated.

- `--model_type` indicates whether the uncontaminated model is the vanilla model or the model fine-tuned only on orca.

- `--contaminated_type` indicates whether the contaminated model is a fine-tuned version of the original benchmark (open) or a paraphrased benchmark (Evasive).


```bash
cd contamination_classifier
CUDA_VISIBLE_DEVICES=0 python data_maker.py \
--edited_model=meta-llama/Llama-2-7b-hf \
--hparams_dir=../hparams/DICE/llama-7b \
--test_dataset=GSM8K_seen \
--is_contaminated=True \
--model_type=vanilla \
--contaminated_type=open
```
You can also use the following script to directly reproduce test data of the main experiment in our paper.

``` bash
CUDA_VISIBLE_DEVICES=0 bash scripts/make_test_data.sh
```

#### train and test DICE detector

Use `train_test.py` to train and test a DICE.

You can simply use the following script to directly reproduce test results of the main experiment in our paper.

``` bash
CUDA_VISIBLE_DEVICES=0 bash scripts/Test_DICE.sh
```

##### other experiment

The `contamination_classifier` folder contains the code for the main experiments in the paper, including the `performance_vs_score` subfolder that stores the code for the experiment to test the relationship between contaminated probability and model performance,  `draw_OOD.py` is the code for drawing the detection distribution of the OOD dataset, and so on.



# Cite
If you find our code useful, we will sincerely appreciate it and encourage you to cite the following article:

```bibtex
@misc{tu2024dice,
      title={DICE: Detecting In-distribution Contamination in LLM's Fine-tuning Phase for Math Reasoning}, 
      author={Shangqing Tu and Kejian Zhu and Yushi Bai and Zijun Yao and Lei Hou and Juanzi Li},
      year={2024},
      eprint={2406.04197},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
 ```
