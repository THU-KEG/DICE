# Acknowledgements
This code is borrowed from the repository of the paper "Evading Data Contamination Detection for Language Models is (too) Easy" by Jasper Dekoninck, Mark Niklas Müller, Maximilian Baader, Marc Fischer, and Martin Vechev. The original repository can be found [here](https://github.com/eth-sri/malicious-contamination/). Their LICENSE file can be found in this folder as well. We have made some modifications to the code to adapt it to our needs. 

# Evading Data Contamination Detection for Language Models is (too) Easy
This repo contains the code for our paper, *Evading Data Contamination Detection for Language Models is (too) Easy*. We explain how to install and run the code in this repository to reproduce the results presented in the paper. The code assumes that CUDA is available and that at least 80GB of memory is present on the GPU(s). 

## Installation
You can install the code in this repo by installing [Conda](https://docs.conda.io/projects/miniconda/en/latest/) and running the following commands:

```bash
conda create -n contamination python=3.11
conda activate contamination
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation
```

If you want to have the exact same versions we used for all packages, you can run the following command after this:
```bash
python -m pip install -r requirements.txt
```

## Reproducing Results

Before starting, you should either add your OpenAI API Key as an environment variable with the key OPENAI_API_KEY or create the `scripts/.env` file with the following content:
```bash
OPENAI_API_KEY=[YOUR API KEY]
```

You can reproduce all our results by running the following command:

```bash
bash scripts/main.sh
```

Note that this can take several weeks to run on a single H100 Nvidia GPU, use a couple hundred USD with the OpenAI API and needs around 500GB of storage space. 

We note that this repo contains a baseline copied from [this GitHub repo](https://github.com/swj0419/detect-pretrain-code-contamination) to implement the method by Shi for benchmark-level contamination in the [`code-contamination-detection`](code-contamination-detection) subfolder. Changes that were made were documented with the a comment starting with `# NOTE`, but we note that these changes were solely used to change data preprocessing procedures such that the evaluated benchmarks align with our contamination.

After this, a postprocessing notebook located at [`notebooks/postprocessing.ipynb`](notebooks/postprocessing.ipynb) can be run to get our final results.

## Citation
Please use the following citation to cite this repo or the associated paper:

```
@article{dekoninck2024evading,
      title={Evading Data Contamination Detection for Language Models is (too) Easy}, 
      author={Jasper Dekoninck and Mark Niklas Müller and Maximilian Baader and Marc Fischer and Martin Vechev},
      year={2024},
      eprint={2402.02823},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```