# Protocol for Processing Multivariate Time-series Electronic Health Records of COVID-19 Patients

## Requirements

To get started with the repository, ensure your environment meets the following requirements:

- Python 3.11+
- PyTorch 2.5+ (use Lightning AI)
- See `requirements.txt` for additional dependencies.

## Repository Structure

The code repository includes the following directory structure:

```bash
DataProcessCOVID19/
├── configs/ # contains example configs for machine learning and deep learning models 
├── datasets/ # contains datasets files and utils for loading data
├── losses/ # contains losses designed for tasks
├── metrics/ # contains metrics designed for tasks
├── models/ # backbone machine learning and deep learning models
├── pipelines/ # machine learning and deep learning pipelines under pytorch lightning framework
├── standardize_preprocess.py # steps for preparing and preprocess the EHR data
├── further_process.py # steps for further processing the EHR data
├── train_evaluate.py # steps for training and evaluating the AI models
└── requirements.txt # python software packages required for running the code
```

## Usage

The raw data in our research is already in the `datasets/tjh/raw/` folder.

To start with the data processing steps, use the following commands:

```bash
# Step 1: Standardize and preprocess the EHR data
python standardize_preprocess.py

# Step 2: Further process the EHR data
python further_process.py
```

To start with the training and evaluating, use the following command:

```bash
# Step 3: Train and evaluate the AI models
python train_evaluate.py
```

If you want to use datasets from other sources, please download the datasets and put them in the `datasets/` folder. Then, modify the file paths in the `standardize_preprocess.py` and `further_process.py`, and the `dataset` parameter in the `configs/config.py` file.

For more information about the TJH dataset we use in our research, please refer to the [paper](https://doi.org/10.1038/s42256-020-0180-7).

For a deeper dive into our research, please refer to our associated published [paper](https://doi.org/10.48550/arxiv.2209.07805) and Github [repository](https://github.com/yhzhu99/pyehr/tree/main).
