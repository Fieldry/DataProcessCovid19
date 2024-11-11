# Protocol for Processing Multivariate Time-series Electronic Health Records of COVID-19 Patients

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

For a deeper dive into our research, please refer to our associated published [paper](https://doi.org/10.48550/arxiv.2209.07805) and Github [repository](https://github.com/yhzhu99/pyehr/tree/main).