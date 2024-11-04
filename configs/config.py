ml_param = {
    "model": "CatBoost",
    "dataset": "tjh",
    "task": "outcome",  # "outcome" or "los"
    "max_depth": 5,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "batch_size": 81920,
    "main_metric": "auprc",
}

dl_example_param = {
    "model": "RNN",
    "dataset": "tjh",
    "task": "outcome", # "outcome", "los" or "multitask"
    "seed": 0,
    "epochs": 100,
    "patience": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "main_metric": "auprc",
    "demo_dim": 2,
    "lab_dim": 73,
    "hidden_dim": 32,
    "output_dim": 1,
    "accelerator": "gpu", # "cpu" or "gpu"
    "device": 0, # gpu device id if accelerator is "gpu"
},