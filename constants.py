"""Constants used in scripts."""

# Model experiment set-up
SEED = 117

EXPERIMENT_KEYS = [
    ["broad", "ntee"],
    ["sklearn", "none"],
    ["train", "valid", "test"],
]

# WandB hyperparamter sweep
SWEEP_CONFIG_RANDOM = {
    "method": "random",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "optimizer": {"value": "adam"},
        "classifier_dropout": {"values": [0.2, 0.3]},
        "learning_rate": {"values": [0.00003, 0.00005]},
        "epochs": {"values": [1, 2]},
        "batch_size": {"value": 32},
        "split_size": {"value": 0.2},
        "perc_warmup_steps": {"values": [0, 0.1, 0.25]},
        "clip_grad": {"values": [True, False]},
        "max_length": {"values": [64, 128]},
        "frac": {"value": 1.0},
    },
}

