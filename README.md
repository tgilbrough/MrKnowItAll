# MrKnowItAll
CSE481N NLP Capstone Project [(Project Webpage)](https://tgilbrough.github.io/cse481n-blog/)

## Development

This project uses Python 3.

### Installing Dependencies

First, create a Python 3 virtual environment using `python -m venv venv`.

Next, activate the virtual environment with `source venv/bin/activate`.

Dependencies are listed in `requirements.txt`. They can be installed with `pip`
by using `pip install -r requirements.txt`.

### Downloading Data

Data is stored in the `data` directory and is not under version control.
There are scripts to download the various datasets.

### Using GPU Machine

SSH into `nlpgpu01.cs.washington.edu` and clone the repository.

The `setup_gpu_machine.sh` script will take care of:
 - Downloading GloVe embeddings and MS MARCO.
 - Creating a virtual environment and installing pip.
 - Installing dependencies using pip.

CUDA requires some environment variables. These can be set using `source cuda_path.sh`.

### Disable TensorFlow Logging

Run `source disable_tf_logging.sh`.

## Running Experiments

The `experiment.py` script is a utility for experimenting with different model
parameters. It accepts multiple values for each parameter and tries all
possible combinations of them.

For example, the following command will try batch sizes of 16, 32, and 64,
with question types of entity and location. It will always run for 10 epochs.

```
python experiment.py experiment_name -b 16 32 64 -q entity location -e 10
```

Results are dumped to the `tensorboard_models/` directory and can be viewed
by using `tensorboard --logdir tensorboard_models/`.
The name of each experiment includes the parameters used.

## Error Analysis

The `error_analysis.py` script takes the development dataset used, references, and one or more candidates.
It prints HTML which can be piped to a file.
This file must be in the `error_analysis/` directory for the CSS to work.

```
python error_analysis.py datasets/msmarco/dev/location.json references/location.json candidates/attention-batch_size=128-epochs=50-hidden_size=100-keep_prob=0.3-learning_rate=0.01-question_type=location.json candidates/baseline-batch_size=1024-epochs=50-hidden_size=100-keep_prob=0.3-learning_rate=0.01-question_type=location.json > error_analysis/index.html
```
