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

