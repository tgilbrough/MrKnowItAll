#!/bin/sh
 
cd datasets/

echo "Downloading GloVe embeddings..."
./download_glove.sh 
cd glove/
rm glove.6B.{100,200,300}d.txt
cd ..

echo "Downloading MS MARCO..."
./download_msmarco.sh 
cd ..

echo "Creating a virtual environment..."
python3 -m venv --without-pip venv
source venv/bin/activate

echo "Installing pip..."
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py 
rm get-pip.py

echo "Installing dependencies..."
pip install -r requirements.txt

