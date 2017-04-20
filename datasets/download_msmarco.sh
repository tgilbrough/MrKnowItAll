
cd msmarco
echo "Downloading training data..."
wget https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz
gunzip train_v1.1.json.gz

echo "Downloading testing data..."
wget https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz
gunzip test_public_v1.1.json.gz

echo "Downloading dev data..."
wget https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz
gunzip dev_v1.1.json.gz

echo "Splitting queries by type..."
python3 split_data.py
cd ..
