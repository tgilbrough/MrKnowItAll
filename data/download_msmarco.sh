
echo "Downloading training data..."
wget https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz -P msmarco
gunzip msmarco/train_v1.1.json.gz

echo "Downloading testing data..."
wget https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz -P msmarco
gunzip msmarco/test_public_v1.1.json.gz

echo "Downloading dev data..."
wget https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz -P msmarco
gunzip msmarco/dev_v1.1.json.gz
