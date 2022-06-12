echo "Downloading Data..."
python -m tensorflow.datasets.script.download_prepare --register_checksums --datasets=oxford_iiit_pet:3.1.0
echo "Data downloaded and ready to use!!"