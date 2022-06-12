echo "Running the Segmentor.."
bash modules.sh
bash data.sh
bash callbacks.sh
python main.py
echo "Run Done!!"
