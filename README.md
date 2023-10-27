# DrugCombMrc
The code and dataset for "RCFIND: Reading Comprehension Powered Semantic Fusion Network for Identification of N-ary Drug Combinations"

On this virtual environment, install all required dependencies via pip:
pip install -r requirements.txt

The dataset can be download from the following link:
https://drive.google.com/file/d/1PTCvLFV0rX7cFKtmCZ-T8xY2MXBNlQDC/view?usp=sharing)https://drive.google.com/file/d/1PTCvLFV0rX7cFKtmCZ-T8xY2MXBNlQDC/view?usp=sharing

Training
-----
You can train your own with our provided scripts. We recommend training on a GPU machine. We trained our models on machines with a 24GB Nvidia 4090 GPU running Ubuntu 18.04.

Single command to train a relation extractor based on PubmedBERT:
python train.py --do_train
