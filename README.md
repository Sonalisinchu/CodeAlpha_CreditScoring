# Credit Score Project

A simple project to predict creditworthiness using machine learning.

## Files

- `src/` - Python scripts (`train.py`, `predict.py`)  
- `data/` - Sample dataset (`sample_credit.csv`)  
- `requirements.txt` - Python dependencies  

## How to Run

1. Install dependencies:
 pip install -r requirements.txt
2. Train the model:
 python src/train.py --data data/sample_credit.csv --output models/model.pkl
3. Make predictions:
python src/predict.py --model models/model.pkl --input data/sample_credit.csv
Tested in **VS Code**.
