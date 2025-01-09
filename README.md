# My Crypto Prediction
1. Create the environment:
   conda env create -f environment.yml
   conda activate my_crypto_env

2. Split data:
   python src/data/split_data.py

3. Train model:
   python src/models/train_model.py

4. Evaluate on test set:
   python src/models/evaluate_model.py --data data/final/test.csv# kaggle_cryptocomp
