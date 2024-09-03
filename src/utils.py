import pandas as pd
import pickle

def load_data(file_path):
    return pd.read_csv(file_path)

def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
