from load_data import load_data
from visualization import visualize_data
from preprocessing import preprocess_data
from model import build_and_evaluate_model

# File paths
train_path = 'C:/Users/MADHU VARDHAN BK/OneDrive/MYPRO/PYTHON/train.csv'
test_path = 'C:/Users/MADHU VARDHAN BK/OneDrive/MYPRO/PYTHON/test.csv'

# Load data
train_data, test_data = load_data(train_path, test_path)

# Display the first few rows of the dataset
print(train_data.head())

# Check for missing values
print(train_data.isnull().sum())

# Visualize data
visualize_data(train_data)

# Preprocess data
train_data = preprocess_data(train_data)

# Build and evaluate model
build_and_evaluate_model(train_data)
