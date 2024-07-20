# Titanic Survival Prediction

This project aims to build a predictive model to determine the likelihood of survival for passengers on the Titanic using data science techniques in Python. The project is structured into multiple modular files for better organization and maintainability.

## Project Structure


### Files

1. **load_data.py**
   - Contains the function `load_data` which reads the training and test datasets from CSV files.
   - Paths to the datasets are provided as arguments.

2. **visualization.py**
   - Contains the function `visualize_data` which creates visualizations to explore the survival rate and its relationship with different features such as gender and passenger class.
   - Uses Seaborn and Matplotlib for creating the plots.

3. **preprocessing.py**
   - Contains the function `preprocess_data` which handles data cleaning and feature engineering.
   - Fills missing values, drops unnecessary columns, encodes categorical variables, and creates new features such as `FamilySize` and `IsAlone`.

4. **model.py**
   - Contains the function `build_and_evaluate_model` which splits the dataset into training and testing sets, trains a logistic regression model, and evaluates its performance.
   - Prints accuracy, confusion matrix, and classification report, and visualizes the confusion matrix.

5. **main.py**
   - The main script that integrates all the modules.
   - Loads the dataset, performs exploratory data analysis (EDA), preprocesses the data, and builds and evaluates the model.
   - Paths to the dataset files are hardcoded in this script and should be adjusted as necessary.

## Steps in the Project

1. **Load the Dataset**:
   - The dataset is loaded from CSV files using Pandas.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizations are created to understand the distribution of data and relationships between features.
   - Plots such as survival rate distribution, survival rate by gender, and survival rate by class are generated.

3. **Data Preprocessing**:
   - Missing values are filled: median for `Age` and `Fare`, mode for `Embarked`.
   - The `Cabin` column is dropped due to a high percentage of missing values.
   - Categorical variables (`Sex` and `Embarked`) are encoded using `LabelEncoder`.
   - New features `FamilySize` and `IsAlone` are created to enhance the predictive power of the model.

4. **Model Building and Evaluation**:
   - The data is split into training and testing sets.
   - A logistic regression model is trained on the training set.
   - The model's performance is evaluated using accuracy, confusion matrix, and classification report.
   - The confusion matrix is visualized using a heatmap.

## Usage

### Prerequisites

Ensure you have the necessary libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
