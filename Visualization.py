import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(train_data):
    sns.countplot(x='Survived', data=train_data)
    plt.show()
    sns.countplot(x='Survived', hue='Sex', data=train_data)
    plt.show()
    sns.countplot(x='Survived', hue='Pclass', data=train_data)
    plt.show()
