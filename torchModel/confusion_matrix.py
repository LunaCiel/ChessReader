from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# File path - update with the path to your file
file_path = '/Users/luna/PycharmProjects/ml-project-2-apa-main/torchModel/incorrect_predictions.csv'

# Read your data
data = pd.read_csv(file_path)

# All unique characters in your labels
characters = sorted(set(''.join(data['True Label'].values) + ''.join(data['Predicted Label'].values)))

# Create an empty matrix of zeroes
character_conf_matrix = pd.DataFrame(
    [[0 for _ in range(len(characters))] for _ in range(len(characters))],
    index=characters, columns=characters
)

# Fill your matrix
for _, row in data.iterrows():
    true = row['True Label']
    predicted = row['Predicted Label']
    for i in range(min(len(true), len(predicted))):  # in case there are labels of different lengths
        if true[i] != predicted[i]:
            character_conf_matrix.loc[true[i], predicted[i]] += 1

# Visualize the matrix
plt.figure(figsize=(14, 14))
sns.heatmap(character_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted characters')
plt.ylabel('True characters')
plt.title('Character Confusion Matrix')
plt.show()