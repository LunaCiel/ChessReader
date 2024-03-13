import torch
import cv2
from tqdm import tqdm
import pandas as pd

from model import Network

# Configs
configs = ModelConfigs() # It assumes that you're able to restore the saved ModelConfigs object

# Load the trained model
network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
network = network.cpu()
network.load_state_dict(torch.load(configs.model_path + "/model.pt"))

# Create ImageToWordModel from your trained network
model = ImageToWordModel(network, configs.height, configs.width, configs.vocab)

# Load validation data from saved csv
val_data = pd.read_csv(os.path.join(configs.model_path, "val.csv"))
val_dataProvider = list(zip(val_data.image_path, val_data.label))

# Predict and gather incorrect ones
incorrect_predictions = []
for image_path, true_label in tqdm(val_dataProvider):
    image = cv2.imread(image_path)

    prediction, _ = model.predict(image)

    if prediction[0] != true_label:
        incorrect_predictions.append((image_path, true_label, prediction[0]))

# Print image paths where your model made a mistake
incorrect_images = [record[0] for record in incorrect_predictions]
print(incorrect_images)