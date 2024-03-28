import os
import csv
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import get_cer, get_wer
from decoder import ctc_decoder

# Modifying the existing class to include a method for incorrect predictions
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, (self.input_shapes[0][2], self.input_shapes[0][1]))
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_names[0]: image_pred})[0]
        text, unsure = ctc_decoder(preds, self.char_list)
        return text[0], unsure

    def write_incorrect_predictions(self, data):

        correct_count = 0
        incorrect_count = 0

        with open('incorrect_predictions.csv', 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "True Label", "Predicted Label"])

            for image_path, label in tqdm(data):
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image at {image_path}")
                    continue

                prediction_text, unsure = self.predict(image)

                if prediction_text != label:
                    writer.writerow([image_path, label, prediction_text])
                    incorrect_count += 1
                else:
                    correct_count += 1

            print(f"Total correct predictions: {correct_count}")
            print(f"Total incorrect predictions: {incorrect_count}")

if __name__ == "__main__":
    configs = BaseModelConfigs.load("Models/BestModels/best/configs.yaml")
    model = ImageToWordModel(
        model_path="Models/BestModels/best",
        char_list=configs.vocab
    )

    data_cr = pd.read_csv("../test_data/cleaned_labels.csv")
    ids_label = data_cr[["id", "prediction"]]
    data = []

    for id in ids_label["id"]:
        img_path = os.path.join("../test_data/images", f"{id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join("../test_data/images", f"{id}.jpe")
        label = ids_label[ids_label["id"] == id].iloc[:,-1].values[0]
        data.append([img_path, label])

    model.write_incorrect_predictions(data)