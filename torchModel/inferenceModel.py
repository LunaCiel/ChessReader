import cv2
import typing
import numpy as np
import os


from mltu.configs import BaseModelConfigs

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import get_cer, get_wer
from decoder import ctc_decoder

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self,char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list
    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text,unsure = ctc_decoder(preds, self.char_list)

        return text[0],unsure

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    configs = BaseModelConfigs.load("Models/BestModels/best/configs.yaml")
    model = ImageToWordModel(
        model_path="Models/BestModels/best",
        char_list=configs.vocab
    )
    accum_cer = []
    accum_wer = []
    l=[]


    data_cr = pd.read_csv("/Users/luna/PycharmProjects/ml-project-2-apa-main/test_data/cleaned_labels.csv")
    ids_label = data_cr[["id", "prediction"]]
    for id in tqdm(ids_label["id"]):
        #try either png or jpe
        img_path = os.path.join("/Users/luna/PycharmProjects/ml-project-2-apa-main/test_data/images", f"{id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join("/Users/luna/PycharmProjects/ml-project-2-apa-main/test_data/images", f"{id}.jpe")
        label = ids_label[ids_label["id"] == id]["prediction"].values[0]
        l.append([img_path, label])

    for image_path, label in tqdm(l):
        image = cv2.imread(image_path)
        
        prediction_text, unsure = model.predict(image)
        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)

        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}, unsures: {unsure} ")

        accum_cer.append(cer)
        accum_wer.append(wer)

    print(f"Average CER: {np.average(accum_cer)}")
    print(f"Average WER: {np.average(accum_wer)}")
    print(f"Total Accuracy: {(1-np.average(accum_wer))*100:.2f}%")
