#CS-433 Machine Learning - Chess Move OCR

## Overview

This repository holds the code for the second project of the Machine Learning course at [EPFL](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/). The project team consists of the following members:

- Amaury George ([@AmauryGeorge](https://github.com/AmauryGeorge))
- Pierre-Hadrien Levieil ([@PH-14](https://github.com/PH-14))
- Albert Troussard ([@alberttkt](https://github.com/alberttkt))

The Data used for training is located in test_data/images for the one provided by ChessReader. For the samples we created, the code is in the data_generation folder.

## Project Structure

The project is organized as follows:

```markdown
|-- data_generation
|   |-- fonts : contains the fonts used to generate the data
|   |-- fetch_all_moves.ipynb : notebook used to fetch all the moves from the chess.com API
|   |-- chess_moves_generator.ipynb : notebook used to generate the data
|   |-- all_moves_proba.txt : contains the probability of each move
|
|-- tesseract
|   |-- training_tesseract.md : contains the instructions to train tesseract
|
|-- test_data
|   |-- images : contains the images used for testing
|   |-- prediction.csv : contains the original provided predictions for different OCR engines
|   |-- cleaned_labels.csv : contains the real label for each image
|   |-- readme.txt : explains the columns of the prediction.csv file
|
|-- torchModel
|   |-- Models : contains the saved models used for the prediction
|   |-- model.py : contains the model used for the prediction
|   |-- train.py : contains the training code
|   |-- configs.py : contains the configuration of the training
|   |-- inferenceModel.py : contains the prediction code
|
|-- ComparisonTable.csv : contains the comparison of the different OCR engines, including ours
|-- README.md
|-- requirements.txt : contains the required libraries to run the code
|-- paper.pdf : contains the report of the project
```

    
