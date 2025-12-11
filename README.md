# Music Style Transfer

This repository contains the notebooks and scripts used for our music style/velocity transfer project. The code includes data preprocessing, two separate training pipelines, inference/evaluation, and utilities for preparing your own data. If you want to see example results, check the outputs2/ and pred_piano/ folders. Our trained models are also in the models/ folder. 

# Model Training

You can train the two models provided in the project by running:

- music_transfer_violintotrumpet.py
  OR
- Data_Processing.ipynb

Users should supply their own aligned audio files.

Both notebooks allow you to load your own dataset and trigger full training. They output trained weights and intermediate plots, including MSE curves.

# Evaluation 

To test your trained model on new audio:

- predict_output_model1.ipynb
Load your trained model and run inference on test clips.
This notebook outputs:

Mean Squared Error (MSE) between predictions and ground truth

Place your test data in the appropriate folder and adjust the file paths in the notebook.

# Data Preparation Tools

If you want to preprocess your own dataset before training:

- saving_5_sec_clips.ipynb
Allows user to listen to audio and split longer recordings into 5-second audio segments suitable for training.

This notebook allow users to reproduce the dataset format we used or adapt it to their own instruments or genres

# Reproducibility

All notebooks are self-contained. Running the training notebooks end-to-end will reproduce our models, loss curves, and evaluation metrics.
