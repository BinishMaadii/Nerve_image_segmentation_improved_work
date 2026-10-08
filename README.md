# Nerve_image_segmentation_improved_work
In this work the Ultrasound images for nerve detection in data from Kaggle is checked. Different approaches are used to improve the performance of nerve detection.

!pip install mlflow


pip install mlflow torch scikit-learn scipy pandas matplotlib pillow
python step8_train.py
python step9_diagnostics.py
python step10_explain.py
python step11_ablation.py   # trains 15 extra models, about 15 to 20 min on a GPU


The aim is not only a mask. The aim is a result you can check: how good it is, how sure the numbers are, whether the model looks at the nerve, and which design choice helped.

Status

The scripts were tested end to end on synthetic ultrasound-like images (15 patients, 40 frames each). They run, the tables and plots are written, and the internal checks pass.


Data

Download the Kaggle competition data and point DATA_DIR at it. The scripts search it recursively for <patient>_<frame>.tif images and their _mask.tif masks. Frames without a mask file are skipped. Images are shrunk to 96 x 128 pixels. A frame with an empty mask counts as "no nerve".

Many frames have no nerve. A model that always predicts "empty" is therefore right most of the time, which is why the main score is built to expose it (see below).

The model
Small U-Net with two encoder levels (16 and 32 channels) and a 64-channel middle layer.
Segmentation output: one mask per frame.
Detection head: a linear layer on the mean and max pooled middle layer. It answers "is there a nerve in this frame?".
Loss: binary cross-entropy with a positive pixel weight, plus Dice loss, plus 0.5 times the detection loss.
Training: Adam, learning rate 1e-3, batch size 32, oversampling of nerve frames with a weighted sampler.
Final prediction: a frame is called empty unless the head says there is a nerve. If it does, the largest connected blob of the mask is kept.
