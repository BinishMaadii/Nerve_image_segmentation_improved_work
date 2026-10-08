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
