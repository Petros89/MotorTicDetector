# Motor Tic Detector 

## Description
- Pytorh GPU implementation of CNN for Motor-Tic Detection in Movement Disorders
- Eye Tics vs Normal Classification


## Code
- All source code is in `src`.
- Train using the `CNN.py` file.
- Get best model using the `TestCNN.py` file.

## Documentation
- Code is the documentation of itself.

## Usage
- Use `python3 CNN.py` to generate a confusion matrix.
- A summary of the pipeline is given in `report.pdf`.

## Demonstration
The pipeline is demonstrated below.

- Training Curves.

| Losses | Accuracies |
| --- | --- |
| ![](./figs/real_loss.PNG) | ![](./figs/real_accuracy.PNG) |

- Classification Confusion Matrix.

| Absolute Values | Percentages |
| --- | --- |
| ![](./figs/val_conf_mat.png) | ![](./figs/val_conf_mat_percent.png) |

## Contact
- apost035@umn.edu, trs.apostolou@gmail.com


