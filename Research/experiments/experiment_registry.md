# DeepShield-X Experiment Registry

| Experiment ID | Architecture | Pretrained | Balanced | F1 | ROC-AUC | PR-AUC | Specificity | Notes |
|---|---|---|---|---|---|---|---|---|
| custom_cnn | custom_cnn | False | False | 0.9154 | 0.5607 | 0.8799 | 0.0000 | Baseline model. |
| class_balanced_custom_cnn | custom_cnn | False | True | 0.5265 | 0.6196 | 0.8966 | 0.7487 | Controlled experiment: tests effect of class weighting. |
| resnet50 | resnet50 | True | False | 0.9474 | 0.9578 | 0.9914 | 0.8503 | Baseline model. |
| resnet50_regularized | resnet50 | True | False | 0.9633 | 0.9718 | 0.9938 | 0.8710 | Controlled experiment: applies strong regularization to combat overfitting observed in ResNet50 baseline. |
| efficientnet | efficientnet | True | False | 0.9521 | 0.9547 | 0.9879 | 0.8549 | Baseline model. |
| mobilenet | mobilenet | True | False | 0.9274 | 0.9187 | 0.9820 | 0.7099 | Baseline model. |
| xception | xception | True | False | 0.9621 | 0.9706 | 0.9931 | 0.8830 | Baseline model. |
| vit | vit | True | False | 0.9394 | 0.9512 | 0.9894 | 0.8275 | Baseline model. |