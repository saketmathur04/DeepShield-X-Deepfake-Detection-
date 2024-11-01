# DeepShield-X — Manipulation-Wise Sub-Category Evaluation

Evaluation of ResNet50 Regularized across individual manipulation methods in FaceForensics++ C23.

| Manipulation Method | Ground Truth | Sample Count | Accuracy | Recall | Specificity | Mean Fake Prob | Missed Samples (Errors) |
|---|---|---|---|---|---|---|---|
| **DeepFakeDetection** | FAKE | 615 | 0.9967 | 0.9967 | N/A (Fake) | 0.9889 | FN = 2 |
| **Deepfakes** | FAKE | 1,492 | 0.9732 | 0.9732 | N/A (Fake) | 0.9657 | FN = 40 |
| **Face2Face** | FAKE | 1,499 | 0.9700 | 0.9700 | N/A (Fake) | 0.9622 | FN = 45 |
| **FaceShifter** | FAKE | 1,495 | 0.9686 | 0.9686 | N/A (Fake) | 0.9568 | FN = 47 |
| **FaceSwap** | FAKE | 1,496 | 0.9733 | 0.9733 | N/A (Fake) | 0.9658 | FN = 40 |
| **NeuralTextures** | FAKE | 1,496 | 0.8536 | 0.8536 | N/A (Fake) | 0.8363 | FN = 219 |
| **original** | REAL | 1,496 | 0.8710 | 0.8710 | 0.8710 | 0.1497 | FP = 193 |

### Key Scientific Insights:
- **Hardest Manipulation (Highest Miss Rate)**: **NeuralTextures** (Recall = 85.36%, FN = 219). Over 55.7% of all false negatives (219 / 393) occur on NeuralTextures due to subtle facial reenactment confined to the perioral/lip region.
- **Strongest Detection**: **DeepFakeDetection** (Recall = 99.67%, FN = 2) and **FaceSwap** (Recall = 97.33%, FN = 40) display prominent boundary artifacts easily caught by residual feature pyramids.
- **Original REAL Specificity**: 87.10% (1,303 correctly identified, 193 false alarms).
