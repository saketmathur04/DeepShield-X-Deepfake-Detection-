# DeepShield-X: Viva Questions & Answers

This document provides a comprehensive set of 80 viva questions covering the entire scope of the DeepShield-X project, designed to prepare you for defense.

## Section 1: Basic Machine Learning & Project Fundamentals (1–15)

**1. What is the primary objective of the DeepShield-X project?**
The primary objective is to build a robust, reproducible, and explainable deepfake detection system by evaluating various CNN and Vision Transformer architectures on the FaceForensics++ dataset.

**2. What is a "deepfake"?**
A deepfake is synthetic media where a person in an existing image or video is replaced with someone else's likeness using artificial neural networks, typically autoencoders or GANs.

**3. Why did you choose the FaceForensics++ (FF++) dataset?**
FF++ is a standard, widely recognized academic benchmark that provides multiple distinct manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures), allowing us to test generalization across different synthesis techniques.

**4. What does "C23" refer to in the dataset?**
C23 refers to the H.264 video compression level (Constant Rate Factor 23). It represents a moderate level of compression typical of social media uploads, making the detection task more realistic than using raw, uncompressed (C0) video.

**5. Is your problem a binary or multi-class classification task?**
It is a binary classification task: classifying an input face as either REAL (0) or FAKE (1).

**6. What is the input to your models?**
The input is a single $224 \times 224$ RGB face image, extracted and cropped from video frames, normalized using ImageNet statistics.

**7. Why use Binary Cross Entropy (BCE) Loss instead of standard Cross Entropy?**
Because this is a binary classification task with a single output logit per image, `BCEWithLogitsLoss` is computationally more stable and appropriate than standard multi-class cross-entropy which requires two output nodes.

**8. Why use `BCEWithLogitsLoss` instead of applying Sigmoid then `BCELoss`?**
`BCEWithLogitsLoss` combines the Sigmoid layer and the BCELoss into a single class, providing better numerical stability and preventing catastrophic cancellation during backpropagation.

**9. What is an epoch?**
An epoch is one complete pass of the entire training dataset through the neural network.

**10. What is a batch size?**
The batch size is the number of training samples processed simultaneously before the model's internal weights are updated. We used a batch size of 32 for training.

**11. What optimizer did you use and why?**
We used AdamW (Adam with decoupled weight decay). It adapts the learning rate per parameter while explicitly separating weight decay, which generally yields better generalization than standard Adam.

**12. What is data augmentation and why is it used?**
Data augmentation artificially expands the training set by applying random transformations (flips, color jitter, etc.) to the images. It acts as a regularizer, preventing the model from memorizing the training data and improving generalization.

**13. Why did you only apply augmentation to the training set?**
Augmentation must not be applied to validation or test sets because these sets must represent the true, unperturbed data distribution to accurately evaluate the model's real-world performance.

**14. What is transfer learning?**
Transfer learning involves taking a model trained on a large, generic dataset (like ImageNet) and fine-tuning it on a smaller, specific dataset (like FF++). It drastically reduces training time and improves performance by leveraging pre-learned feature extractors.

**15. What are the manipulation categories in FaceForensics++?**
Deepfakes (autoencoder-based face replacement), Face2Face (computer graphics-based facial reenactment), FaceSwap (graphics-based face replacement), and NeuralTextures (GAN-based facial reenactment).

---

## Section 2: Machine Learning Metrics & Evaluation (16–30)

**16. Why is Accuracy a misleading metric in this project?**
Because the dataset is highly imbalanced (e.g., 6,964 REAL vs. 43,742 FAKE faces in train). A model that naively predicts "FAKE" for every image would achieve ~84.4% accuracy despite having zero discriminative ability.

**17. What are False Positives in the context of DeepShield-X?**
A False Positive occurs when a true, authentic REAL image is incorrectly classified as FAKE.

**18. What are False Negatives?**
A False Negative occurs when a manipulated FAKE image is incorrectly classified as REAL.

**19. Which is more dangerous: a False Positive or a False Negative?**
It depends on the application. A False Positive might wrongly accuse someone of distributing manipulated media. A False Negative allows malicious deepfakes to bypass security. Our evaluation tracks both.

**20. What is Specificity?**
Specificity (True Negative Rate) measures the proportion of actual REAL images that are correctly identified. It is critical for measuring false-accusation rates.

**21. What is Recall (Sensitivity)?**
Recall measures the proportion of actual FAKE images that the model successfully detects.

**22. What is the F1 Score?**
The F1 score is the harmonic mean of Precision and Recall. It is much more informative than accuracy when classes are imbalanced.

**23. What is an ROC Curve?**
The Receiver Operating Characteristic curve plots the True Positive Rate (Recall) against the False Positive Rate across all possible classification thresholds.

**24. What does ROC-AUC measure?**
The Area Under the ROC Curve (ROC-AUC) measures the model's ability to rank a random FAKE example higher than a random REAL example. A perfect model has an AUC of 1.0; a random model has 0.5.

**25. Why is PR-AUC (Precision-Recall AUC) sometimes preferred over ROC-AUC?**
In highly imbalanced datasets with a massive majority of positives (FAKES), ROC-AUC can remain overly optimistic because the False Positive Rate is pulled down by the large number of True Positives. PR-AUC focuses specifically on the minority class dynamics and precision.

**26. What is MCC (Matthews Correlation Coefficient)?**
MCC is a correlation coefficient between observed and predicted binary classifications. It considers true and false positives and negatives, making it one of the most reliable statistical rates for imbalanced datasets.

**27. What is model calibration?**
Calibration measures whether a model's predicted probabilities align with the actual true frequencies. If a calibrated model predicts FAKE with 90% confidence for 100 images, exactly 90 of them should actually be FAKE.

**28. Why must decision thresholds be tuned using the validation set, never the test set?**
Tuning the threshold on the test set is a form of data leakage ("cherry-picking"). It artificially inflates reported performance because you are adapting to the specific test data rather than evaluating generalization.

**29. What is overfitting?**
Overfitting occurs when a model learns the statistical noise and specific details of the training data rather than the underlying general patterns, leading to high training accuracy but poor validation/test accuracy.

**30. How did you identify overfitting in the baseline ResNet50?**
The baseline ResNet50 achieved nearly 100% training accuracy very quickly, but validation loss plateaued and eventually began to increase, proving the model was memorizing the training set.

---

## Section 3: Deep Learning Architectures (31–45)

**31. What is a Convolutional Neural Network (CNN)?**
A CNN is a specialized deep neural network that uses sliding convolutional filters to automatically extract spatial features (like edges, textures, and shapes) from grid-like data such as images.

**32. Why did the Custom CNN fail to learn effectively?**
The Custom CNN lacked pre-trained weights (starting from random initialization) and had a relatively shallow capacity. Given the severe class imbalance, it collapsed into predicting the majority class (FAKE) to minimize loss, achieving 0% specificity.

**33. What is the core innovation of ResNet (Residual Networks)?**
ResNet introduces "skip connections" or residual blocks that allow gradients to bypass layers during backpropagation. This solves the vanishing gradient problem, enabling the training of very deep networks (e.g., 50 layers).

**34. What is Xception?**
Xception stands for "Extreme Inception." It replaces standard convolutions with depthwise separable convolutions, which decouple spatial filtering from channel-wise feature mixing, reducing parameters while increasing representational efficiency.

**35. Why is Xception highly competitive in your results?**
Depthwise separable convolutions are exceptionally good at capturing subtle spatial anomalies—like the microscopic blending boundaries and compression artifacts created during deepfake generation.

**36. What is EfficientNet?**
EfficientNet scales network width, depth, and resolution simultaneously using a compound coefficient, achieving state-of-the-art accuracy with significantly fewer parameters than ResNets.

**37. What is MobileNetV3 designed for?**
MobileNetV3 is designed for highly efficient inference on edge devices (like smartphones). It uses lightweight depthwise convolutions and hardware-aware neural architecture search.

**38. Why include MobileNet if its accuracy is lower?**
To demonstrate the Accuracy vs. Efficiency tradeoff. MobileNetV3 provides extremely fast inference and low memory footprint, making it viable for real-time mobile deepfake detection despite a slight drop in absolute F1 score.

**39. What is a Vision Transformer (ViT)?**
ViT adapts the Transformer architecture (originally for NLP) to images. It splits an image into fixed-size patches, linearly embeds them, adds positional encodings, and processes them through self-attention layers.

**40. How does a ViT process an image differently than a CNN?**
CNNs process images locally using sliding windows (convolutions), building up global context slowly. ViTs use self-attention, meaning every patch can interact with every other patch directly from the very first layer, capturing global relationships immediately.

**41. Why did ViT not decisively outperform the CNNs in your results?**
ViTs typically require massive datasets (e.g., JFT-300M) to outperform CNNs because they lack the "inductive bias" (translation invariance and locality) inherently built into convolutions. On our mid-sized dataset, pre-trained CNNs generalized slightly better.

**42. How does class imbalance affect backpropagation?**
If a batch contains mostly FAKE examples, the gradient updates will overwhelmingly optimize for FAKE features. The model quickly learns that simply guessing FAKE yields a mathematically low loss, ignoring the REAL class.

**43. How did you fix class imbalance in the controlled experiment?**
By applying class weighting to the loss function. The minority class (REAL) is given a higher penalty weight during backpropagation, forcing the network to pay equal attention to REAL errors.

**44. What was the effect of class balancing on the Custom CNN?**
It dramatically improved Specificity (from 0% to ~74.8%), proving the model finally learned to recognize REAL faces. However, overall F1 dropped because the model lost its trivial "guess FAKE" advantage.

**45. How did regularization help ResNet50?**
By applying early stopping, augmentation, and validation-based checkpointing, we prevented the network from memorizing the training set. The Regularized ResNet50 stopped at 10 epochs (preventing over-training) and achieved higher test F1 and Specificity.

---

## Section 4: Data Integrity and Leakage (46–60)

**46. What is Data Leakage?**
Data leakage occurs when information from outside the training dataset (specifically, from the test dataset) is used to create the model, artificially inflating performance metrics.

**47. Why is random frame-level splitting dangerous in deepfake detection?**
Video frames are highly correlated. If you randomly split frames, Frame 10 might go to training and Frame 11 to testing. The model learns the specific background, lighting, and identity of the video rather than general deepfake artifacts.

**48. How did you prevent data leakage?**
We implemented an identity/source-aware split. We grouped videos by source identity using a Union-Find (connected components) algorithm before extracting faces, ensuring that an identity in the training set never appears in the test set.

**49. What is the Union-Find algorithm?**
It is a data structure that tracks a set of elements partitioned into a number of disjoint subsets. We used it to merge any videos sharing the same source identity into single, indivisible clusters before train/test splitting.

**50. Why split before face extraction rather than after?**
Splitting after face extraction is too late, as the mapping back to the source identity might be lost or computationally expensive. Splitting at the video level guarantees zero overlap at the frame level.

**51. What preprocessing is applied to the face images?**
Faces are cropped using a face detector, resized to $224 \times 224$ pixels, and normalized using standard ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

**52. Why use ImageNet normalization?**
Because we are using transfer learning. The pre-trained weights expect input data to have the exact same statistical distribution (mean and variance) as the ImageNet data they were originally trained on.

**53. What does early stopping do?**
It monitors a validation metric (like `val_loss` or `val_roc_auc`) during training. If the metric fails to improve for a set number of epochs (patience), training is halted to prevent overfitting.

**54. Why do you save the "best" checkpoint rather than the "last"?**
Because validation performance often peaks and then degrades as a model overfits. Saving the "best" checkpoint ensures we evaluate the model at its optimal generalization state, regardless of when training finally stops.

**55. How do you ensure your experiments are reproducible?**
We fixed the random seed (`42`) across PyTorch, NumPy, and Python's random module. We also documented the exact environment (`environment.json`), configurations, and data splitting logic.

**56. What is cross-dataset generalization?**
It is testing a model on a completely different dataset (e.g., Celeb-DF) than it was trained on (FF++). It is the ultimate test of true generalization, revealing whether the model learned generic deepfake artifacts or dataset-specific biases.

**57. Why did you not fabricate or blindly invent results for missing experiments?**
Scientific integrity. Fabricating data destroys the reproducibility and validity of the research. Real engineering involves analyzing failure modes and limitations honestly.

**58. What is a "controlled experiment"?**
An experiment where exactly one variable is changed while all others are held constant. For example, comparing Unweighted vs. Class-Balanced CNNs using the same architecture, seed, and data.

**59. Why is a single authoritative result registry important?**
It prevents versioning errors and accidental cherry-picking. It ensures the final report is generated programmatically from the raw JSON metrics produced by the evaluation scripts.

**60. What hardware was used for this project?**
Training was conducted on an AWS EC2 g5.xlarge instance utilizing an NVIDIA A10G GPU with 24GB of VRAM, running PyTorch 2.1.

---

## Section 5: Advanced Topics, Robustness & Explainability (61–80)

**61. What is model robustness?**
Robustness is a model's ability to maintain high performance when subjected to perturbations or degradations not seen during training, such as compression, blur, or noise.

**62. Why test for JPEG compression robustness?**
Real-world deepfakes are distributed via social media (WhatsApp, Twitter, Facebook), which automatically apply heavy JPEG compression. A detector must survive this compression to be practically useful.

**63. How is robustness testing implemented without modifying the dataset?**
We use a wrapper class (`DegradedDataset`) that applies the specific degradation function (e.g., Gaussian Blur) dynamically on-the-fly during the test dataloader evaluation phase.

**64. What happens to CNN performance under heavy blurring?**
Performance typically degrades significantly because CNNs rely heavily on high-frequency spatial features (sharp edges, blending boundaries, pixel-level artifacts) which are destroyed by Gaussian blur.

**65. What is Grad-CAM?**
Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of a target concept flowing into the final convolutional layer to produce a coarse localization map, highlighting the regions of the image critical to the prediction.

**66. How does Grad-CAM differ from basic saliency maps?**
Basic saliency takes gradients back to the input pixels, which can be noisy. Grad-CAM uses the spatial information naturally preserved in the final convolutional feature maps, yielding smoother, more human-interpretable regions.

**67. Does Grad-CAM prove the model uses "causal reasoning"?**
No. Grad-CAM only shows correlation—regions associated with the model's activation. It highlights what the model looked at, but not necessarily the logical reasoning behind why it chose FAKE.

**68. What are common visual indicators of a deepfake?**
Inconsistent eye blinking, temporal flickering, smooth blurring around the jawline/cheeks (blending boundaries), asymmetrical lighting, and visible rendering artifacts in the teeth or hair.

**69. Why might a model produce a high-confidence False Positive?**
If a REAL face has unusual lighting, heavy makeup, severe compression artifacts, or occlusion, the model's feature extractors might misinterpret these anomalies as synthetic blending artifacts.

**70. What is Brier Score?**
The Brier Score measures the mean squared difference between predicted probabilities and actual outcomes. It evaluates both the calibration and discrimination of a probabilistic classifier.

**71. What is Expected Calibration Error (ECE)?**
ECE partitions predictions into bins based on confidence and calculates the weighted average difference between the model's confidence and its actual accuracy in each bin.

**72. Why might a highly accurate model have poor calibration?**
Deep neural networks are notoriously overconfident. They often push probabilities to 0.99 or 0.01 to aggressively minimize the cross-entropy loss, even when they are unsure.

**73. What is Temperature Scaling?**
It is a post-processing technique used to calibrate models. A single scalar (temperature) is optimized on the validation set to soften the logits before the softmax/sigmoid function, aligning confidence with accuracy without changing the rank order of predictions.

**74. What is the difference between Parameter Count and Model Size?**
Parameter count is the number of learnable weights (e.g., 25 million). Model size is the actual disk/memory space required to store them (e.g., ~100 MB for 32-bit float parameters).

**75. What limits inference latency in real-world deployment?**
Network bandwidth (transferring the video), CPU-to-GPU memory transfer bottlenecks, and the computational complexity (FLOPs) of the model architecture.

**76. What are the ethical implications of DeepShield-X?**
Deepfake detection can prevent fraud and misinformation. However, false positives could lead to false accusations of media manipulation. The system should be a decision-support tool, not an unquestionable autonomous judge.

**77. Why didn't you train a full video-level temporal model (like an LSTM or 3D CNN)?**
Video-level models require vastly more computational power and VRAM (e.g., processing 30 frames simultaneously). Due to hardware constraints, we utilized frame-level spatial detection, which remains highly effective.

**78. What is the most significant limitation of your project?**
The reliance on a single primary dataset (FF++ C23) and frame-level evaluation. Models may experience domain shift when deployed in the wild on unseen manipulation techniques.

**79. How would you improve the project as Future Work?**
Implement temporal modeling (e.g., TimeSformer or CNN+LSTM) to detect temporal flickering, train on diverse datasets like Celeb-DF and DFDC, and integrate adversarial training to improve robustness against anti-forensic attacks.

**80. Summarize your most important finding.**
Using transfer learning with proper regularization (ResNet50 / Xception) combined with strict identity-safe data splitting produces highly accurate and specific deepfake detection. Naive architectures fail completely due to class imbalance, proving that careful experimental design is as important as the model architecture itself.
