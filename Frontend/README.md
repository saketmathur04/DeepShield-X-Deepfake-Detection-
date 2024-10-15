# DeepShield Analytics

Build a beautiful, premium, research-grade frontend web application for my final-year AI/ML project:

# DeepShield-X

## Robust and Explainable DeepFake Detection Using CNNs and Vision Transformers

IMPORTANT:

This is currently FRONTEND ONLY.

Do NOT implement the actual ML model.

Do NOT invent a backend.

Do NOT create fake API integrations.

Do NOT use Supabase unless absolutely required.

Use realistic MOCK DATA and clean frontend service abstractions so I can connect my real Flask/Python backend later.

The frontend must look like a serious AI research product, not a generic college dashboard.

============================================================

PRODUCT VISION

============================================================

DeepShield-X is an AI-powered deepfake detection platform.

The future system will accept:

- images

- videos

and analyze them using trained deep-learning models.

Models in the research pipeline:

1. Custom CNN

2. ResNet50

3. EfficientNet

4. MobileNetV3

5. Xception

6. Vision Transformer (ViT)

The frontend should eventually allow users to:

- upload an image

- upload a video

- select a detection model

- run inference

- see REAL / FAKE prediction

- see confidence

- inspect probability

- view visual explanations

- view model comparisons

- inspect robustness results

- inspect evaluation metrics

- inspect error analysis

- view experiment history

============================================================

DESIGN DIRECTION

============================================================

Create a premium dark AI/research interface.

Visual identity:

Name:

DeepShield-X

Tagline:

"Robust and Explainable DeepFake Detection"

Style:

- futuristic

- sophisticated

- minimal

- scientific

- trustworthy

- high-end AI laboratory

- cybersecurity-inspired

- not overly flashy

- not a gaming dashboard

Avoid:

- generic SaaS appearance

- excessive gradients

- excessive glassmorphism

- huge glowing elements

- cartoon illustrations

- stock images

- unnecessary animations

- childish colors

Use a sophisticated dark interface with:

- near-black background

- charcoal surfaces

- subtle borders

- cool blue / cyan accent

- restrained violet accent where useful

- excellent contrast

- subtle glow only for important states

Typography should feel modern and technical.

Use a strong sans-serif font such as Inter or a similarly professional font.

============================================================

GLOBAL LAYOUT

============================================================

Desktop-first responsive application.

Layout:

------------------------------------------------

Sidebar

------------------------------------------------

DeepShield-X logo

Navigation

Overview

Detection

Experiments

Model Comparison

Robustness

Explainability

Error Analysis

Reports

About

Bottom:

System Status

"ML Engine: Offline / Mock Mode"

------------------------------------------------

Main Content

------------------------------------------------

Top bar:

Page title

Breadcrumb

Search if useful

Notifications

User/avatar placeholder

Make the sidebar collapsible.

On mobile:

- responsive drawer

- bottom navigation only if appropriate

- preserve usability

============================================================

BRAND / LOGO

============================================================

Create a simple professional DeepShield-X wordmark.

Logo concept:

shield + subtle neural-network / circuit motif.

Do not use copyrighted logos.

The logo should work in:

- sidebar

- login/landing page

- report header

============================================================

PAGE 1 — OVERVIEW / DASHBOARD

============================================================

Create a premium research dashboard.

Header:

"DeepShield-X"

"Robust and Explainable DeepFake Detection"

Subtitle:

"An experimental platform for evaluating deepfake detection models across performance, robustness, and explainability."

Hero status card:

Detection Engine

Status:

"MOCK MODE"

Later this will become:

"ONLINE"

Show a subtle status indicator.

============================================================

KEY METRIC CARDS

============================================================

Display cards for:

Models Evaluated

6

Best ROC-AUC

0.XXXX

Best F1

XX.XX%

Best Specificity

XX.XX%

Experiments

XX

IMPORTANT:

Use clearly labeled mock/demo values.

Do not present mock values as actual research results.

Add a small badge:

"DEMO DATA"

============================================================

RESEARCH SNAPSHOT

============================================================

Create a large chart section.

Show model comparison:

Model

ROC-AUC

F1

Balanced Accuracy

Specificity

Use realistic-looking demo values, clearly marked:

"Illustrative / Demo"

Do not imply these are actual measured results.

============================================================

RECENT EXPERIMENTS

============================================================

Table:

Experiment

Model

Dataset

Status

ROC-AUC

F1

Date

Example rows:

Custom CNN Baseline

Custom CNN

FF++ C23

Completed

Class Balanced CNN

Custom CNN

FF++ C23

Completed

ResNet50

ResNet50

FF++ C23

Not Run

EfficientNet

EfficientNet

FF++ C23

Not Run

ViT

ViT

FF++ C23

Not Run

Use actual known completed CNN values ONLY where appropriate:

Custom CNN baseline:

Accuracy 84.40%

ROC-AUC 0.5607

PR-AUC 0.8799

Specificity 0%

Class-balanced CNN:

Accuracy 43.24%

Precision 88.95%

Recall 37.39%

F1 52.65%

Specificity 74.87%

ROC-AUC 0.6196

PR-AUC 0.8966

Clearly label these as:

"Measured"

Do NOT invent results for models that have not been trained.

============================================================

PAGE 2 — DETECTION

============================================================

This is the primary user-facing feature.

Header:

"DeepFake Detection"

Subtitle:

"Analyze an image or video using a trained DeepShield-X model."

Create two large tabs:

IMAGE

VIDEO

============================================================

IMAGE UPLOAD

============================================================

Premium drag-and-drop upload area.

Text:

"Drop an image here"

"or browse from your device"

Supported:

JPG

JPEG

PNG

WEBP

Maximum file size indicator.

After upload:

show preview.

Display:

File name

Resolution

File size

Controls:

Model:

Dropdown:

Custom CNN

ResNet50

EfficientNet

MobileNetV3

Xception

ViT

Threshold:

0.50

Advanced options collapsed by default.

Button:

"Analyze Image"

============================================================

RESULT STATE

============================================================

Create a beautiful result card.

Example DEMO:

Prediction

FAKE

Confidence

93.7%

Probability

0.937

Use a large but tasteful REAL/FAKE indicator.

Show:

Model

ResNet50

Threshold

0.50

Inference time

128 ms

Again:

These are MOCK values until backend integration.

Clearly show:

"DEMO RESULT"

============================================================

PROBABILITY VISUALIZATION

============================================================

Create a horizontal probability bar:

REAL

6.3%

FAKE

93.7%

Use a visually clear representation.

============================================================

EXPLAINABILITY SECTION

============================================================

After prediction:

"Why did the model decide this?"

Show:

Original Image

Heatmap

Overlay

Use placeholder/demo heatmap imagery generated locally or CSS-style representation.

IMPORTANT:

Label:

"Demo visualization — Grad-CAM will be connected to the real model later."

Do not claim the displayed heatmap came from the actual model.

============================================================

VIDEO DETECTION

============================================================

Create a professional video upload interface.

After upload:

Video preview

Metadata:

Duration

Resolution

FPS

Frames analyzed

Model selection.

Button:

"Analyze Video"

Result:

Video Prediction:

FAKE

Confidence:

91.4%

Then show:

Frame-level analysis chart.

X-axis:

Frame

Y-axis:

Fake Probability

Plot a smooth line.

Show threshold = 0.50.

Below:

"Video-level decision"

Explain that future backend will aggregate frame-level predictions into a video-level result.

============================================================

PAGE 3 — EXPERIMENTS

============================================================

Research experiment management page.

Header:

"Experiments"

Filters:

Model

Dataset

Experiment status

Date

Table columns:

Experiment ID

Model

Configuration

Epochs

Augmentation

Class Weighting

Status

ROC-AUC

F1

Actions

Actions:

View

Compare

Report

Clicking View opens experiment detail.

============================================================

PAGE 4 — EXPERIMENT DETAIL

============================================================

Create a research-grade experiment page.

Header:

Experiment ID

Example:

class_balanced_custom_cnn

Status:

Completed

Model:

Custom CNN

Dataset:

FaceForensics++ C23

Configuration card:

Epochs

20

Batch Size

32

Optimizer

AdamW

Learning Rate

0.0001

Scheduler

Cosine

Class Weight

0.159215

Augmentation

Disabled

Seed

42

============================================================

METRIC GRID

============================================================

Show:

Accuracy

Balanced Accuracy

Precision

Recall

F1

Sensitivity

Specificity

MCC

ROC-AUC

PR-AUC

For the actual completed class-balanced CNN, use:

Accuracy:

43.24%

Precision:

88.95%

Recall:

37.39%

F1:

52.65%

Specificity:

74.87%

ROC-AUC:

0.6196

PR-AUC:

0.8966

Clearly indicate:

"Measured result"

============================================================

CONFUSION MATRIX

============================================================

Create a beautiful matrix:

                  Predicted

                REAL    FAKE

Actual REAL     1120    376

Actual FAKE     5067    3026

Show counts and percentages where useful.

============================================================

ROC / PR SECTION

============================================================

Create two separate polished chart cards.

ROC Curve

PR Curve

Use the measured CNN values where available.

Do not fabricate curve points.

If actual curve data isn't available, display:

"Curve data will appear after enhanced evaluation."

============================================================

TRAINING CURVES

============================================================

Create cards for:

Training Loss

Validation Loss

Training Accuracy

Validation Accuracy

Learning Rate

If actual historical data isn't available:

"Training history unavailable for this experiment."

Do not fabricate the curves.

For future API data, build components ready to consume real arrays.

============================================================

PAGE 5 — MODEL COMPARISON

============================================================

This should be one of the most visually impressive pages.

Title:

"Architecture Benchmark"

Subtitle:

"Compare CNN and Transformer architectures under a consistent evaluation protocol."

Models:

Custom CNN

ResNet50

EfficientNet

MobileNetV3

Xception

ViT

Create comparison table:

Model

Parameters

Accuracy

Balanced Accuracy

Precision

Recall

F1

Specificity

ROC-AUC

PR-AUC

MCC

Models without results should display:

—

not fake values.

Create charts:

ROC-AUC comparison

F1 comparison

Specificity comparison

Balanced Accuracy comparison

Parameter count comparison

Use professional charts.

============================================================

PAGE 6 — ROBUSTNESS

============================================================

Title:

"Robustness Analysis"

Subtitle:

"Evaluate how detection performance changes under real-world image degradation."

Create experiment cards:

Original

JPEG Compression

Gaussian Blur

Gaussian Noise

Low Resolution

Controls:

Select Model

Select Severity

Run Robustness Evaluation

Results:

Clean

JPEG

Blur

Noise

Low Resolution

Metrics:

Accuracy

F1

ROC-AUC

PR-AUC

Specificity

Create degradation-performance charts.

Clearly mark:

"Results will populate after robustness evaluation is connected."

============================================================

PAGE 7 — EXPLAINABILITY

============================================================

Title:

"Model Explainability"

Subtitle:

"Understand which facial regions influence deepfake predictions."

Create:

Model selector

Experiment selector

Sample selector

Then three image panels:

Original

Grad-CAM

Overlay

Below:

Prediction

Confidence

"Most influential regions"

Use demo placeholders until real Grad-CAM API is connected.

Explain:

"Grad-CAM visualization will be generated by the backend model pipeline."

============================================================

PAGE 8 — ERROR ANALYSIS

============================================================

Title:

"Error Analysis"

Create four tabs:

False Positives

False Negatives

Correct REAL

Correct FAKE

Each sample card:

Image

True Label

Predicted Label

Probability

Confidence

Add filters:

Confidence

Model

Error Type

Create:

High-confidence errors

section.

For example:

"High-confidence False Negatives"

This page should look like a serious ML research analysis tool.

============================================================

PAGE 9 — REPORTS

============================================================

Title:

"Research Reports"

Cards:

Experiment Report

Model Comparison Report

Robustness Report

Explainability Report

Each:

Report name

Experiment

Generated date

Status

Buttons:

View Report

Export PDF

Export CSV

For now buttons can show:

"Backend/report generator not connected yet."

Do not fake downloads.

============================================================

PAGE 10 — ABOUT

============================================================

Create an elegant project overview.

DeepShield-X

"Robust and Explainable DeepFake Detection Using CNNs and Vision Transformers"

Sections:

Research Objective

Architecture Families

Dataset

Evaluation Protocol

Robustness

Explainability

Research Workflow

Workflow visualization:

Dataset

→

Identity-Safe Split

→

Face Extraction

→

Preprocessing

→

Model Training

→

Evaluation

→

Robustness

→

Explainability

→

Error Analysis

Dataset:

FaceForensics++ C23

Labels:

REAL = 0

FAKE = 1

Mention that the project evaluates multiple architectures under a consistent protocol.

============================================================

COMPONENT SYSTEM

============================================================

Build reusable components.

Examples:

Sidebar

TopBar

MetricCard

StatusBadge

UploadZone

PredictionCard

ProbabilityBar

ConfusionMatrix

MetricGrid

ChartCard

ExperimentTable

ExperimentCard

ModelComparisonTable

ImageComparison

ErrorSampleCard

ReportCard

EmptyState

LoadingState

ErrorState

Do not duplicate UI code.

============================================================

FRONTEND DATA ARCHITECTURE

============================================================

Create a clean service abstraction for future backend integration.

For example conceptually:

src/services/api.ts

src/services/detection.ts

src/services/experiments.ts

src/services/reports.ts

Use mocked service implementations now.

The UI should NOT directly depend on hardcoded mock values everywhere.

Create typed interfaces such as:

DetectionResult

Experiment

ExperimentMetrics

ModelInfo

RobustnessResult

ExplainabilityResult

ErrorSample

Report

This is extremely important because later I will connect the actual Flask/Python backend.

============================================================

FUTURE API CONTRACT

============================================================

Design the frontend so it can later consume endpoints conceptually like:

POST /api/predict/image

POST /api/predict/video

GET /api/models

GET /api/experiments

GET /api/experiments/:id

GET /api/experiments/:id/metrics

GET /api/experiments/:id/predictions

GET /api/experiments/:id/robustness

GET /api/experiments/:id/explainability

Do NOT implement these endpoints now.

Create a mock service layer with the same conceptual response shapes.

============================================================

STATES

============================================================

Every important page should support:

Loading

Empty

Success

Error

Detection page:

Idle

Uploading

Processing

Completed

Failed

Use polished skeleton loaders.

============================================================

MICROINTERACTIONS

============================================================

Use subtle animations only:

- page transitions

- card hover

- upload drag state

- progress indicator

- result reveal

- chart entrance

Keep animations fast and professional.

Do NOT overanimate.

============================================================

ACCESSIBILITY

============================================================

Ensure:

- keyboard navigation

- visible focus states

- accessible buttons

- proper labels

- sufficient contrast

- alt text

- semantic HTML

============================================================

RESPONSIVENESS

============================================================

Desktop:

full dashboard experience.

Tablet:

adaptive grid.

Mobile:

single-column layouts

collapsible sidebar

touch-friendly controls.

============================================================

TECHNICAL REQUIREMENTS

============================================================

Use:

React

TypeScript

Tailwind CSS

shadcn/ui

Use a proper chart library already supported by the project, preferably Recharts if available.

Use Lucide icons.

Avoid unnecessary dependencies.

Use reusable typed components.

Keep code clean and maintainable.

============================================================

IMPORTANT DATA RULE

============================================================

There are currently only TWO completed CNN experiments with real measured results.

DO NOT invent results for:

ResNet50

EfficientNet

MobileNetV3

Xception

ViT

Display "Not evaluated" / "—" until real results are connected.

The following measured results are legitimate:

Original Custom CNN baseline:

Accuracy: 84.40%

Precision: 84.40%

Recall: 100.00%

F1: 91.54%

Specificity: 0%

ROC-AUC: 0.5607

PR-AUC: 0.8799

Important:

This baseline essentially predicted everything as FAKE and therefore its high accuracy/F1 are misleading because of class imbalance.

Class-balanced Custom CNN:

Accuracy: 43.24%

Precision: 88.95%

Recall: 37.39%

F1: 52.65%

Specificity: 74.87%

ROC-AUC: 0.6196

PR-AUC: 0.8966

Confusion matrix:

TN = 1120

FP = 376

FN = 5067

TP = 3026

Use these as "Measured" results.

============================================================

IMPORTANT VISUALIZATION RULE

============================================================

Never create fake ROC curves, fake training curves, fake Grad-CAM images, fake robustness results, or fake model performance.

For unavailable data, use a polished empty state:

"Evaluation data not available yet"

and explain:

"Connect the DeepShield-X evaluation API to populate this section."

============================================================

FINAL QUALITY BAR

============================================================

The result should feel like:

A professional AI research laboratory dashboard.

Imagine a combination of:

- modern cybersecurity product

- ML experiment tracking platform

- scientific visualization dashboard

- premium AI inference tool

It should be something I can confidently demonstrate to:

- professor

- project evaluator

- interviewer

- recruiter

The interface must immediately communicate:

"These people are doing serious ML experimentation."

Before finishing:

- ensure no TypeScript errors

- ensure responsive layout

- ensure navigation works

- ensure all pages render

- ensure charts render

- ensure mock service layer works

- ensure no broken buttons

- ensure no fake scientific claims

- ensure the design is consistent throughout

Do not stop at a landing page.

Build the COMPLETE frontend experience described above.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
