import os
import sys
from pathlib import Path
from fpdf import FPDF

class DeepfakeProjectReport(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=False)
        self.set_margins(14, 12, 14)
        
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(40, 60, 95)
            self.cell(110, 4.5, 'DEEPFAKE DETECTION SYSTEM (DEEPSHIELD-X) | ACADEMIC PROJECT REPORT', align='L')
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(100, 110, 120)
            self.cell(72, 4.5, 'Dept. of Computer Science & Engineering', align='R')
            self.ln(4.5)
            self.set_draw_color(190, 205, 225)
            self.set_line_width(0.3)
            self.line(14, self.get_y(), 196, self.get_y())
            self.ln(2.5)

    def footer(self):
        self.set_y(-11)
        self.set_draw_color(200, 210, 225)
        self.set_line_width(0.2)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(1.5)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(110, 115, 125)
        self.cell(90, 4, 'Academic Research & Engineering Project Evaluation', align='L')
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(50, 70, 100)
        self.cell(92, 4, f'Page {self.page_no()} of 6', align='R')

    def sec_heading(self, number, title):
        self.set_font('Helvetica', 'B', 10.5)
        self.set_text_color(18, 42, 74) # Deep Navy
        self.set_fill_color(238, 243, 250)
        text = f" {number}. {title.upper()}"
        self.cell(182, 6, text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(18, 42, 74)
        self.set_line_width(0.4)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(1.5)

    def subsec_heading(self, title):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(28, 55, 90)
        self.cell(182, 5, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def p(self, text, h=3.8):
        self.set_font('Helvetica', '', 8.2)
        self.set_text_color(30, 35, 45)
        self.multi_cell(182, h, text)
        self.ln(1)

    def bullet(self, label, text, h=3.4):
        self.set_font('Helvetica', 'B', 8.0)
        self.set_text_color(25, 45, 75)
        bw = self.get_string_width(f"- {label}: ") + 1
        self.cell(bw, h, f"- {label}: ")
        self.set_font('Helvetica', '', 8.0)
        self.set_text_color(30, 35, 45)
        self.multi_cell(182 - bw, h, text)
        self.ln(0.5)

    def box_alert(self, title, text, bg_r=244, bg_g=248, bg_b=254, border_r=30, border_g=80, border_b=150):
        start_y = self.get_y()
        self.set_fill_color(bg_r, bg_g, bg_b)
        self.set_font('Helvetica', '', 8.0)
        lines = self.multi_cell(176, 3.6, text, output="LINES")
        box_h = 5.5 + (len(lines) * 3.6) + 3
        
        self.rect(14, start_y, 182, box_h, 'F')
        # left accent bar
        self.set_draw_color(border_r, border_g, border_b)
        self.set_line_width(1.2)
        self.line(14, start_y, 14, start_y + box_h)
        # outer light border
        self.set_draw_color(210, 220, 235)
        self.set_line_width(0.3)
        self.rect(14, start_y, 182, box_h, 'D')
        
        self.set_xy(17, start_y + 1.8)
        self.set_font('Helvetica', 'B', 8.5)
        self.set_text_color(border_r, border_g, border_b)
        self.cell(176, 4.5, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(17)
        self.set_font('Helvetica', '', 8.0)
        self.set_text_color(35, 40, 50)
        self.multi_cell(176, 3.6, text)
        self.set_xy(14, start_y + box_h + 1.8)


def generate_pdf():
    pdf = DeepfakeProjectReport()
    img_dir = str(Path(__file__).resolve().parent / "images")
    
    # =========================================================================
    # PAGE 1: TITLE PAGE & TABLE OF CONTENTS
    # =========================================================================
    pdf.add_page()
    
    # Outer academic frame
    pdf.set_draw_color(24, 43, 73)
    pdf.set_line_width(0.8)
    pdf.rect(10, 10, 190, 277)
    pdf.set_draw_color(180, 195, 215)
    pdf.set_line_width(0.3)
    pdf.rect(11.5, 11.5, 187, 274)
    
    # Top Institution Header
    pdf.set_xy(14, 18)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(18, 42, 74)
    pdf.cell(182, 5, 'DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 9.5)
    pdf.set_text_color(80, 90, 105)
    pdf.cell(182, 4.5, 'ACADEMIC MAJOR PROJECT REPORT - 2025-2026', align='C', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_draw_color(24, 43, 73)
    pdf.set_line_width(0.5)
    pdf.line(40, pdf.get_y() + 2, 170, pdf.get_y() + 2)
    pdf.ln(8)
    
    # Project Title Box
    pdf.set_fill_color(240, 244, 252)
    pdf.rect(18, pdf.get_y(), 174, 30, 'F')
    pdf.set_draw_color(30, 70, 130)
    pdf.set_line_width(0.6)
    pdf.rect(18, pdf.get_y(), 174, 30, 'D')
    
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(18, 42, 74)
    pdf.cell(182, 7, 'DEEPFAKE DETECTION SYSTEM', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(40, 70, 120)
    pdf.cell(182, 5.5, 'Project DeepShield-X: Robust and Explainable Facial Manipulation Forensics', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', 'I', 8.5)
    pdf.set_text_color(90, 100, 115)
    pdf.cell(182, 4.5, 'Evaluation of CNNs & Vision Transformers under Identity-Disjoint Protocols', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    
    # Student & Supervisor Metadata Table
    pdf.set_xy(22, pdf.get_y())
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(85, 5, 'SUBMITTED BY:', align='L')
    pdf.cell(85, 5, 'SUPERVISED BY:', align='L', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_xy(22, pdf.get_y())
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(40, 45, 55)
    pdf.cell(85, 4.5, 'Name: Saket', align='L')
    pdf.cell(85, 4.5, 'Project Evaluation Committee', align='L', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_xy(22, pdf.get_y())
    pdf.cell(85, 4.5, 'Enrollment No: E23CSEU1361', align='L')
    pdf.cell(85, 4.5, 'Department of Computer Science & Engg.', align='L', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_xy(22, pdf.get_y())
    pdf.cell(85, 4.5, 'Degree: Bachelor of Technology (B.Tech)', align='L')
    pdf.cell(85, 4.5, 'Institution: Engineering College / University', align='L', new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(6)
    
    # Empirical Integrity Banner
    pdf.box_alert("EMPIRICAL RESEARCH INTEGRITY STATEMENT",
                  "All metrics, latency profiles, and error breakdowns reported in this document are strictly measured from "
                  "actual checkpointed models on an isolated test partition (N = 9,589 crops) derived from FaceForensics++ C23. "
                  "No values have been fabricated, estimated, or artificially smoothed. Experiments are reproducible with fixed seed (S=42).",
                  bg_r=242, bg_g=247, bg_b=254, border_r=25, border_g=70, border_b=140)
    
    pdf.ln(2)
    
    # TABLE OF CONTENTS
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(18, 42, 74)
    pdf.set_fill_color(235, 240, 248)
    pdf.cell(182, 6, ' TABLE OF CONTENTS', fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(18, 42, 74)
    pdf.set_line_width(0.3)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(2)
    
    toc_items = [
        ("1. Abstract & Executive Summary", "Overview, core problem, key research findings, and champion architecture", "Page 2"),
        ("2. Introduction & Problem Statement", "Deepfake technology overview, security risks, and four core evaluation bottlenecks", "Page 2"),
        ("3. Project Objectives", "Primary engineering and scientific milestones established for DeepShield-X", "Page 2"),
        ("4. Proposed Methodology & Architecture", "End-to-end pipeline: face cropping, Union-Find identity isolation, and backbones", "Page 3"),
        ("5. Novelty & Key Contributions", "CRITICAL VIVA SECTION: 5 genuine contributions + Student Explanation Script", "Page 4"),
        ("6. Results & Discussion", "Master benchmark (6 models), class imbalance ablation, NeuralTextures failure audit", "Page 5"),
        ("7. Comparative Analysis with Research", "Systematic 6-paper comparative matrix with verified 2021-2023 literature", "Page 6"),
        ("8. Conclusion, Limitations & Future Scope", "Deployment recommendations, operational constraints, and frequency-domain roadmap", "Page 6"),
        ("9. IEEE References", "Curated and verified peer-reviewed academic citations", "Page 6"),
    ]
    
    for sec_title, desc, pg in toc_items:
        pdf.set_font('Helvetica', 'B', 8.2)
        pdf.set_text_color(20, 45, 80)
        pdf.cell(58, 4.6, sec_title, border='B')
        pdf.set_font('Helvetica', '', 7.6)
        pdf.set_text_color(70, 75, 85)
        pdf.cell(106, 4.6, desc, border='B')
        pdf.set_font('Helvetica', 'B', 8.0)
        pdf.set_text_color(30, 60, 110)
        pdf.cell(18, 4.6, pg, border='B', align='R', new_x="LMARGIN", new_y="NEXT")
    
    # Bottom date / metadata
    pdf.set_xy(14, 274)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(120, 125, 135)
    pdf.cell(182, 4, 'Final Evaluation Copy | Deepfake Detection System Documentation | Project DeepShield-X', align='C')

    # =========================================================================
    # PAGE 2: ABSTRACT, INTRODUCTION & OBJECTIVES
    # =========================================================================
    pdf.add_page()
    
    # SECTION 1: ABSTRACT
    pdf.sec_heading("1", "Abstract & Executive Summary")
    pdf.p(
        "Hyper-realistic synthetic facial media ('deepfakes') generated via autoencoders, generative adversarial networks "
        "(GANs), and diffusion models have rendered visual authenticity verification acutely difficult, threatening digital forensics, "
        "courtroom evidence, journalism, and biometric authentication. Although modern deep neural networks frequently claim "
        ">95% detection accuracy on academic benchmarks, these reported scores frequently suffer from two severe methodological "
        "flaws: (i) identity leakage between training and testing partitions, where models memorize facial biometric features rather than "
        "forgery boundaries, and (ii) extreme class imbalance, which deceitfully inflates raw accuracy metrics. In this project (DeepShield-X), "
        "we design, implement, and benchmark a complete deepfake detection framework on the compressed FaceForensics++ (FF++ C23) "
        "dataset. Sequence splitting is formulated as a graph connected-components problem solved via disjoint-set Union-Find, formally "
        "guaranteeing zero identity leakage across 69,273 standardized facial crops (train = 50,706; validation = 8,978; held-out test = 9,589)."
    )
    pdf.p(
        "We systematically compare six neural architectures: a bespoke 4-stage Custom CNN, ResNet50, Xception, EfficientNet-B0, "
        "MobileNetV3-Large, and Vision Transformer (ViT-B/16). A controlled ablation confirms that an unweighted Custom CNN collapses "
        "to predicting the majority fake class, yielding 84.40% accuracy but 0.00% specificity (failing on every real face). Implementing positive-class "
        "weighting (w_pos = 0.1592) and multi-stage regularization enables ResNet50 Regularized to achieve champion performance (F1 = 0.9633, "
        "ROC-AUC = 0.9718, PR-AUC = 0.9938, Specificity = 87.10%, MCC = 0.7821), closely followed by Xception (F1 = 0.9621, Specificity = 88.30%). "
        "Sub-category audits expose NeuralTextures as the primary failure mode (55.7% of all false negatives). Probabilistic calibration (ECE = 4.18%) "
        "and Grad-CAM heatmaps confirm predictions are grounded in authentic jawline and perioral manipulation boundaries."
    )
    
    # SECTION 2: INTRODUCTION & PROBLEM STATEMENT
    pdf.sec_heading("2", "Introduction & Problem Statement")
    pdf.p(
        "Creating convincing manipulated facial imagery previously required specialized digital VFX artists and extensive computation. "
        "Today, automated synthesis pipelines can generate deceptive face swaps and reenactments in minutes. The societal risks are "
        "profound: financial fraud via synthetic executive video calls, defamatory media targeting private citizens, fabricated geopolitical "
        "announcements, and spoofed biometric face-recognition check-ins. To build trustworthy defense mechanisms, automated forensic "
        "classifiers must analyze high-frequency spatial gradients and localized blending boundaries."
    )
    pdf.p(
        "However, our systematic review of current deepfake detection research reveals four critical scientific bottlenecks:"
    )
    
    pdf.bullet("1. Identity & Subject Leakage", 
               "Benchmark datasets apply multiple manipulation algorithms to a shared pool of original videos. When data is split randomly "
               "or naively by video files, the same actor's face appears in both training and test subsets. The network simply memorizes facial identities "
               "rather than learning synthesis artifacts, causing up to a 10% artificial performance inflation (Dong et al., 2023).")
    pdf.bullet("2. Pathological Class Imbalance", 
               "Forensic benchmarks contain roughly 5 to 6 manipulated videos for every authentic video. Under standard unweighted cross-entropy "
               "loss, an uninformative model that trivially outputs 'fake' for every input achieves 84.40% accuracy while missing 100% of authentic faces.")
    pdf.bullet("3. Black-Box Decision Opacity", 
               "Standard classifiers output a single probability score without visual grounding, preventing forensic analysts from verifying whether "
               "predictions stem from genuine face-boundary artifacts or background image shortcuts.")
    pdf.bullet("4. Social Media Compression Cliff", 
               "Platforms apply aggressive lossy compression (H.264, JPEG) that severely attenuates high-frequency forensic signals, causing steep "
               "detector degradation under low-quality transmissions.")
    
    # SECTION 3: OBJECTIVES
    pdf.sec_heading("3", "Project Objectives")
    pdf.p("To resolve these bottlenecks, DeepShield-X was engineered to achieve six concrete objectives:")
    
    # Table of objectives
    pdf.set_font('Helvetica', 'B', 7.8)
    pdf.set_fill_color(240, 244, 252)
    pdf.set_text_color(18, 42, 74)
    pdf.cell(16, 5, 'Obj ID', border=1, fill=True, align='C')
    pdf.cell(52, 5, 'Target Milestone', border=1, fill=True, align='L')
    pdf.cell(96, 5, 'Technical Approach / Engineering Implementation', border=1, fill=True, align='L')
    pdf.cell(18, 5, 'Status', border=1, fill=True, align='C', new_x="LMARGIN", new_y="NEXT")
    
    objs = [
        ("OBJ-1", "Identity-Safe Splitting Protocol", "Build bipartite video graph; use Disjoint-Set Union-Find for atomic zero-leakage splits", "Verified"),
        ("OBJ-2", "Multi-Architecture Benchmark", "Benchmark 6 architectures across CNN, depthwise separable, compound, and ViT paradigms", "Complete"),
        ("OBJ-3", "Class Imbalance Resolution", "Analytically derive BCE stationary point; implement loss weighting w_pos = 0.1592", "Complete"),
        ("OBJ-4", "Granular Failure Audits", "Evaluate error distributions across 6 distinct manipulation algorithms on 9,589 test crops", "Complete"),
        ("OBJ-5", "Reliability & Explainability", "Compute 10-bin Expected Calibration Error (ECE) and extract layer-wise Grad-CAM heatmaps", "Complete"),
        ("OBJ-6", "Perturbation Stress-Testing", "Evaluate accuracy retention across JPEG sweeps (Q=90 to 30), noise, blur, and scale", "Complete"),
    ]
    pdf.set_font('Helvetica', '', 7.5)
    for j, (oid, mil, app, st) in enumerate(objs):
        fill = (j % 2 == 1)
        pdf.set_fill_color(248, 250, 254)
        pdf.cell(16, 4.4, oid, border=1, fill=fill, align='C')
        pdf.set_font('Helvetica', 'B', 7.5)
        pdf.cell(52, 4.4, mil, border=1, fill=fill, align='L')
        pdf.set_font('Helvetica', '', 7.5)
        pdf.cell(96, 4.4, app, border=1, fill=fill, align='L')
        pdf.set_font('Helvetica', 'B', 7.5)
        pdf.set_text_color(20, 110, 40)
        pdf.cell(18, 4.4, st, border=1, fill=fill, align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(30, 35, 45)

    # =========================================================================
    # PAGE 3: PROPOSED METHODOLOGY & SYSTEM ARCHITECTURE
    # =========================================================================
    pdf.add_page()
    pdf.sec_heading("4", "Proposed Methodology & System Architecture")
    pdf.p(
        "DeepShield-X executes a strictly pipelined forensic workflow designed to eliminate data contamination, handle class imbalance, "
        "and provide interpretable spatial predictions. The architecture decouples data curation, modeling, and diagnostic auditing."
    )
    
    # Embed Architecture Pipeline Image if exists
    arch_img_path = os.path.join(img_dir, "architecture_pipeline.png")
    if os.path.exists(arch_img_path):
        # Draw image neatly
        pdf.image(arch_img_path, x=15, y=pdf.get_y(), w=180, h=36)
        pdf.set_y(pdf.get_y() + 37.5)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(80, 90, 105)
        pdf.cell(182, 3.5, 'Figure 1: End-to-end DeepShield-X forensic pipeline: Graph Union-Find data isolation, standardized cropping, multi-model evaluation, and diagnostics.', align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    
    pdf.subsec_heading("4.1 Dataset Curation & Graph Union-Find Partitioning")
    pdf.p(
        "We utilize the FaceForensics++ (FF++ C23) benchmark, containing 1,000 authentic pristine video sequences manipulated by four primary "
        "algorithms (Deepfakes, Face2Face, FaceSwap, NeuralTextures), alongside FaceShifter and DeepFakeDetection. The H.264 C23 (CRF=23) "
        "tier is chosen because its moderate compression accurately mimics modern video distribution platforms (YouTube, Twitter, WhatsApp)."
    )
    pdf.p(
        "To enforce zero identity leakage, we construct an undirected bipartite graph G = (V, E) where vertices V denote unique video identifiers "
        "and edges E represent manipulation donor-target pairings. Disjoint-Set Union-Find with path compression and union-by-rank groups "
        "all transitively linked videos into isolated connected components. Entire components are assigned atomically to training, validation, or test "
        "partitions, guaranteeing that no tracked actor's face crosses split boundaries: Identities(Train) INTERSECT Identities(Test) = EMPTY."
    )
    
    # Dataset Table
    pdf.set_font('Helvetica', 'B', 7.8)
    pdf.set_fill_color(240, 244, 252)
    pdf.set_text_color(18, 42, 74)
    dw = [42, 32, 36, 36, 36]
    for i, h in enumerate(['Partition', 'Authentic (Real)', 'Manipulated (Fake)', 'Total Face Crops', 'Class Ratio (F:R)']):
        pdf.cell(dw[i], 4.8, h, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 7.8)
    pdf.set_text_color(30, 35, 45)
    for j, r in enumerate([
        ['Training Set (Train)', '6,964 (13.7%)', '43,742 (86.3%)', '50,706 (73.2%)', '6.28 : 1'],
        ['Validation Set (Val)', '1,497 (16.7%)', '7,481 (83.3%)', '8,978 (13.0%)', '5.00 : 1'],
        ['Frozen Test Set (Test)', '1,496 (15.6%)', '8,093 (84.4%)', '9,589 (13.8%)', '5.41 : 1'],
    ]):
        fill = (j % 2 == 1)
        pdf.set_fill_color(248, 250, 254)
        for i, val in enumerate(r):
            pdf.cell(dw[i], 4.4, val, border=1, fill=fill, align='C')
        pdf.ln()
    pdf.set_font('Helvetica', 'B', 7.8)
    pdf.set_fill_color(230, 236, 246)
    for i, val in enumerate(['Total Verified Data', '9,957 (14.4%)', '59,316 (85.6%)', '69,273 (100.0%)', '5.96 : 1']):
        pdf.cell(dw[i], 4.4, val, border=1, fill=True, align='C')
    pdf.ln(2.5)

    pdf.subsec_heading("4.2 Preprocessing & 1.3x Expanded Face Extraction")
    pdf.p(
        "Video frames are extracted at a uniform stride of 10 to suppress redundant temporal autocorrelation. Facial landmarks are detected, "
        "and bounding boxes are expanded by a scale factor of alpha = 1.30 (30% perimeter expansion). This expansion is mathematically crucial: "
        "standard tight crops eliminate the jawline, chin, and forehead borders where Poisson image blending seams and color interpolation "
        "discontinuities reside. Crops are resized to 224x224x3 using bilinear interpolation and standardized with ImageNet RGB parameters."
    )

    pdf.subsec_heading("4.3 Evaluated Neural Backbone Architectures")
    pdf.p(
        "To study contrasting inductive biases, we benchmark six models: (1) Custom CNN: 4 Conv-BatchNorm-ReLU-MaxPool stages (94K params, "
        "trained from scratch); (2) ResNet50: 23.5M params, residual shortcuts preventing gradient vanishing; (3) Xception: 20.8M params, depthwise "
        "separable convolutions decoupling spatial and cross-channel color correlation; (4) EfficientNet-B0: 4.0M params, NAS-optimized compound "
        "scaling; (5) MobileNetV3-Large: 1.5M params, hardware-aware edge backbone; (6) ViT-B/16: 21.7M params, 14x14 non-overlapping patch self-attention."
    )

    pdf.subsec_heading("4.4 Optimization & Multi-Stage Regularization")
    pdf.p(
        "Models output a scalar logit z in R. To prevent majority collapse under the 6.28:1 ratio, we apply positive-class loss weighting: "
        "w_pos = N_real / N_fake = 6,964 / 43,742 = 0.1592 in binary cross-entropy with logits. Optimization uses AdamW (lr = 1e-4, weight decay = 1e-4) "
        "with cosine annealing and 2-epoch warmup. Regularization includes random horizontal flips (p=0.5), rotation (+/-15 deg), brightness/contrast "
        "jitter, dropout (p=0.50), and checkpointing strictly on validation ROC-AUC (patience = 5). Gaussian blur and JPEG augmentation are intentionally "
        "omitted during training to protect subtle forensic edge cues."
    )

    # =========================================================================
    # PAGE 4: NOVELTY & KEY CONTRIBUTIONS (MOST IMPORTANT SECTION)
    # =========================================================================
    pdf.add_page()
    pdf.sec_heading("5", "Novelty and Key Contributions")
    pdf.p(
        "A critical question in forensic system evaluation is distinguishing genuine research contributions from standard deep learning pipelines. "
        "Merely applying transfer learning, running standard CNNs/Transformers, or loading the public FaceForensics++ benchmark does NOT "
        "constitute scientific novelty. DeepShield-X contributes five distinct, verifiable methodological and empirical innovations designed "
        "to solve systemic vulnerabilities identified in prior published forensic literature."
    )
    
    # 5 Novelty Points
    pdf.bullet("Novelty 1: Graph-Connected Disjoint-Set Identity Isolation Protocol",
               "Prior studies frequently employ naive video-level or random frame splitting, which allows identical actors to appear in both "
               "train and test partitions through manipulation donor-target pairings. We formulated dataset partitioning as a graph connected-components "
               "problem solved via Disjoint-Set Union-Find with path compression and union-by-rank. By assigning entire multi-edge clusters atomically, "
               "we provide a mathematically verifiable guarantee of zero identity leakage across 69,273 face crops, ensuring models learn genuine "
               "forgery artifacts rather than facial biometric memorization.")
    
    pdf.bullet("Novelty 2: Analytical Derivation & Controlled Empirical Proof of Imbalance Collapse",
               "We mathematically derived from the BCE gradient expectation stationary condition (E[dL/dz] = sigma(z) - pi_1 = 0) that an "
               "unweighted model optimizes training risk by producing a constant positive logit (z* = +1.837), causing complete majority-class collapse. "
               "We then proved this experimentally: an unweighted CNN achieves 84.40% accuracy but exactly 0.00% specificity (fails on every real face). "
               "This concrete proof demonstrates why raw accuracy is an invalid forensic metric and establishes MCC as the primary evaluation standard.")
    
    pdf.bullet("Novelty 3: Discovery & Attribution of the NeuralTextures Blind Spot",
               "Through granular sub-category auditing across 9,589 frozen test samples, we uncovered that NeuralTextures accounts for 55.7% "
               "(219 of 393) of all false negatives across the entire benchmark. We identified the root cause: deferred neural rendering modifies only "
               "the mouth cavity, and Global Average Pooling (GAP) in standard backbones dilutes these localized signals across the vast authentic face. "
               "This isolates a concrete architectural deficiency in standard vision backbones for facial reenactment detection.")
    
    pdf.bullet("Novelty 4: Empirical Quantification of the JPEG Compression Cliff",
               "We performed multi-tier perturbation stress-testing (N = 502) mapping accuracy retention across compression rates. While spatial models "
               "remain resilient to additive noise (99.4%) and blur (100%), we identified a non-linear performance cliff below JPEG Quality 50, where "
               "recall collapses from 96.41% (Q=70) to 39.64% (Q=30). This quantifies the exact boundary where lossy quantization destroys high-frequency "
               "Poisson boundary gradients, defining operational limits for social media screening.")
    
    pdf.bullet("Novelty 5: Explainability & Probabilistic Calibration Grounding",
               "Unlike black-box classifiers, DeepShield-X integrates 10-bin Expected Calibration Error analysis (achieving ECE = 4.18%, Brier = 0.0524) "
               "demonstrating near-perfect reliability at extreme operational bins (gap < 0.008). Layer-wise Grad-CAM heatmaps verify that ResNet50 "
               "attends specifically to facial boundary seams for face-swap fakes and the perioral cavity for reenactments, proving forensic grounding.")

    pdf.ln(2)
    
    # STUDENT VIVA EXPLANATION BOX (Requested by user!)
    viva_script = (
        "\"Respected Examiner, the core novelty of DeepShield-X is not simply training a CNN on images. Our primary contribution is solving "
        "the two biggest methodological flaws in deepfake research: identity leakage and misleading accuracy. First, we implemented a graph Union-Find "
        "protocol that mathematically guarantees no actor's face appears in both training and testing across 69,273 crops, preventing facial memorization. "
        "Second, we proved both mathematically and experimentally that an unweighted model scores 84.40% accuracy while failing 100% of authentic faces, "
        "and we solved this using positive-class weighting. Third, our error audit revealed that 55.7% of all detection failures come from NeuralTextures "
        "because mouth modifications get washed out by global pooling, and we mapped a severe compression cliff below JPEG Q=50. Thus, our work delivers "
        "an empirically verified, leakage-free forensic benchmark with actionable failure insights.\""
    )
    pdf.box_alert("HOW TO EXPLAIN THE NOVELTY TO THE TEACHER (VIVA DEFENSE SCRIPT)", viva_script,
                  bg_r=254, bg_g=250, bg_b=240, border_r=180, border_g=110, border_b=20)
    
    pdf.ln(1)
    
    # Comparative Novelty Matrix
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_fill_color(240, 244, 252)
    pdf.set_text_color(18, 42, 74)
    nw = [38, 24, 24, 24, 24, 24, 24]
    for i, h in enumerate(['Evaluation Dimension', 'Standard Baselines', 'ID-Reveal (2021)', 'SBI (2022)', 'Dong et al. (2023)', 'DeepfakeBench (2023)', 'DeepShield-X (Ours)']):
        pdf.cell(nw[i], 5, h, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 7.3)
    pdf.set_text_color(30, 35, 45)
    novelty_table = [
        ['Identity-Disjoint Split', 'Random (Leaky)', 'Partial Video', 'Synthetic Pairs', 'Verified Disjoint', 'Video-level Only', 'Graph Union-Find Verified'],
        ['Imbalance Derivation', 'None (Ignored)', 'None', 'None', 'None', 'None', 'Mathematical Proof + Ablation'],
        ['Multi-Architecture Suite', 'Single Model', 'Single Backbone', 'Single Backbone', 'ResNet / Xception', '15+ Baselines', '6 Architectures (CNN to ViT)'],
        ['Sub-Category Failure Audit', 'None', 'None', 'None', 'None', 'Partial', 'Granular 6-Family Breakdown'],
        ['Compression Cliff Profile', 'None', 'None', 'H.264 Only', 'None', 'H.264 Tiers', 'JPEG Sweeps (Q=90 to 30)'],
        ['Spatial Explainability', 'Black-Box', 'Temporal Emb.', 'Mask Blending', 'Feature Norm.', 'AUC Only', 'Grad-CAM + 10-Bin ECE'],
    ]
    for j, row in enumerate(novelty_table):
        fill = (j % 2 == 1)
        pdf.set_fill_color(248, 250, 254)
        for i, val in enumerate(row):
            if i == 6:
                pdf.set_font('Helvetica', 'B', 7.3)
                pdf.set_text_color(18, 42, 74)
            else:
                pdf.set_font('Helvetica', '', 7.3)
                pdf.set_text_color(40, 45, 55)
            pdf.cell(nw[i], 3.8, val, border=1, fill=fill, align='C')
        pdf.ln()

    # =========================================================================
    # PAGE 5: RESULTS AND DISCUSSION
    # =========================================================================
    pdf.add_page()
    pdf.sec_heading("6", "Results and Discussion")
    pdf.p(
        "All models were evaluated on the strictly frozen, held-out test partition (N = 9,589; Authentic Real = 1,496; Manipulated Fake = 8,093) "
        "under a fixed decision threshold tau = 0.50 without test-set tuning. Table 1 summarizes complete benchmark metrics."
    )
    
    # MASTER BENCHMARK TABLE
    pdf.subsec_heading("Table 1: Master Multi-Architecture Benchmark on Frozen FF++ C23 Test Set (N = 9,589)")
    pdf.set_font('Helvetica', 'B', 7.2)
    pdf.set_fill_color(20, 50, 95)
    pdf.set_text_color(255, 255, 255)
    bw = [38, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    for i, h in enumerate(['Architecture', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'ROC-AUC', 'PR-AUC', 'Bal. Acc.', 'MCC']):
        pdf.cell(bw[i], 5, h, border=1, fill=True, align='C')
    pdf.ln()
    
    res_rows = [
        ['ResNet50 Regularized *', '0.9389', '0.9755', '0.9514', '0.8710', '0.9633', '0.9718', '0.9938', '0.9112', '0.7821'],
        ['Xception', '0.9370', '0.9777', '0.9470', '0.8830', '0.9621', '0.9706', '0.9931', '0.9150', '0.7798'],
        ['EfficientNet-B0', '0.9208', '0.9721', '0.9330', '0.8549', '0.9521', '0.9547', '0.9879', '0.8940', '0.7290'],
        ['Vision Transformer (ViT-B/16)', '0.9005', '0.9663', '0.9140', '0.8275', '0.9394', '0.9512', '0.9894', '0.8708', '0.6706'],
        ['MobileNetV3-Large', '0.8797', '0.9444', '0.9110', '0.7099', '0.9274', '0.9187', '0.9820', '0.8105', '0.5792'],
        ['ResNet50 Baseline (Unregularized)', '0.9133', '0.9709', '0.9250', '0.8503', '0.9474', '0.9578', '0.9914', '0.8876', '0.7087'],
        ['Custom CNN (Class-Balanced)', '0.4324', '0.8895', '0.3739', '0.7487', '0.5265', '0.6196', '0.8966', '0.5613', '0.0987'],
        ['Custom CNN (Unweighted Baseline)', '0.8440', '0.8440', '1.0000', '0.0000', '0.9154', '0.5607', '0.8799', '0.5000', '0.0000'],
    ]
    for j, r in enumerate(res_rows):
        is_champ = (j == 0)
        is_bad = (j == 7)
        if is_champ:
            pdf.set_font('Helvetica', 'B', 7.2)
            pdf.set_fill_color(230, 242, 255)
            pdf.set_text_color(15, 45, 90)
        elif is_bad:
            pdf.set_font('Helvetica', 'I', 7.0)
            pdf.set_fill_color(255, 235, 235)
            pdf.set_text_color(160, 30, 30)
        else:
            pdf.set_font('Helvetica', '', 7.2)
            if j % 2 == 0:
                pdf.set_fill_color(250, 252, 255)
            else:
                pdf.set_fill_color(242, 246, 252)
            pdf.set_text_color(30, 35, 45)
        for i, val in enumerate(r):
            pdf.cell(bw[i], 4.3, val, border=1, fill=True, align='C')
        pdf.ln()
    pdf.ln(1.5)

    # TWO COLUMNS: ABLATION & MANIPULATION BREAKDOWN
    pdf.subsec_heading("6.1 Controlled Class-Imbalance & Regularization Ablations")
    pdf.p(
        "Table 1 demonstrates the critical importance of loss weighting. The Unweighted Custom CNN achieves an apparent accuracy of 84.40% "
        "and F1 of 0.9154, but possesses Specificity = 0.0000 and MCC = 0.0000. It produces 8,093 True Positives and 1,496 False Positives, classifying "
        "every authentic face as fake. Implementing positive-class weighting (w_pos = 0.1592) forces gradient equilibrium, restoring specificity to 74.87%. "
        "On ResNet50, multi-stage regularization (spatial augmentation, dropout 0.50, ROC-AUC early stopping) improves Accuracy by +2.56%, F1 by +1.59%, "
        "Specificity by +2.07%, and MCC by +7.34% (0.7087 to 0.7821), proving regularization prevents identity memorization."
    )
    
    # SUB-CATEGORY BREAKDOWN TABLE
    pdf.subsec_heading("Table 2: ResNet50 Regularized Performance Across Manipulation Pipelines (N = 9,589)")
    pdf.set_font('Helvetica', 'B', 7.2)
    pdf.set_fill_color(240, 244, 252)
    pdf.set_text_color(18, 42, 74)
    mw = [36, 22, 24, 28, 28, 24, 20]
    for i, h in enumerate(['Manipulation Category', 'Ground Truth', 'Test Samples', 'Recall / Specificity', 'Mean Pred. P(fake)', 'Errors', 'Error Share']):
        pdf.cell(mw[i], 4.6, h, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 7.2)
    subcat_rows = [
        ['DeepFakeDetection', 'FAKE', '615', '99.67% (Recall)', '0.9889', 'FN = 2', '0.5%'],
        ['FaceSwap', 'FAKE', '1,496', '97.33% (Recall)', '0.9658', 'FN = 40', '10.2%'],
        ['Deepfakes', 'FAKE', '1,492', '97.32% (Recall)', '0.9657', 'FN = 40', '10.2%'],
        ['Face2Face', 'FAKE', '1,499', '97.00% (Recall)', '0.9622', 'FN = 45', '11.5%'],
        ['FaceShifter', 'FAKE', '1,495', '96.86% (Recall)', '0.9568', 'FN = 47', '12.0%'],
        ['NeuralTextures (Blind Spot)', 'FAKE', '1,496', '85.36% (Recall)', '0.8363', 'FN = 219', '55.7% !'],
        ['Original (Authentic Real)', 'REAL', '1,496', '87.10% (Specificity)', '0.1497', 'FP = 193', '--'],
    ]
    for j, r in enumerate(subcat_rows):
        is_nt = (j == 5)
        if is_nt:
            pdf.set_font('Helvetica', 'B', 7.2)
            pdf.set_fill_color(255, 238, 238)
            pdf.set_text_color(170, 30, 30)
        else:
            pdf.set_font('Helvetica', '', 7.2)
            if j % 2 == 1:
                pdf.set_fill_color(248, 250, 254)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 35, 45)
        for i, val in enumerate(r):
            pdf.cell(mw[i], 4.2, val, border=1, fill=True, align='C')
        pdf.ln()
    pdf.ln(1.5)

    pdf.subsec_heading("6.2 Key Scientific Insights & Diagnostic Audits")
    pdf.p(
        "1. CNNs Outperform Vision Transformers: ResNet50 (F1=0.9633) and Xception (F1=0.9621) substantially outperformed ViT-B/16 (F1=0.9394). "
        "Facial manipulation boundaries (Poisson stitching seams, blending artifacts) are local, high-frequency spatial gradients. The translation "
        "equivariance of convolutional kernels naturally captures these sharp edges, whereas ViT's fixed 16x16 patch projection tokenizes and fragments "
        "continuous boundary lines across adjacent patch borders."
    )
    pdf.p(
        "2. The NeuralTextures Vulnerability: Out of 393 total false negatives across all 9,589 test samples, 219 (55.7%) occurred on NeuralTextures. "
        "Because NeuralTextures limits facial reenactment strictly to the mouth interior, standard Global Average Pooling aggregates oral anomalies "
        "with the remaining 90% unedited facial skin, severely attenuating the discriminative gradient signal."
    )
    pdf.p(
        "3. Perturbation Robustness & JPEG Cliff: Testing on N=502 synthetic crops revealed remarkable stability under Gaussian blur (100% recall), "
        "additive Gaussian noise (99.4%), and 4x downscale-upscale cycles (100%). However, performance plunged sharply from 96.41% at JPEG Q=70 "
        "to 72.51% at Q=50, collapsing to 39.64% at Q=30, indicating spatial detectors depend on fine-grained DCT coefficients eliminated by heavy quantization."
    )
    pdf.p(
        "4. Calibration & Grad-CAM: ResNet50 achieves Expected Calibration Error (ECE) = 4.18% and Brier score = 0.0524. 80.6% of test predictions reside "
        "in the [0.9, 1.0] confidence bin with a calibration gap of only 0.0074. Grad-CAM confirms attention focuses on the jawline and chin for face-swaps "
        "and perioral regions for reenactments, verifying predictions are anchored in physical manipulation artifacts rather than spurious background cues."
    )

    # =========================================================================
    # PAGE 6: COMPARATIVE ANALYSIS, CONCLUSION, REFERENCES
    # =========================================================================
    pdf.add_page()
    pdf.sec_heading("7", "Comparative Analysis with Existing Research")
    pdf.p(
        "To position DeepShield-X within current academic research, Table 3 compares our system against seven prominent deepfake detection papers "
        "published between 2021 and 2023. Unlike prior works that address isolated aspects, DeepShield-X unifies identity isolation, class weighting, "
        "multi-architecture benchmarking, granular error auditing, and spatial explainability into a reproducible experimental pipeline."
    )
    
    # COMPARATIVE TABLE
    pdf.subsec_heading("Table 3: Systematic Comparison with Key Recent Peer-Reviewed Deepfake Detection Literature (2021-2023)")
    pdf.set_font('Helvetica', 'B', 7.0)
    pdf.set_fill_color(20, 50, 95)
    pdf.set_text_color(255, 255, 255)
    cw = [24, 10, 24, 20, 28, 30, 46]
    for i, h in enumerate(['Model / Paper', 'Year', 'Detection Paradigm', 'Benchmark', 'Core Strength', 'Primary Limitation', 'How DeepShield-X Relates / Differs']):
        pdf.cell(cw[i], 5, h, border=1, fill=True, align='C')
    pdf.ln()
    
    comp_data = [
        ['ID-Reveal [7]', '2021', 'Temporal Biometric Representation', 'FF++ (C23, C40)', 'Robust to facial reenactment; identity-aware', 'Requires temporal video; partial identity split', 'We enforce formal graph Union-Find disjointness on static single frames'],
        ['Multi-Attn [8]', '2021', 'Region Multi-Attention CNN', 'FF++, Celeb-DF', 'Attends to localized facial components', 'High computational cost; no imbalance analysis', 'We diagnose why GAP causes failure on NeuralTextures and benchmark 6 models'],
        ['LipForensics [9]', '2021', 'Spatio-temporal Mouth Dynamics', 'FF++, DFDC', 'High generalization across unseen face swaps', 'Limited to speaking faces; misses static images', 'We evaluate full-face boundary cues and reveal oral cavity failure under static pooling'],
        ['SBI [10]', '2022', 'Self-Blended Boundary Augmentation', 'FF++, CDF, DFD', 'Superior out-of-distribution transfer', 'Synthesizes synthetic artifacts; no identity audit', 'We benchmark empirical architectures under real-world C23 social media compression'],
        ['Cross-Eff-ViT [11]', '2022', 'EfficientNet + ViT Hybrid', 'FF++, DFDC', 'Combines local CNN & global ViT tokens', 'High parameter count; untested class imbalance', 'We isolate CNN vs ViT backbones directly and prove pure CNNs lead on C23 media'],
        ['Dong et al. [12]', '2023', 'Implicit Identity Leakage Proof', 'FF++, Celeb-DF', 'Formal mathematical proof of leakage bias', 'Focuses only on proof; no full forensic suite', 'We solve the leakage problem via graph Union-Find and benchmark 6 architectures'],
        ['DeepfakeBench [13]', '2023', 'Unified Detection Benchmark', 'FF++, CDF, DFDC', 'Standardized benchmark of 15+ detectors', 'Uses naive video splits; no imbalance derivation', 'We add Union-Find identity isolation and provide explicit class imbalance proofs'],
        ['DeepShield-X (Ours)', '2026', 'Identity-Disjoint Forensic Suite', 'FF++ C23 (69K)', 'Unified 6-dimension experimental investigation', 'Frame-level scope; single benchmark evaluation', 'First unified framework combining Union-Find, imbalance proof, & 6-arch suite'],
    ]
    for j, r in enumerate(comp_data):
        is_ours = (j == len(comp_data) - 1)
        if is_ours:
            pdf.set_font('Helvetica', 'B', 6.8)
            pdf.set_fill_color(225, 238, 255)
            pdf.set_text_color(15, 45, 90)
        else:
            pdf.set_font('Helvetica', '', 6.8)
            if j % 2 == 1:
                pdf.set_fill_color(248, 250, 254)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 35, 45)
        for i, val in enumerate(r):
            pdf.cell(cw[i], 4.3, val, border=1, fill=True, align='L' if i in [4,5,6] else 'C')
        pdf.ln()
    pdf.ln(1.5)

    # SECTION 8: CONCLUSION, LIMITATIONS & FUTURE SCOPE
    pdf.sec_heading("8", "Conclusion, Limitations & Future Scope")
    pdf.p(
        "Conclusion: DeepShield-X delivers an academically rigorous, reproducible deepfake detection framework. By applying graph Union-Find, "
        "we eliminated identity leakage across 69,273 FF++ C23 crops. We proved mathematically and experimentally that raw accuracy masks catastrophic "
        "failure under class skew, while positive-class weighting enables ResNet50 Regularized (F1 = 0.9633, ROC-AUC = 0.9718) and Xception (F1 = 0.9621) "
        "to lead detection performance. MobileNetV3-Large (35.4 FPS, 17.6 MB) provides a viable edge deployment option. Spatial heatmaps and calibration "
        "metrics confirm predictions are anchored in authentic boundary anomalies."
    )
    pdf.p(
        "Limitations & Future Scope: (1) Cross-Dataset Scope: Current evaluation is restricted to FaceForensics++ C23; future work will test out-of-distribution "
        "transfer to Celeb-DF, DFDC, and diffusion-generated faces. (2) Temporal Integration: Operating on static crops misses inter-frame lip-sync jitter; "
        "future extensions will integrate spatio-temporal LSTM/Transformer heads. (3) Perioral Attention: Dedicated regional attention sub-networks will be "
        "engineered to resolve the NeuralTextures blind spot (55.7% FN). (4) Frequency-Domain Fusion: Combining spatial features with DCT/DTCWT spectral analysis "
        "will mitigate detector collapse below the JPEG Q=50 compression cliff."
    )

    # SECTION 9: IEEE REFERENCES
    pdf.sec_heading("9", "References")
    pdf.set_font('Helvetica', '', 6.6)
    pdf.set_text_color(35, 40, 50)
    refs = [
        "[1] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Niessner, 'FaceForensics++: Learning to detect manipulated facial images,' in Proc. IEEE/CVF ICCV, 2019, pp. 1-11.",
        "[2] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in Proc. IEEE CVPR, 2016, pp. 770-778.",
        "[3] F. Chollet, 'Xception: Deep learning with depthwise separable convolutions,' in Proc. IEEE CVPR, 2017, pp. 1251-1258.",
        "[4] A. Dosovitskiy et al., 'An image is worth 16x16 words: Transformers for image recognition at scale,' in Int. Conf. Learn. Representations (ICLR), 2021.",
        "[5] R. R. Selvaraju et al., 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in Proc. IEEE ICCV, 2017, pp. 618-626.",
        "[6] D. Chicco and G. Jurman, 'The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy,' BMC Genomics, vol. 21, no. 6, 2020.",
        "[7] D. Cozzolino, G. Poggi, J. Thies, M. Niessner, and L. Verdoliva, 'ID-Reveal: Identity-aware deepfake video detection,' in Proc. IEEE/CVF ICCV, 2021, pp. 15108-15117.",
        "[8] H. Zhao, W. Zhou, D. Chen, T. Wei, W. Zhang, and N. Yu, 'Multi-attentional deepfake detection,' in Proc. IEEE/CVF CVPR, 2021, pp. 2185-2194.",
        "[9] A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, 'Lips don't lie: A generalisable approach to face forgery detection,' in Proc. IEEE/CVF CVPR, 2021, pp. 5039-5049.",
        "[10] K. Shiohara and T. Yamasaki, 'Detecting deepfakes with self-blended images,' in Proc. IEEE/CVF CVPR, 2022, pp. 18720-18729.",
        "[11] I. Coccomini, N. Messina, C. Gennaro, and F. Falchi, 'Combining EfficientNet and Vision Transformers for video deepfake detection,' in ICIAP, 2022, pp. 219-229.",
        "[12] S. Dong, J. Wang, J. Liang, H. Fan, and R. Ji, 'Implicit identity leakage: The stumbling block to improving deepfake detection,' in Proc. IEEE/CVF CVPR, 2023, pp. 3994-4004.",
        "[13] Z. Yan, Y. Zhang, Y. Fan, and B. Wu, 'DeepfakeBench: A comprehensive benchmark of deepfake detection,' in NeurIPS Datasets and Benchmarks Track, 2023.",
    ]
    for r in refs:
        pdf.multi_cell(182, 3.2, r)
        pdf.ln(0.4)

    # Output file
    out_dir = str(Path(__file__).resolve().parent)
    out_pdf_path = os.path.join(out_dir, "Deepfake_Detection_Project_Report.pdf")
    pdf.output(out_pdf_path)
    print(f"Generated PDF successfully: {out_pdf_path}")
    print(f"Total Pages: {pdf.page_no()}")

if __name__ == '__main__':
    generate_pdf()
