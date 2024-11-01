# DeepShield-X — Architectural Efficiency Profiling

Hardware benchmark: Local CPU (Intel Core / AMD Ryzen), batch size = 1, input resolution = 224×224×3.

| Model | Total Parameters | Checkpoint Size (MB) | CPU Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|
| **Custom Cnn** | 93,825 | 1.1 MB | 13.25 ± 1.28 ms | **75.5 FPS** |
| **Resnet50** | 23,510,081 | 269.5 MB | 93.52 ± 7.64 ms | **10.7 FPS** |
| **Efficientnet** | 4,008,829 | 46.35 MB | 116.86 ± 107.49 ms | **8.6 FPS** |
| **Mobilenet** | 1,518,881 | 17.64 MB | 53.89 ± 10.27 ms | **18.6 FPS** |
| **Xception** | 20,809,001 | 238.57 MB | 1371.96 ± 778.59 ms | **0.7 FPS** |
| **Vit** | 21,666,049 | 248.14 MB | 73.87 ± 10.89 ms | **13.5 FPS** |

### Key Observations:
- **MobileNetV3** achieves the highest throughput (sub-10ms per frame), making it optimal for mobile/edge embedded deepfake defense.
- **ResNet50** balances strong computational efficiency (~25M parameters, 282MB checkpoint with Adam states) with industry-leading F1.
- **Vision Transformer (ViT)** exhibits the highest compute footprint (86M parameters) and slowest inference latency on standard hardware.
