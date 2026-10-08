# Title
  Crowd Activity Recognition Using Dynamic Salient Frames and Multi-granularity Information

# Introduction 
  Crowd activity recognition aims to analyze the collective behaviors exhibited by individuals in a crowd, with broad applications in sports analysis, anomaly detection, and real-time video classification. To manage computational cost, existing methods typically segment crowd videos at fixed intervals and apply random frame sampling. However, this approach overlooks differences in how quickly individual behaviors evolve that fast-changing individuals lose critical behavioral detail while slow-changing ones introduce redundant information. Moreover, some individuals act in ways inconsistent with the crowd's overall activity, degrading recognition accuracy. To address these issues, this paper introduces two techniques. The first is a dynamic salient frame selection mechanism that constructs a per-individual weight matrix from inter-frame dissimilarity, scores each frame by its behavioral key-degree, and retains only the most informative frames to capture behavioral progression with minimal redundancy. The second is a multi-granularity representation that inserts groups as an intermediate granularity between individuals and the crowd. Groups contain individual discrepancies locally while weighted aggregation amplifies behaviorally representative members within each group. The representation jointly models individuals, groups, and the crowd, constraining individual-level inconsistencies within group-level representations and weighting individuals by their behavioral criticality to enhance group features. The resulting multi-granularity features are integrated via residual connections. On public benchmarks, the proposed method achieves 96.8% multi-class classification accuracy and 96.9% mean per-class accuracy using only RGB input.

# Main Features
**Multi-granularity Hierarchical Recognition: employs a three-level recognition architecture, individual-group-crowd to achieve feature aggregation from fine to coarse granularity. 
**Keyframe Selection: automatically selects keyframes using an attention mechanism to enhance computational efficiency. 
**Group Modeling: supports group recognition and modeling in dynamic crowd scenarios, capturing both intra-group and inter-group interactions. 
**Dual-path Inference: features a dual-path feature fusion mechanism that combines direct individual inference with group-level inference. 
**End-to-end Training: supports end-to-end model training and inference. 
**Flexible Configuration: allows for flexible model parameter adjustment via YAML configuration files.

# Environment requirements
**Python 3.7+
**PyTorch 1.8+
**CUDA (Recommended for GPU acceleration)

# Training model
**python train.py

# Verifying model 
**python val.py
