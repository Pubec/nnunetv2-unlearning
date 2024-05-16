# Welcome

Unofficial fork of SOTA lesion segmentation tool **nnUNet model**, developed by Division of Medical Image Computing, German Cancer Research Center (DKFZ). [https://github.com/MIC-DKFZ](https://github.com/MIC-DKFZ)

# Unlearning

This SOTA nnUNet framewok for medical image segmentation is upgraded with **Inter-Scanner-Variability-Suppression** technique of **multi-stage unlearning**.

Unlearning represents technique of backpropagating *confusion loss*, computed from seperate domain predictors on their ability of predicting proper source domain.

Multi-Stage unlearning enables per-downsample-block backpropagation, to enable controlled and uniform confusion loss backpropagation.

# Acknowledgements

Check [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) for nnUNet and framework desciption.

