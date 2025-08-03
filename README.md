This is the repository for the paper "Leveraging self-supervised pretraining using transformers for enhanced lung nodule detection in CT scans" accepted to MICCAI MLMI 2025 workshop.

Authors: Jiaying Liu, Qi Ma , Anna Corti, Valentina D. A. Corino, Luca Mainardi, and Ender Konukoglu

**Abstract.** Lung nodule detection is critical for early diagnosis of lung cancer, but remains challenging due to the nodules’ resemblance to normal tissues. Recent transformer-based approaches have made significant progress; however, their large number of parameters necessitates extensive annotated datasets to achieve robust and reliable results. To address this, we leverage state-of-the-art self-supervised training methods, specifically Masked Image Modeling, on a large domain-specific dataset of lung screening CTs, followed by finetuning on the annotated LUNA16 dataset. Our method achieves an AP of 82.63% and an mAP of 81.23%, outperforming the baseline nnDetection. The experiments demonstrate the effectiveness of pretraining, yielding an increase of 24.0% in performance on the Video-ViT backbone and 4.1% on the Swin Transformer. Additionally, we examine the effect of RGB video pretraining and architectural variations during both pretraining and fine-tuning stages. This work highlights the potential of self-supervised learning in improving efficiency and accuracy in lung cancer screening.

For more information related to the pretraining and nnDetection framework, please see:
https://github.com/facebookresearch/mae_st
https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR
https://github.com/MIC-DKFZ/nnDetection 
