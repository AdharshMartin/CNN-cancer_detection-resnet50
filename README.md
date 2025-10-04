# CNN-cancer_detection-resnet50
```
project-root/
│
├── dataset/                          # Histopathological image dataset
│   ├── Acute Lymphoblastic Leukemia (early, pre)/
│   ├── Acute Lymphoblastic Leukemia (pro)/
│   ├── breast_malignant/
│   ├── breast_normal/
│   ├── lung_colon_Adenocarcinoma/
│   ├── lung_colon_normal/
│   └── lung_squamous cell carcinoma/
│
├── dataset_mri/                      # MRI image dataset
│   ├── brain_glioma_tumor/
│   ├── brain_meningioma_tumor/
│   ├── brain_normal/
│   ├── brain_pituitary_tumor/
│   ├── kidney_cyst/
│   ├── kidney_normal/
│   ├── kidney_stone/
│   ├── kidney_tumor/
│   ├── pancreatic_normal/
│   └── pancreatic_tumor/
│
├── dataset-MRI-histo/               # Binary classifier dataset
│   ├── histo/
│   └── MRI/
│
├── test-image/                      # Generic test images
├── test-image-MRI/                  # MRI test samples
├── test-image-mri-histo/           # Binary classifier test samples
│
├── models/                          # Saved models
│   ├── MRI-histopathological-classifier.keras
│   ├── resnet50_cancer_model-MRI-finetuned-version-1.keras
│   └── resnet50_cancer_model-finetuned-version-1.keras
│
├── training-scripts/                # Jupyter notebooks for training and evaluation
│   ├── histopathological-training.ipynb
│   ├── histopathological-fine-tune.ipynb
│   ├── histopathological-predict.ipynb
│   ├── MRI-training.ipynb
│   ├── MRI-finetune.ipynb
│   ├── MRI-predict.ipynb
│   └── Z_MRI-histopathological-classifier-training.ipynb
│
├── README.md                        # Project overview and instructions
└── requirements.txt                 # Dependencies
```
