# SPAM-XAI: Explainable Software Defect Prediction

An explainable software defect prediction pipeline evaluated on the NASA
MDP datasets **CM1**, **PC1** and **PC2**.

## Pipeline

1. Load dataset (last column is the defect label)
2. Standardize features (`StandardScaler`)
3. Balance classes with **SMOTE**
4. 70/30 stratified train/test split
5. **PCA** feature extraction (95% explained variance)
6. **MLP** classifier (hidden layers 100, 50; ReLU; Adam)
7. Evaluation: confusion matrix, classification report, AU-ROC
8. **LIME** explanation of a test instance, saved as an HTML report

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Place `cm1.csv`, `pc1.csv` and `pc2.csv` in the project folder, then run:

```bash
python spam_xai.py
```

Output: metrics printed to the console and one LIME report per dataset
(`lime_cm1_explanation.html`, etc.).

## Datasets

The NASA MDP defect datasets are not included in this repository. Download
them from a public source (e.g. PROMISE / Zenodo / Kaggle) and place the CSV
files next to `spam_xai.py`.

## License

Add a license of your choice (e.g. MIT).
