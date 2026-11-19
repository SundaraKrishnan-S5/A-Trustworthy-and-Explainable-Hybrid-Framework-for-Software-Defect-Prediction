import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
import lime
import lime.lime_tabular

# Suppress minor convergence and sklearn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_spam_xai(file_path, dataset_name):
    print("=" * 60)
    print(f"Running SPAM-XAI Pipeline on {dataset_name}")
    print("=" * 60)

    # 1. Load Dataset
    df = pd.read_csv(file_path)
    target_col = df.columns[-1]

    X = df.drop(columns=[target_col]).values
    y = df[target_col].apply(
        lambda val: 1 if str(val).strip().lower() in ['true', 'y', '1', 'defective', 'yes'] else 0
    ).values

    # 2. Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. SMOTE Balancing (replicates paper setup)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # 4. Train/Test Split (~30% test split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.30, random_state=42, stratify=y_resampled
    )

    # 5. PCA Feature Extraction (95% EVP variance threshold)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 6. MLP Classifier (max_iter=500 to guarantee clean convergence)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        momentum=0.9,
        epsilon=1e-08,
        random_state=42
    )
    mlp.fit(X_train_pca, y_train)

    # 7. Model Evaluation
    y_pred = mlp.predict(X_test_pca)
    y_prob = mlp.predict_proba(X_test_pca)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AU-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # 8. LIME Interpretability
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_pca,
        feature_names=[f"PC_{i+1}" for i in range(X_train_pca.shape[1])],
        class_names=['Non-Defective', 'Defective'],
        kernel_width=0.25,
        mode='classification'
    )

    exp = explainer.explain_instance(
        data_row=X_test_pca[0],
        predict_fn=mlp.predict_proba,
        num_features=10
    )

    # Save standalone HTML file
    out_name = f"lime_{dataset_name.lower()}_explanation.html"
    exp.save_to_file(out_name)
    print(f"\nSaved LIME report to '{out_name}'")

    # Display directly in notebook output cell (no-op outside notebooks)
    try:
        exp.show_in_notebook(show_table=True)
    except Exception:
        pass


if __name__ == "__main__":
    # Adjust filenames to match your dataset files
    run_spam_xai("cm1.csv", "CM1")
    run_spam_xai("pc1.csv", "PC1")
    run_spam_xai("pc2.csv", "PC2")
