import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


def main(file_path, output_file=None, batch_size=1000):
    data = np.load(file_path)
    print("Successful Load in")
    X, y = data["features"], data["labels"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Starting training...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest baseline validation")
    parser.add_argument('files', default='./data/USTC_concate_data.npz', help='Input NPZ file path')
    parser.add_argument('-o', '--output', default=None, help='Output path to save trained model (optional)')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='Batch size for processing (optional)')
    args = parser.parse_args()

    main(args.files, args.output, args.batch_size)
