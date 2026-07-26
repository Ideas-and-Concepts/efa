# src/model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import json

def load_merged_data(filepath="data/processed/merged_data.csv"):
    """Load the labelled dataset."""
    df = pd.read_csv(filepath)
    required = {"odd_1", "odd_X", "odd_2", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def create_target(df):
    """Convert goals into 1/X/2 labels.
       1 = home win, 0 = draw, 2 = away win.
    """
    y = np.where(df["home_goals"] > df["away_goals"], 1,
                 np.where(df["home_goals"] == df["away_goals"], 0, 2))
    return y

def build_features(df):
    """Create the feature matrix from odds.
       This exactly matches the features expected by the dashboard.
    """
    X = pd.DataFrame()
    X['prob_1'] = 1 / df['odd_1']
    X['prob_X'] = 1 / df['odd_X']
    X['prob_2'] = 1 / df['odd_2']
    margin = X['prob_1'] + X['prob_X'] + X['prob_2']
    X['margin'] = margin
    X['norm_prob_1'] = X['prob_1'] / margin
    X['norm_prob_X'] = X['prob_X'] / margin
    X['norm_prob_2'] = X['prob_2'] / margin
    return X

def train_model(X, y):
    """Train a Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Draw (0)", "Home (1)", "Away (2)"]))

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return clf, X.columns.tolist()

def save_model(model, feature_names, model_path="models/rf_model.pkl", meta_path="models/features.json"):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    # Also save feature names for future use
    with open(meta_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"Model saved to {model_path}")
    print(f"Feature names saved to {meta_path}")

def main():
    print("Loading merged data...")
    df = load_merged_data()

    print(f"Dataset shape: {df.shape}")
    print("Creating target and features...")
    y = create_target(df)
    X = build_features(df)

    print(f"Features: {list(X.columns)}")
    print("Training model...")
    model, feature_names = train_model(X, y)

    save_model(model, feature_names)

if __name__ == "__main__":
    main()