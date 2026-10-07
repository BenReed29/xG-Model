import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier

DATA_PATH = "shots_2025.csv"
shot_data = pd.read_csv(DATA_PATH)
print(shot_data.columns.to_list())

cleaned_data = shot_data.dropna(subset=["xGoal", "goal", "shotDistance"]).copy()

numerical_features = ["shotAngleAdjusted","shotRush", "shotOnEmptyNet","speedFromLastEvent","shotDistance","defendingTeamMinTimeOnIceSinceFaceoff","shootingTeamAverageTimeOnIce","lastEventShotAngle","shooterTimeOnIce","arenaAdjustedXCordABS","arenaAdjustedYCordAbs","shotRebound"]
categorical_features = ["shotType","lastEventCategory","location","playerPositionThatDidEvent"]



Net_X = 89

dx = Net_X - cleaned_data["xCordAdjusted"]
dy = cleaned_data["yCordAdjusted"].abs()

cleaned_data["calculated_distance"] = np.sqrt(dx**2 + dy**2)
cleaned_data["shot_angle"] = np.degrees(np.arctan2(dy, dx))

missing = [c for c in numerical_features + categorical_features if c not in cleaned_data.columns]
print("Missing columns:", missing)
numerical_features = [c for c in numerical_features if c in cleaned_data.columns]
categorical_features = [c for c in categorical_features if c in cleaned_data.columns]

X = pd.get_dummies(cleaned_data[numerical_features + categorical_features],
columns=categorical_features,)
y = cleaned_data["goal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, label=f"Model (AUC = {auc_score:.4f})")

mp_auc = roc_auc_score(y_test, cleaned_data.loc[X_test.index, "xGoal"])
print(f"MoneyPuck xGoal ROC-AUC: {mp_auc:.4f}")

axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random guessing")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve")
axes[0].legend()

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy="quantile")
axes[1].plot(prob_pred, prob_true, marker="o", label="Model")
axes[1].plot([0, 0.3], [0, 0.3], "--", color="gray", label="Perfect calibration")
axes[1].set_xlabel("Predicted goal probability")
axes[1].set_ylabel("Actual goal rate")
axes[1].set_title("Calibration")
axes[1].legend()

plt.tight_layout()
plt.savefig("xg_model_plots.png", dpi=150)
plt.show()
