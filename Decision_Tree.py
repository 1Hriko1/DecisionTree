import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 1. CSV einlesen
df = pd.read_csv("gender_classification.csv")

# 2. Merkmale und Zielvariable trennen
X = df.drop("gender", axis=1)
y = df["gender"]

# 3. Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modell trainieren
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 5. Vorhersage treffen
y_pred = clf.predict(X_test)

# 6. Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit: {accuracy:.2%}")


# Entscheidungsbaum visualisieren
plt.figure(figsize=(20,10))
plot_tree(
    clf,
    filled=True,
    feature_names=X.columns.tolist(),
    class_names=clf.classes_.tolist()
)
plt.show()


# Konfusionsmatrix anzeigen
'''
True Positive (TP): Modell sagt „Positiv“ und es ist wirklich Positiv

False Negative (FN): Modell sagt „Negativ“, aber es ist tatsächlich Positiv (Fehler)

False Positive (FP): Modell sagt „Positiv“, aber es ist tatsächlich Negativ (Fehler)

True Negative (TN): Modell sagt „Negativ“ und es ist wirklich Negativ
'''

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Konfusionsmatrix")


# Metriken berechnen
report = classification_report(y_test, y_pred)

# Ausgabe
print(report)
plt.show()
