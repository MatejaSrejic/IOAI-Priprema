# Klasifikacija Teksta — TF-IDF + XGBoost

Ovaj fajl pokazuje kompletan pipeline za klasifikaciju teksta. Zadatak je **analiza sentimenta recenzija filmova** — predvidi da li je recenzija pozitivna ili negativna. Ovo je tipičan IOAI NLP zadatak.

---

## Šta Ćemo Napraviti

```
Ulaz: "This movie was fantastic! Great acting."
Izlaz: 1 (pozitivno)

Ulaz: "Terrible film, waste of time."
Izlaz: 0 (negativno)
```

Pipeline:
1. Učitaj tekst
2. Preprocessing (čišćenje teksta)
3. TF-IDF vektorizacija
4. Trening XGBoost modela
5. Evaluacija
6. Predikcija na test setu

---

## Instalacija

```bash
pip install xgboost scikit-learn
```

---

## Uvoz Biblioteka

```python
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
```

---

## Kreiranje Demo Dataseta

U praksi ćeš učitati CSV. Za demonstraciju, kreiramo mali dataset.

```python
tekstovi = [
    "This movie was absolutely fantastic! Great acting and story.",
    "Terrible film, complete waste of time and money.",
    "I loved every minute of it. Highly recommended!",
    "The worst movie I have ever seen. Boring and slow.",
    "Amazing cinematography and wonderful performances.",
    "Dull, predictable and poorly written. Avoid.",
    "A masterpiece! One of the best films of the year.",
    "Disappointing. Expected much more from this director.",
    "Brilliant script and outstanding cast. Must watch!",
    "Absolute garbage. Not worth your time.",
    "Heartwarming and beautifully made. Loved it!",
    "Painfully boring. I fell asleep halfway through.",
    "Incredible visual effects and gripping plot.",
    "Awful acting and nonsensical story. Terrible.",
    "A fun, entertaining ride from start to finish.",
    "Very bad movie. No plot, no character development.",
    "Wonderful performances by the entire cast.",
    "I regret watching this. Complete disappointment.",
    "One of the most moving films I have ever seen.",
    "Cheap production, bad writing, skip this one."
]

labele = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

df = pd.DataFrame({"tekst": tekstovi, "sentiment": labele})
print(df.head())
print("\nRaspodela klasa:")
print(df["sentiment"].value_counts())
```

---

## Učitavanje Iz CSV-a

Ako imaš pravi dataset:

```python
df = pd.read_csv("train.csv")
print(df.head())
print(df.columns)
```

Prilagodi nazive kolona — tekst može biti u koloni `"text"`, `"review"`, `"comment"` itd.

---

## Preprocessing (Čišćenje Teksta)

Sirovi tekst često sadrži HTML tagove, specijalne karaktere, višak razmaka. Čišćenje poboljšava kvalitet TF-IDF vektorizacije.

```python
def ocisti_tekst(tekst):
    tekst = tekst.lower()
    tekst = re.sub(r"<.*?>", " ", tekst)
    tekst = re.sub(r"[^a-z0-9\s]", " ", tekst)
    tekst = re.sub(r"\s+", " ", tekst)
    tekst = tekst.strip()
    return tekst
```

Ova funkcija:
1. Pretvara u mala slova — "GREAT" i "great" su ista reč
2. Uklanja HTML tagove (npr. `<br>`)
3. Uklanja interpunkciju i specijalne karaktere
4. Uklanja višestruke razmake

```python
df["tekst_ociscen"] = df["tekst"].apply(ocisti_tekst)

print(df[["tekst", "tekst_ociscen"]].head(3))
```

---

## TF-IDF Vektorizacija

Pretvori tekst u numeričke vektore.

```python
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)
```

Objašnjenje parametara:

| Parametar | Šta radi | Preporuka |
|---|---|---|
| `max_features` | Maksimalan broj reči u rječniku | 5000-50000 |
| `ngram_range=(1,2)` | Koristi i pojedinačne reči i bigrame ("not good") | Uvek (1,2) |
| `min_df` | Minimalan broj dokumenata u kojima se reč pojavljuje | 1-5 |
| `sublinear_tf=True` | Logaritamska TF — smanjuje dominaciju čestih reči | Uvek True |

`ngram_range=(1,2)` je posebno važno — "not good" je bigram koji nosi suprotan smisao od samih reči.

---

## Train/Test Split

```python
X = df["tekst_ociscen"]
y = df["sentiment"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Fitovanje TF-IDF

Vektorizator se "fita" samo na train setu, a primjenjuje i na train i na val.

```python
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)

print("Train matrica shape:", X_train_vec.shape)
print("Val matrica shape:  ", X_val_vec.shape)
```

`fit_transform` na train-u uči rječnik i pretvara. `transform` na val-u samo pretvara (bez učenja).

Nikad ne koristi `fit_transform` na val/test setu — to bi bio data leakage.

---

## Trening XGBoost Modela

```python
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model.fit(X_train_vec, y_train)
print("Trening zavrsen.")
```

---

## Evaluacija

```python
y_pred = model.predict(X_val_vec)

acc = accuracy_score(y_val, y_pred)
f1  = f1_score(y_val, y_pred, average="weighted")

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nDetaljni izveštaj:")
print(classification_report(y_val, y_pred, target_names=["Negativno", "Pozitivno"]))
```

`classification_report` pokazuje precision, recall i F1 po svakoj klasi — korisno za dijagnozu gdje model griješi.

---

## Prilagođavanje Hiperparametara

Isti princip kao i za ML regresiju — ručno podešavaš i pratiš metrike.

```python
model_v2 = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model_v2.fit(X_train_vec, y_train)
y_pred_v2 = model_v2.predict(X_val_vec)

print(f"V2 F1: {f1_score(y_val, y_pred_v2, average='weighted'):.4f}")
```

---

## Re-Train na Celom Train Setu

Kad nađeš optimalne parametre:

```python
X_sve = df["tekst_ociscen"]
y_sve = df["sentiment"]

X_sve_vec = vectorizer.fit_transform(X_sve)

best_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

best_model.fit(X_sve_vec, y_sve)
print("Finalni model istreniran.")
```

---

## Predikcija na Test Setu

```python
test = pd.read_csv("test.csv")
test["tekst_ociscen"] = test["tekst"].apply(ocisti_tekst)

X_test_vec = vectorizer.transform(test["tekst_ociscen"])

predikcije = best_model.predict(X_test_vec)
```

---

## Generisanje Submission Fajla

```python
submission = pd.DataFrame({
    "id":        test["id"],
    "sentiment": predikcije
})

submission.to_csv("submission.csv", index=False)
print("Submission sacuvan!")
```

---

## Savjeti za Poboljšanje Skora

1. **Bolje čišćenje teksta** — ukloni stopwords ("the", "a", "is"...) ako nisu važni
2. **Više n-grama** — probaj `ngram_range=(1, 3)`
3. **Veći rječnik** — probaj `max_features=10000` ili više
4. **Feature Engineering** — dodaj features poput broja reči, broja uzvičnika, itd.
5. **Ensemble** — kombiniraj predikcije više modela
6. **BERT** — za visoke bodove, pređi na **bert_intro_primer.md**

---

## Prilagođavanje Drugom Zadatku

| Šta se mijenja | Kako |
|---|---|
| Više klasa (3+) | `XGBClassifier` radi automatski; promijeni `average` u metrici |
| Drugačiji tekst kolona | Promijeni `df["tekst"]` na odgovarajući naziv |
| Drugačija metrika | Promijeni metriku u evaluaciji (accuracy → AUC, itd.) |
| Regresija na tekstu | Koristi `XGBRegressor` + RMSE umesto F1 |
