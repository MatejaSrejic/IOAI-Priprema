# Praktičan Primer — NYC Taxi Trip Duration

U ovom fajlu prolazimo korak po korak kroz kompletan takmičarski ML pipeline na **NYC Taxi Trip Duration** zadatku.

**Cilj:** Na osnovu podataka o taksi vožnji (vreme polaska, koordinate, broj putnika...) predvidi **trajanje vožnje u sekundama**.

**Tip zadatka:** Regresija  
**Metrika:** RMSLE (Root Mean Squared Log Error) — tipično za ovaj zadatak jer trajanja variraju od nekoliko minuta do sat i više.

---

## Korak 1: Uvoz biblioteka

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import math
```

Svaki korak počinje sa ovim importima. Uvek ih stavljaj na vrh notebook-a.

---

## Korak 2: Učitavanje Podataka

```python
train = pd.read_csv("Train.csv")
test  = pd.read_csv("Test.csv")

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
train.head()
```

Provjeri dimenzije — koliko redova i kolona imaš. `train.head()` ti pokazuje prvih 5 redova.

```python
print(train.info())
print(train.describe())
```

`info()` pokazuje tipove podataka i broj non-null vrijednosti po koloni. `describe()` daje brzu statističku sliku (min, max, prosek, kvartili).

---

## Korak 3: Istraživanje Podataka (EDA)

Uvijek pogledaj distribuciju target-a. Veoma neravnomjerna distribucija može zahtijevati log-transformaciju.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(train["trip_duration"], bins=50, color="steelblue")
plt.title("Distribucija trajanja vožnje")
plt.xlabel("Sekunde")

plt.subplot(1, 2, 2)
plt.hist(np.log1p(train["trip_duration"]), bins=50, color="coral")
plt.title("Log distribucija trajanja")
plt.xlabel("log(Sekunde + 1)")

plt.tight_layout()
plt.show()
```

`np.log1p(x)` je `log(x + 1)`. Koristimo `+1` jer `log(0)` je `-∞`. Nakon log transformacije distribucija je mnogo ravnomernija — ovo pomaže modelu.

Provijerimo NaN vrijednosti:

```python
print(train.isnull().sum())
```

---

## Korak 4: Feature Engineering

Ovo je najvažniji korak. Kreiramo feature-e koji daju modelu smislene informacije.

### 4a. Log-transformacija targeta

RMSLE metrika je bazirana na logaritmima. Ako transformišemo target sa `log1p`, možemo koristiti RMSE tokom treninga i dobit iste rezultate.

```python
train["log_trip_duration"] = np.log1p(train["trip_duration"])
```

### 4b. Obrada datuma

```python
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
test["pickup_datetime"]  = pd.to_datetime(test["pickup_datetime"])

for df in [train, test]:
    df["sat"]            = df["pickup_datetime"].dt.hour
    df["dan_sedmice"]    = df["pickup_datetime"].dt.dayofweek
    df["mesec"]          = df["pickup_datetime"].dt.month
    df["dan_u_mesecu"]   = df["pickup_datetime"].dt.day
```

Sada model može naučiti da vožnje u određenim satima (jutarnji/večernji špic) ili danima traju duže.

Sin/cos enkodiranje za sat (da 23h i 0h budu blizu):

```python
for df in [train, test]:
    df["sat_sin"] = np.sin(2 * math.pi * df["sat"] / 24)
    df["sat_cos"] = np.cos(2 * math.pi * df["sat"] / 24)
```

### 4c. Euklidska distanca — Najvažniji Feature

Direktna udaljenost između polazišta i odredišta je najjači prediktor trajanja vožnje. Duža distanca → duža vožnja.

```python
for df in [train, test]:
    df["distanca"] = np.sqrt(
        (df["pickup_latitude"]  - df["dropoff_latitude"])**2 +
        (df["pickup_longitude"] - df["dropoff_longitude"])**2
    )
```

### 4d. Enkodiranje kategorijskih kolona

`vendor_id` je tekst ("CMT" ili "VTS"). XGBoost treba broj.

```python
for df in [train, test]:
    df["vendor_id"] = df["vendor_id"].map({"CMT": 0, "VTS": 1})
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1})
```

`map` zamjenjuje svaku string vrijednost odgovarajućim brojem.

---

## Korak 5: Rukovanje Nedostajućim Vrednostima

```python
print("NaN u train:", train.isnull().sum()[train.isnull().sum() > 0])
```

Popuni NaN vrijednosti medijanom (robustniji od proseka):

```python
train["passenger_count"] = train["passenger_count"].fillna(train["passenger_count"].median())
test["passenger_count"]  = test["passenger_count"].fillna(train["passenger_count"].median())
```

Napomena: Uvek koristiš statistiku iz **train** seta da popuniš i train i test — nikad iz test seta, jer bi to bio data leakage.

---

## Korak 6: Priprema Feature-a i Splita

Definiši koje kolone su feature-i (ulaz modela):

```python
FEATURES = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "store_and_fwd_flag",
    "sat", "dan_sedmice", "mesec", "dan_u_mesecu",
    "sat_sin", "sat_cos",
    "distanca"
]

TARGET = "log_trip_duration"
```

Kolone `id`, `pickup_datetime`, i `trip_duration` (originalni target) se ne uključuju kao feature-i.

Napravi train/validation split:

```python
X = train[FEATURES]
y = train[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Val size:  ", X_val.shape)
```

`random_state=42` osigurava da svaki put kad pokrenemo kod dobijemo **isti** split.

---

## Korak 7: Trening XGBoost Modela

```python
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
```

Objašnjenje ključnih hiperparametara:

| Parametar | Šta radi | Tipičan opseg |
|---|---|---|
| `n_estimators` | Broj stabala | 100 – 2000 |
| `learning_rate` | Koliko svako stablo "uči" | 0.01 – 0.3 |
| `max_depth` | Maksimalna dubina stabla | 3 – 10 |
| `subsample` | Procenat redova po stablu | 0.5 – 1.0 |
| `colsample_bytree` | Procenat feature-a po stablu | 0.5 – 1.0 |
| `random_state` | Seed za reproducibilnost | Bilo koji broj (uvek isti!) |

---

## Korak 8: Evaluacija na Validation Setu

```python
y_pred_val = model.predict(X_val)

rmsle = np.sqrt(mean_squared_log_error(
    np.expm1(y_val),
    np.expm1(y_pred_val)
))

print(f"Validation RMSLE: {rmsle:.4f}")
```

`np.expm1` je inverz `log1p` — vraća predikcije nazad u originalni prostor (sekunde).

---

## Korak 9: Podešavanje Hiperparametara

**Ovo je sada najvažniji posao.** Takmičar ručno podešava hiperparametre dok ne dobije što bolji RMSLE.

**Proces:**

1. Promijeni jedan ili više hiperparametara
2. Ponovi trening (Korak 7)
3. Evaluiraj (Korak 8)
4. Uporedi rezultate

```python
# Pokušaj sa većim brojem stabala i manjim learning rate
model_v2 = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1
)

model_v2.fit(X_train, y_train)
y_pred_v2 = model_v2.predict(X_val)

rmsle_v2 = np.sqrt(mean_squared_log_error(
    np.expm1(y_val),
    np.expm1(y_pred_v2)
))

print(f"V1 RMSLE: {rmsle:.4f}")
print(f"V2 RMSLE: {rmsle_v2:.4f}")
```

Zapamti kombinaciju koja daje najmanji RMSLE.

**Savjeti za podešavanje:**
- Manji `learning_rate` + više `n_estimators` = generalno bolji rezultat, ali sporiji trening
- Manji `max_depth` = manje overfitting-a
- Manji `subsample` i `colsample_bytree` = više regularizacije

---

## Korak 10: Re-Train na Celom Trening Setu

Kad nađeš optimalne hiperparametre, istreniraj model na **svim** trening podacima (bez validation splita) — više podataka za trening znači bolji model.

```python
best_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X, y)
print("Finalni model istreniran na celom datasetu.")
```

---

## Korak 11: Predikcija na Test Setu

```python
X_test = test[FEATURES]

log_predictions = best_model.predict(X_test)

predictions = np.expm1(log_predictions)
```

`np.expm1` pretvara log predikcije nazad u sekunde.

---

## Korak 12: Generisanje Submission Fajla

```python
submission = pd.DataFrame({
    "id":            test["id"],
    "trip_duration": predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission sačuvan!")
print(submission.head())
```

Provjeri fajl:

```python
sub_check = pd.read_csv("submission.csv")
print("Shape:", sub_check.shape)
print(sub_check.describe())
```

Submission fajl mora imati tačno 2 kolone: `id` i `trip_duration`.

---

## Recap — Kompletan Pipeline

```
1. pd.read_csv()                         ← učitaj podatke
2. .info(), .describe(), .isnull().sum() ← istraži podatke
3. Feature engineering                   ← kreiraj nove feature-e
4. .fillna()                             ← rukuj NaN
5. train_test_split(random_state=42)     ← podeli podatke
6. XGBRegressor.fit()                    ← treniraj model
7. RMSLE evaluacija                      ← meri kvalitet
8. Podešavaj hiperparametre              ← ponavljaj 6-7
9. Re-train na svim podacima             ← finalni model
10. .predict() → np.expm1()             ← predikcija
11. pd.DataFrame().to_csv()              ← submission
```

---

## Prilagođavanje Drugom Zadatku

| Šta treba promijeniti | Kako |
|---|---|
| Metrika | Promijeni `mean_squared_log_error` u odgovarajuću |
| Target | Promijeni `TARGET = "nova_kolona"` |
| Feature-i | Dodaj/ukloni kolone u listi `FEATURES` |
| Regresija → Klasifikacija | Koristi `XGBClassifier` umesto `XGBRegressor` |
| Log transformacija | Koristi samo ako target nije normalno distribuiran |

---

## Sledeći Korak

Otvori `nyc_taxi.ipynb` i prođi kroz isti kod interaktivno. Eksperimentiši sa hiperparametrima i feature-ima i pokušaj poboljšati RMSLE score.
