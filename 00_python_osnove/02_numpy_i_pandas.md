# 02 — NumPy i Pandas

Ovo su dve najvažnije biblioteke za rad sa podacima u Pythonu. Bez njih ne možeš raditi ML.

---

## NumPy — Numeričke operacije

### Šta je NumPy?

NumPy (Numerical Python) je biblioteka za efikasan rad sa **matricama i nizovima brojeva**. Gotovo sve ML biblioteke interno koriste NumPy.

Zamišljaj NumPy array kao listu, ali mnogo brži i sa podrškom za matematičke operacije nad celim nizom odjednom.

### Kreiranje array-a

```python
import numpy as np

# 1D niz (vektor)
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.shape)    # (5,) — 5 elemenata

# 2D niz (matrica)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)    # (2, 3) — 2 reda, 3 kolone
```

### Kreiranje specijalnih array-a

```python
np.zeros((3, 4))      # matrica nula 3x4
np.ones((2, 2))       # matrica jedinica 2x2
np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Matematičke operacije

NumPy operacije se primenjuju na svaki element — nema potrebe za for petljom.

```python
a = np.array([1, 2, 3])

print(a + 10)      # [11 12 13]
print(a * 2)       # [2 4 6]
print(a ** 2)      # [1 4 9]
print(np.sqrt(a))  # [1.0, 1.414, 1.732]
```

Operacije između dva array-a (moraju biti iste veličine):

```python
b = np.array([4, 5, 6])
print(a + b)   # [5 7 9]
print(a * b)   # [4 10 18]
```

### Bitne operacije za ML

```python
a = np.array([3, 1, 4, 1, 5, 9])

print(a.mean())    # prosek
print(a.std())     # standardna devijacija
print(a.min())     # minimum
print(a.max())     # maksimum
print(a.sum())     # suma
```

### Indeksiranje i slicing

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(M[0, 1])      # 2 — red 0, kolona 1
print(M[1, :])      # [4 5 6] — ceo red 1
print(M[:, 2])      # [3 6 9] — cela kolona 2
print(M[0:2, 0:2])  # gornji levi blok 2x2
```

### Zašto NumPy a ne obična lista?

Obična Python lista je spora za matematičke operacije. NumPy je optimizovan za to i radi **10-100x brže** na velikim podacima. Sve ML biblioteke (XGBoost, PyTorch, sklearn) interno koriste NumPy.

---

## Pandas — Rad sa tabelarnim podacima

### Šta je Pandas?

Pandas je biblioteka za rad sa **tabelarnim podacima** — zamišljaj ga kao Excel u Pythonu. Osnovna struktura je **DataFrame** (tabela sa imenovanim kolonama).

### Učitavanje CSV fajla

```python
import pandas as pd

df = pd.read_csv("train.csv")
```

Promenljiva `df` je tvoja tabela.

### Brzi pregled podataka

```python
df.head()        # prvih 5 redova
df.tail()        # poslednjih 5 redova
df.shape         # (broj_redova, broj_kolona)
df.columns       # lista naziva kolona
df.dtypes        # tip podataka svake kolone
df.info()        # pregled svih kolona, tipova i broja non-null vrednosti
df.describe()    # statistički pregled (mean, std, min, max, ...)
```

### Pristup podacima

```python
# Jedna kolona — vraca Series (kao lista)
df["trip_duration"]

# Više kolona — vraca DataFrame
df[["pickup_latitude", "pickup_longitude"]]

# Filtriranje redova
df[df["passenger_count"] > 2]

# Filtriranje sa više uslova
df[(df["passenger_count"] > 2) & (df["trip_duration"] < 3600)]
```

### Kreiranje novih kolona (Feature Engineering)

```python
# Aritmetika
df["ukupna_udaljenost"] = df["kolona1"] + df["kolona2"]

# Primeni funkciju na kolonu
def pretvori_u_km(metara):
    return metara / 1000.0

df["distanca_km"] = df["distanca_m"].apply(pretvori_u_km)
```

### Rukovanje nedostajućim vrednostima

```python
df.isnull().sum()              # broj NaN po svakoj koloni

df["kolona"].fillna(0)         # zameni NaN sa 0
df["kolona"].fillna(df["kolona"].mean())  # zameni NaN sa prosekom
df.dropna()                    # ukloni sve redove sa NaN
```

### Osnovna analiza podataka

```python
# Koliko jedinstvenih vrednosti ima u koloni
df["vendor_id"].nunique()
df["vendor_id"].value_counts()   # koliko puta se svaka vrednost pojavljuje

# Korelacija između kolona (vrednosti blizu 1 ili -1 znace jaku vezu)
df.corr(numeric_only=True)

# Grupno računanje
df.groupby("vendor_id")["trip_duration"].mean()
```

### Vizualizacija (kratki primer)

```python
import matplotlib.pyplot as plt

# Histogram
df["trip_duration"].hist(bins=50)
plt.title("Raspodela trajanja vožnje")
plt.xlabel("Sekunde")
plt.show()

# Scatter plot
plt.scatter(df["pickup_longitude"], df["pickup_latitude"], alpha=0.1, s=1)
plt.title("Mapa pickup lokacija")
plt.show()
```

### Čuvanje rezultata

```python
# Sačuvaj DataFrame u CSV
df.to_csv("submission.csv", index=False)
```

`index=False` znači da ne upisuješ broj reda u fajl — uvek koristi ovo za submission fajlove.

---

## Kratki pregled najvažnijih komandi

| Operacija | Komanda |
|---|---|
| Učitaj CSV | `pd.read_csv("fajl.csv")` |
| Prvih 5 redova | `df.head()` |
| Statistika | `df.describe()` |
| Pristup koloni | `df["kolona"]` |
| Nova kolona | `df["nova"] = ...` |
| Filtriranje | `df[df["kol"] > 5]` |
| Broj NaN | `df.isnull().sum()` |
| Sačuvaj | `df.to_csv("fajl.csv", index=False)` |

---

## Sledeći korak

Sada kada znaš Python, NumPy i Pandas, spreman si za pravu Machine Learning sekciju. Pređi na **01_ml/teorija_ml.md**.
