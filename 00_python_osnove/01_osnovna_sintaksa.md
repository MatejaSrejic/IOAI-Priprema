# 01 — Osnovna Sintaksa Python-a

Ovaj fajl pokriva sve Python koncepte koje trebaš znati za IOAI. Fokus je na praktičnim primerima koji se direktno primenjuju u ML zadacima.

---

## Promenljive i tipovi podataka

Promenljiva je ime koje daješ nekoj vrednosti. Python automatski prepoznaje tip.

```python
ime = "Nikola"          # string (tekst)
godine = 17             # int (ceo broj)
visina = 1.82           # float (decimalni broj)
takmicar = True         # bool (True/False)
```

Možeš proveriti tip:

```python
print(type(visina))     # <class 'float'>
```

---

## Input i Output

```python
print("Zdravo, svete!")

# Unos od korisnika
ime = input("Upiši ime: ")
print("Zdravo, " + ime)

# Formatiran ispis (f-string) — najčešće korišten
godine = 17
print(f"Imam {godine} godina.")
```

---

## Uslovi (if/elif/else)

```python
bodovi = 85

if bodovi >= 90:
    print("Odličan")
elif bodovi >= 70:
    print("Dobar")
else:
    print("Pao")
```

Uslovi u ML-u se često koriste za feature engineering — npr. da provjeriš da li neka vrijednost nedostaje.

---

## For petlja

```python
# Iteracija kroz listu
vozaci = ["Ana", "Marko", "Petra"]
for vozac in vozaci:
    print(vozac)

# Iteracija kroz opseg brojeva
for i in range(5):
    print(i)   # ispisuje 0, 1, 2, 3, 4

# range(start, stop, korak)
for i in range(0, 10, 2):
    print(i)   # 0, 2, 4, 6, 8
```

---

## While petlja

```python
brojac = 0
while brojac < 5:
    print(brojac)
    brojac = brojac + 1
```

---

## Funkcije

Funkcija je blok koda koji možeš pozvati više puta. Veoma korisno za ML pipeline.

```python
def pozdravi(ime):
    print(f"Zdravo, {ime}!")

pozdravi("Ana")

# Funkcija koja vraća vrednost
def kvadrat(x):
    return x * x

rezultat = kvadrat(5)
print(rezultat)   # 25
```

---

## Liste

Lista je uređena kolekcija vrednosti. Ovo je osnovna struktura u Pythonu.

```python
ocene = [5, 4, 3, 5, 2]

print(ocene[0])       # 5 — prvi element (indeks počinje od 0)
print(ocene[-1])      # 2 — poslednji element

ocene.append(4)       # dodaj na kraj
ocene.remove(3)       # ukloni prvu trojku

print(len(ocene))     # dužina liste
```

Slice — iseci deo liste:

```python
# lista[start:stop]  — stop nije uključen
print(ocene[1:3])     # [4, 3]
```

---

## Rečnici (Dictionary)

Rečnik čuva parove **ključ: vrednost**. Često se koristi za čuvanje parametara modela.

```python
takmicar = {
    "ime": "Nikola",
    "godine": 17,
    "bodovi": 92
}

print(takmicar["ime"])        # Nikola
takmicar["bodovi"] = 95       # izmijeni vrijednost
takmicar["drzava"] = "Srbija" # dodaj novi par

# Iteracija kroz rečnik
for kljuc in takmicar:
    print(kljuc, "->", takmicar[kljuc])
```

---

## Setovi

Set je kolekcija **jedinstvenih** vrednosti. Koristan kad hoćeš da ukloniš duplikate.

```python
kategorije = {"pas", "macka", "pas", "ptica"}
print(kategorije)    # {'macka', 'pas', 'ptica'} — duplikat uklonjen

# Pretvori listu u set i nazad
lista = [1, 2, 2, 3, 3, 3]
jedinstveni = list(set(lista))
print(jedinstveni)   # [1, 2, 3]
```

---

## Rad sa fajlovima — Pandas osnove

U ML-u gotovo uvek radiš sa CSV fajlovima (tabele). Pandas je biblioteka koja ti to olakšava.

```python
import pandas as pd

# Učitaj CSV
df = pd.read_csv("train.csv")

# Pregledaj prvih 5 redova
print(df.head())

# Informacije o kolonama i tipovima
print(df.info())

# Osnovna statistika (min, max, mean...)
print(df.describe())
```

Promenljiva `df` je **DataFrame** — zamisli je kao Excel tabelu u Pythonu.

Pristup kolonama:

```python
# Jedna kolona — rezultat je Series (lista)
trajanje = df["trip_duration"]

# Više kolona — rezultat je DataFrame
koordinate = df[["pickup_latitude", "pickup_longitude"]]

# Filtriranje redova
dugi_putevi = df[df["trip_duration"] > 3600]
```

Kreiranje nove kolone:

```python
df["nova_kolona"] = df["kolona1"] + df["kolona2"]
```

---

## Kratak pregled najvažnijeg

| Koncept | Primer |
|---|---|
| Promenljiva | `x = 5` |
| Lista | `[1, 2, 3]` |
| Rečnik | `{"kljuc": "vrednost"}` |
| For petlja | `for i in range(10):` |
| Funkcija | `def ime(x): return x` |
| Import | `import pandas as pd` |
| CSV učitavanje | `pd.read_csv("fajl.csv")` |

---

## Sledeći korak

Pređi na **02_numpy_i_pandas.md** za detaljniji rad sa matricama i tabelama podataka.
