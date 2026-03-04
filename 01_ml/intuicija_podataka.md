# Intuicija Podataka i Feature Engineering

---

## Kako Model "Razume" Podatke

Jedan od najvažnijih koncepta koji mnogi početnici previde:

> **Model razume samo brojeve.**

Ništa više, ništa manje. Ne razume tekst, ne razume datum, ne razume kategoriju "grad" ili "selo". Sve što daš modelu mora biti broj.

Zamišljaj model kao funkciju: `f(1.2, 0, 3, 45.6) → 1523`. Samo cifre ulaze, cifra izlazi.

---

## Enkodiranje Kategorijskih Promenljivih

Kategorijska promenljiva je kolona čije vrednosti su reči/labele, a ne brojevi. Npr: `"vendor_id"` može biti `"CMT"` ili `"VTS"`.

### OneHotEncoding

Za svaku kategoriju pravi novu binarnu kolonu (0 ili 1). Ovo je standardni pristup za kategorije **bez redosljeda**.

Primer — kolona `"boja"` sa vrednostima: `"crvena"`, `"plava"`, `"zelena"`:

| boja | boja_crvena | boja_plava | boja_zelena |
|---|---|---|---|
| crvena | 1 | 0 | 0 |
| plava | 0 | 1 | 0 |
| zelena | 0 | 0 | 1 |

```python
df = pd.get_dummies(df, columns=["boja"])
```

Kada da koristiš: Kategorije bez redosljeda (boje, gradovi, tipovi...).

**Problem:** Ako kategorija ima 1000 vrednosti, praviš 1000 novih kolona. Tada je bolje koristiti OrdinalEncoder.

### OrdinalEncoder

Zamenjuje svaku kategoriju brojem: `"crvena"` → 0, `"plava"` → 1, `"zelena"` → 2.

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
df["boja_encoded"] = enc.fit_transform(df[["boja"]])
```

Kada da koristiš: Kategorije sa prirodnim redosledom (malo/srednje/veliko, ocjena 1-5...) ili kada ima previše kategorija za OneHot.

### StandardScaler

Skalira feature na prosek 0 i standardnu devijaciju 1. Korisno za modele koji su osetljivi na veličinu feature-a (logistička regresija, SVM, neuralne mreže).

**Napomena:** Kao što smo rekli u teoriji, XGBoost koji ćemo koristiti ne zahteva skaliranje. Ovo ostavljaš po strani za IOAI ML zadatke.

---

## Feature Selection — Izbor Feature-a

Nije svaki feature koristan. Neke kolone treba izbaciti.

### Zašto izbacujemo neke feature-e?

**1. Data Leakage (Curenje podataka)**

Ovo je najopasnija greška u ML-u. Dešava se kada u trening podacima imaš informaciju koja u stvarnom svetu ne bi bila dostupna u trenutku predikcije.

Primer: Predviđaš da li će kupac platiti kredit. Ako uključiš kolonu `"datum_zatvaranja_kredita"`, model će naučiti da kad postoji datum — kredit je plaćen. Ali u trenutku predikcije, taj datum ne postoji! Model se ponaša savršeno na treningu, a potpuno podbacuje u produkciji.

**Uvek izbaci:**
- Kolone koje direktno "otkrivaju" target
- Kolone koje su nastale nakon događaja koji predviđaš
- ID kolone (ne nose informaciju, samo zbunjuju model)
- Kolone sa gotovo svim istim vrednostima (nema informacije)

**2. Redundantne kolone**

Ako imaš `"duzina_km"` i `"duzina_m"` (isti podatak u različitim jedinicama), jedna je suvišna.

```python
# Ukloni kolone
df = df.drop(columns=["id", "naziv_kolone"])
```

### Malo podataka + previše feature-a = loš model

Ovo je važno pravilo:

Zamišljaj da imaš 100 primera i 200 feature-a. Model ima premalo primera da nauči šta je zaista važno — počinje da "pamti" šum u podacima. Ovo je klasičan overfitting.

**Zlatno pravilo:** Bolje 10 dobrih feature-a nego 200 nasumičnih. Uvek razmišljaj o tome da li feature zaista nosi smislenu informaciju.

Ipak, nije pravilo koje se uvek mora striktno poštovati — ponekad model sam prepozna koje feature-e da ignoriše. Budi umeren i testiraj.

---

## Feature Engineering — Kreiranje Boljih Feature-a

Feature engineering je umetnost kreiranja novih, korisnijih kolona iz postojećih podataka. **Ovo je najvažnija veština na takmičenju.**

Dobar feature može podići tvoj score više od bilo kakvog podešavanja modela.

---

### Primer 1: Razdvajanje Datuma

Kolona `"datum"` sa vrednostima tipa `"2016-03-14 08:23:15"` je za model samo string (tekst). Pretvori je u korisne numeričke feature-e.

```python
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

df["sat"] = df["pickup_datetime"].dt.hour
df["dan_u_sedmici"] = df["pickup_datetime"].dt.dayofweek  # 0=ponedeljak
df["mesec"] = df["pickup_datetime"].dt.month
df["dan_u_mesecu"] = df["pickup_datetime"].dt.day
df["godina"] = df["pickup_datetime"].dt.year
```

Zašto ovo pomaže? Model sada može da nauči da su vožnje u petak uveče (sat=22, dan=4) duže zbog gužve. Bez ove transformacije, ne može.

---

### Primer 2: Sin/Cos Enkodiranje za Ciklične Feature-e

Problem: Dan u sedmici 0 (ponedeljak) i 6 (nedjelja) su numerički daleko, ali su u stvarnosti blizu (vikend-ponedeljak). Mesec 12 i mesec 1 su daleko, ali su blizu (novo godišnje vreme).

Rješenje: Sin/cos transformacija čini da su vrednosti na "kružnici" — 12. i 1. su blizu.

```python
import numpy as np

# Za sate u danu (0-23)
df["sat_sin"] = np.sin(2 * np.pi * df["sat"] / 24)
df["sat_cos"] = np.cos(2 * np.pi * df["sat"] / 24)

# Za mesece (1-12)
df["mesec_sin"] = np.sin(2 * np.pi * df["mesec"] / 12)
df["mesec_cos"] = np.cos(2 * np.pi * df["mesec"] / 12)
```

---

### Primer 3: Euklidska Distanca

Kad imaš koordinate polazišta i odredišta (kao u NYC Taxi zadatku), možeš izračunati direktnu udaljenost između dve tačke.

```python
import numpy as np

df["distanca"] = np.sqrt(
    (df["pickup_latitude"]  - df["dropoff_latitude"])**2 +
    (df["pickup_longitude"] - df["dropoff_longitude"])**2
)
```

Ovo je jedan od najbitnijih feature-a za NYC Taxi: vožnja od tačke A do tačke B generalno traje duže ako su one dalje.

---

### Primer 4: Kombinovanje Feature-a

Nekad dva slaba feature-a zajedno daju jak signal.

```python
# Površina sobe = ukupna površina / broj soba
df["povrsina_po_sobi"] = df["ukupna_povrsina"] / df["broj_soba"]

# Prihod po clanu porodice
df["prihod_per_capita"] = df["prihod"] / df["velicina_porodice"]
```

---

### Primer 5: Binning (Grupisanje u kategorije)

Pretvori kontinualni feature u kategoriju.

```python
# Starost -> kategorija
for i in range(len(df)):
    if df.loc[i, "starost"] < 18:
        df.loc[i, "kategorija_starosti"] = "mlad"
    elif df.loc[i, "starost"] < 65:
        df.loc[i, "kategorija_starosti"] = "odrasao"
    else:
        df.loc[i, "kategorija_starosti"] = "stariji"
```

---

### Primer 6: Statistike po Grupi

Za svaki red, dodaj statistiku grupe kojoj pripada.

```python
# Prosečno trajanje vožnje po vendor-u
prosek_po_vendoru = df.groupby("vendor_id")["trip_duration"].mean()
df["prosek_vendor"] = df["vendor_id"].map(prosek_po_vendoru)
```

---

## Da li uvek treba raditi Feature Engineering?

**Ne.** Feature engineering je korisno, ali varira od zadatka do zadatka.

Opšte pravilo:
- Malo podataka + puno feature-a → Loš model (overfit)
- Puno podataka + dobri feature-i → Odličan model
- Puno podataka + previše slabih feature-a → Sporiji trening, malo koristi

**Budi umeren.** Dodaj feature koji ima logično objašnjenje zašto bi pomogao. Ne dodaj nasumične kombinacije u nadi da će nešto "upasti".

---

## Zadatak za Vežbu

### Dataset

Zamišljaj dataset za predviđanje **cene kuće** sa sledećim kolonama:

| Kolona | Opis |
|---|---|
| `datum_prodaje` | Datum kada je kuća prodata (format: YYYY-MM-DD) |
| `povrsina_m2` | Ukupna površina u m² |
| `broj_soba` | Ukupan broj soba |
| `sprat` | Sprat na kome se nalazi |
| `ukupno_spratova` | Ukupan broj spratova u zgradi |
| `godina_izgradnje` | Godina kada je zgrada izgrađena |
| `udaljenost_centar_km` | Udaljenost od centra grada u km |
| `id_nekretnine` | Jedinstveni identifikator |
| `cena_eur` | **TARGET** — cena u evrima |

**Zadatak:** Koji feature-i bi se mogli kreirati ili transformisati da pomognu modelu? Pokušaj da smisliš barem 5 ideja pre nego pogledaš rješenje.

<details>
<summary>📖 Klikni za rješenje</summary>

**Feature-i koje treba odmah izbaciti:**
- `id_nekretnine` — samo redni broj, nema informacije

**Transformacije datuma:**
- `godina_prodaje`, `mesec_prodaje`, `dan_u_sedmici_prodaje` iz `datum_prodaje`
- Prodaje u decembru/januaru mogu biti jeftinije (kraj/početak godine)

**Novi feature-i:**
- `starost_zgrade = godina_prodaje - godina_izgradnje` — starije zgrade su generalno jeftinije
- `povrsina_po_sobi = povrsina_m2 / broj_soba` — kuće sa više malih soba vs. manje velikih
- `relativni_sprat = sprat / ukupno_spratova` — penthouse (gornji sprat) je skuplji

**Sin/Cos za mesec** (ako verujemo da je tržište ciklično):
- `mesec_sin = sin(2π * mesec / 12)`
- `mesec_cos = cos(2π * mesec / 12)`

**Napomena:** Kolona `udaljenost_centar_km` je već dobar feature — ne treba transformisati.

</details>

---

## Sledeći korak

Sada prelazi na **prakticni_primer.md** gdje ćeš implementirati kompletan ML pipeline na stvarnom NYC Taxi zadatku.
