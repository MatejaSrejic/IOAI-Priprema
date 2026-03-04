# Teorija Machine Learning-a

> **Napomena:** Ovaj fajl je čista teorija bez koda. Fokus je na intuiciji i razumevanju. Kod dolazi u narednim fajlovima.

---

## Šta je Machine Learning?

Machine Learning (ML) je grana veštačke inteligencije gde **kompjuter uči iz podataka** — bez toga da programer eksplicitno napiše pravila.

### Klasičan pristup (bez ML):
Programer piše pravila: `if temperatura > 37 then "groznica"`.

### ML pristup:
Daš kompjuteru hiljade primera (temperaturu + dijagnozu), i on sam nauči pravila. Posle toga može da predvidi dijagnozu za novu temperaturu.

### Ključna ideja:
> ML model je **funkcija** koja prima ulazne podatke i vraća predikciju.

Primer: model koji prima (površinu stana, lokaciju, sprat) i predviđa **cenu stana**.

---

## Supervised vs Unsupervised Learning

### Supervised Learning (Nadzirano učenje)

Imaš podatke **sa oznakama** (labelama) — svaki primer ima tačan odgovor.

**Primer:**
- Email → "spam" ili "nije spam" ✓
- Slika → "mačka" ili "pas" ✓
- Putovanje → trajanje u sekundama ✓

Kod supervised learning-a, model uči vezu između ulaza (features) i izlaza (label/target).

**Ovo je ono što radiš na IOAI — gotovo uvek supervised.**

---

### Unsupervised Learning (Nenadzirano učenje)

Imaš podatke **bez oznaka** — model sam pronalazi strukture.

**Primer:**
- Grupiši kupce po sličnom ponašanju (clustering)
- Smanji broj kolona bez gubljenja informacija (dimensionality reduction)

Na IOAI se ovo ređe pojavljuje, ali vredi znati da postoji.

---

## Overfitting i Underfitting

Ovo je jedan od najvažnijih koncepta u ML-u. Ako ga ne razumeš, ne možeš napraviti dobar model.

### Underfitting — model je previše jednostavan

Zamišljaj studenta koji nije ni naučio gradivo. Na ispitu pada i na pitanjima iz udžbenika i na novim pitanjima.

Model sa underfitting-om:
- Loš rezultat na trening podacima
- Loš rezultat na novim podacima
- **Rešenje:** Složeniji model, više feature-a, više iteracija treninga

### Overfitting — model je zapamtio podatke, a ne naučio

Zamišljaj studenta koji je zapamtio sve odgovore iz udžbenika napamet, ali ne razume gradivo. Na ispitu prolazi, ali samo ako su pitanja identična.

Model sa overfitting-om:
- Odličan rezultat na trening podacima
- Loš rezultat na novim podacima
- **Rešenje:** Manje kompleksan model, regularizacija, više podataka

### Dobar model — generalizuje

Razume suštinu podataka i dobro radi na novim, neviđenim primerima.

```
Underfitting          Dobar model          Overfitting
     |                    |                    |
Previše opšt        Generalizuje        Prespecifičan
Loš na svemu        Dobar svuda         Odličan na train,
                                        loš na test
```

---

## Train / Validation / Test Split

Pre treniranja modela, deli podatke u 3 dela:

### Train set (60-80%)
Podaci na kojima model uči. Ovde se odvija trening.

### Validation set (10-20%)
Podaci koje model **nikad nije video** tokom treninga. Koristiš ih da provjeriš koliko dobro model radi i da podesiš parametre.

### Test set (10-20%)
Finalna provera. Koristiš ga samo jednom, na kraju, da dobiješ konačnu ocjenu modela.

**Zašto ova podjela?**

Ako testiraš model na istim podacima na kojima si ga trenirao, dobijaš lažno visoke rezultate (overfitting). Validation/test set ti govori pravu sliku.

> **Na IOAI:** Organizer ti daje `train.csv` i `test.csv`. `test.csv` nema labele — to su tvoji submission podaci. Ti sami praviš validation split unutar `train.csv`.

---

## Regresija vs Klasifikacija

Ovo su dve osnovne vrste supervised learning zadataka.

### Regresija — predviđaš broj

**Pitanje:** Koliko dugo traje vožnja? Kolika je cena stana?

**Izlaz modela:** Kontinualni broj (npr. 1523.4 sekunde, 245000€)

**Primeri metrika:** RMSE, MAE, MSE, R²

### Klasifikacija — predviđaš kategoriju

**Pitanje:** Da li je email spam? Koje je ovo cvijeće?

**Izlaz modela:** Klasa (npr. "spam", "tulipan", "mačka")

**Primeri metrika:** Accuracy, F1 Score, AUC-ROC

---

## Features (Karakteristike)

Feature je svaka **ulazna promenljiva** koja pomaže modelu da nauči.

**Primer — predviđanje cene stana:**

| Feature | Vrednost |
|---|---|
| Površina (m²) | 75 |
| Broj soba | 3 |
| Sprat | 4 |
| Udaljenost od centra (km) | 2.3 |
| Godina izgradnje | 1995 |

**Target (ono što predviđamo):** Cena = 185,000€

Što bolji features → bolji model. Ovo je najvažniji posao na takmičenju.

---

## Evaluation Metrike

### Za Regresiju

**MAE (Mean Absolute Error)**
- Prosečna apsolutna greška
- Lako interpretirati: "Prosečno grešim za X jedinica"
- MAE = 0 znači savršena predikcija

**MSE (Mean Squared Error)**
- Kvadratna greška — kažnjava veće greške više
- Jedina mana: nije u istim jedinicama kao target

**RMSE (Root Mean Squared Error)**
- Kvadratni koren MSE
- U istim je jedinicama kao target
- Najčešće korišćena metrika za regresiju na takmičenjima

**R² (R-squared)**
- Koliki procenat varijanse podataka model objašnjava
- R² = 1.0 znači savršen model, R² = 0 znači model ne radi ništa bolje od proseka

**RMSLE (Root Mean Squared Log Error)**
- Koristi se kada target vrednosti variraju u ogromnom opsegu (npr. 100 do 1,000,000)
- Kažnjava relativne greške umesto apsolutnih

---

### Za Klasifikaciju

**Accuracy (Tačnost)**
- Procenat tačnih predikcija
- Problem: ako 95% emailova nije spam, model koji uvek kaže "nije spam" ima 95% accuracy, a beskoristan je

**Confusion Matrix**
Vizualizuje koliko je TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative) predikcija.

**Precision**
- Od svih koje je model rekao "da" (pozitivno), koliko ih je zaista pozitivno?
- Bitno kad je FP skupo (npr. lažna dijagnoza raka)

**Recall (Sensitivity)**
- Od svih koji su zaista pozitivni, koliko ih je model pronašao?
- Bitno kad je FN skupo (npr. propustiti stvarnog bolesnika)

**F1 Score**
- Harmonijska sredina Precision i Recall
- Koristiti kada su oba jednako važna
- F1 = 1.0 je savršeno, F1 = 0.0 je najgore

**AUC-ROC (Area Under the Curve)**
- Meri koliko dobro model razlikuje klase
- AUC = 1.0 savršen model, AUC = 0.5 nasumično pogađanje
- Korisna kada imaš neuravnotežene klase

---

## Cross Validation (Unakrsna validacija)

Umesto jednog train/validation splita, praviš **K splita** (najčešće K=5).

**Kako radi K-Fold Cross Validation:**
1. Podeli podatke na 5 jednakih delova (fold-ova)
2. Treniraj na 4 folda, testiraj na 1
3. Ponovi 5 puta, svaki put drugi fold je test
4. Uzmi prosek svih 5 rezultata

**Zašto je ovo bolje?**
- Svaki primer se pojavljuje i u treningu i u validaciji
- Rezultat je pouzdaniji (manje slučajnosti)
- Na takmičenjima: daje bolju procenu stvarnog uspeha

---

## Scaling (Skaliranje)

Skaliranje je pretvaranje feature-a u isti opseg vrednosti (npr. 0 do 1, ili -1 do 1). Neki modeli su osetljivi na to da jedan feature ima vrednosti 0-1 a drugi 0-1,000,000.

**Dobra vijest:** Na IOAI ćeš koristiti **XGBoost**, koji je baziran na stablima odlučivanja i **ne zahteva skaliranje**. Stoga skaliranje možeš uglavnom preskočiti.

---

## ML Modeli — Kratki Pregled

Postoji mnogo ML modela. Ovde ćeš naučiti intuiciju iza svakog, ali nemoj se opterećivati detaljima — videćeš zašto na kraju.

### Linearna Regresija
Pronalazi pravu liniju kroz podatke. Jednostavan, brz, ali loš za složene odnose.

### Logistička Regresija
Slična linearnoj, ali za klasifikaciju. Vraća verovatnoću klase. Dobra kao baseline.

### Decision Tree (Stablo odlučivanja)
Postavi niz pitanja da dođe do odluke. Npr: "Da li je temperatura > 37?" → "Da li ima kašalj?" itd. Lako razumeti, ali sklon overfitting-u.

### Random Forest
Kreira **stotine stabala odlučivanja** i uzima prosek. Robustniji od jednog stabla. Dobar all-around model.

### SVM (Support Vector Machine)
Pronalazi liniju (ili ravninu) koja **maksimizuje razmak** između klasa. Efikasan za male skupove podataka, sporiji za velike.

### K-Nearest Neighbors (KNN)
Za novu tačku, gleda K najbližih suseda i uzima većinu. Intuitivan, ali spor na velikim skupovima.

### Neural Networks
Inspirisan mozgom. Izuzetno moćan za slike i tekst. Zahteva puno podataka i računarskih resursa.

### Gradient Boosting (XGBoost)
Pravi seriju stabala gdje svako sledeće stablo **ispravlja greške prethodnog**. Izuzetno moćan, brz i robustan.

---

## Zašto XGBoost za IOAI ML zadatke?

Na gotovo svakom IOAI ML takmičarskom zadatku pobednik ili osvajač medalje koristi **XGBoost** (ili sličan gradient boosting model poput LightGBM).

Razlozi:
1. **Odlične performanse** na tabelarnim podacima (CSV fajlovi)
2. **Ne zahteva skaliranje** — radi direktno sa sirovim podacima
3. **Automatski rukuje NaN vrednostima**
4. **Relativno lak za podešavanje** hiperparametara
5. **Brz** čak i na velikom skupu podataka

**Zaključak:** Ne moraš duboko razumeti sve ove modele. Za ML deo IOAI-a, XGBoost je tvoj jedini alat. Fokusiraj energiju na razumevanje podataka i feature engineering.

---

## Tipičan Takmičarski ML Pipeline

Ovo je redosled koraka koji svaki takmičar pravi:

```
1. Učitaj podatke (train.csv, test.csv)
        ↓
2. Istraži podatke (EDA — Exploratory Data Analysis)
   - Koliko redova/kolona?
   - Koji su tipovi podataka?
   - Da li ima NaN vrednosti?
   - Kako izgleda distribucija target-a?
        ↓
3. Feature Engineering
   - Kreiraj nove korisne feature-e
   - Transformiši datume, koordinate, tekst...
        ↓
4. Encoding + Čišćenje
   - Pretvori kategorijske promenljive u brojeve
   - Popuni ili ukloni NaN vrednosti
        ↓
5. Train/Validation Split
        ↓
6. Treniranje XGBoost modela
        ↓
7. Evaluacija na validation setu
        ↓
8. Podešavanje hiperparametara (ponavljaj 6-8)
        ↓
9. Re-train na celom train setu
        ↓
10. Predikcija na test.csv
         ↓
11. Submission fajl → upload
```

Ovaj pipeline ćeš implementirati u **prakticni_primer.md**.

---

## Sledeći korak

Pređi na **intuicija_podataka.md** da naučiš kako da razmišljaš o podacima i kako da kreirate dobre feature-e.
