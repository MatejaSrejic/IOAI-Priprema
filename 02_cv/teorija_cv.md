# Teorija Computer Vision-a

> Ovaj fajl je teorija bez koda. Fokus je na razumevanju, a kod dolazi u narednim fajlovima.

---

## Šta je Computer Vision?

Computer Vision (CV) je oblast veštačke inteligencije koja omogućava kompjuterima da **"vide" i razumiju slike i video zapise**.

Sve što mi radimo pogledom — prepoznajemo lice prijatelja, čitamo tekst, vidimo da je saobraćaj gust — CV algoritmi pokušavaju da automatizuju.

Primjeri primjene:
- **Medicinsko dijagnostikovanje** — detekcija tumora na RTG snimcima
- **Autonomna vozila** — prepoznavanje saobraćajnih znakova i pješaka
- **Prepoznavanje lica** — otključavanje telefona
- **Klasifikacija slika** — "da li je ovo mačka ili pas?"
- **Detekcija objekata** — "gdje su svi pješaci na ovoj fotografiji?"

**Na IOAI**, najčešći CV zadaci su:
- Klasifikacija slika (koja klasa je ovo?)
- Binarna klasifikacija (ima li ovaj rentgen bolest ili ne?)

---

## Slika kao Matrica

Ovo je fundamentalni koncept: **slika je matrica brojeva**.

### Grayscale (crno-bijela) slika

Zamišljaj tablicu gdje svaka ćelija predstavlja jedan piksel. Vrijednost piksela je broj od **0 do 255**:
- 0 = crno
- 255 = bijelo
- Između = nijanse sive

```
Slika 4x4 piksela (grayscale):
┌─────┬─────┬─────┬─────┐
│  0  │  50 │ 128 │ 255 │
│ 200 │ 180 │  90 │  10 │
│  30 │  60 │ 220 │ 170 │
│ 255 │  40 │  80 │ 130 │
└─────┴─────┴─────┴─────┘
```

Za sliku 28×28 piksela (npr. MNIST cifre), matrica je veličine 28×28 = **784 broja**.

### RGB (kolor) slika

Kolor slika ima **3 kanala**: Red (crvena), Green (zelena), Blue (plava).

Svaki piksel je kombinacija 3 broja, svaki od 0 do 255.

```
Slika visine H, širine W:
Shape = (H, W, 3)

Piksel (0,0): [R=255, G=0, B=0]  → čisto crvena
Piksel (0,1): [R=0, G=0, B=255]  → čisto plava
Piksel (1,0): [R=255, G=255, B=0] → žuta (crvena + zelena)
```

Za sliku 224×224 (standardni CV input), matrica ima 224×224×3 = **150,528 brojeva**.

### Zašto je ovo važno?

Model vidi samo ove brojeve. Ne zna da su "oči" ili "auto" — nauči koje kombinacije piksela odgovaraju kojoj klasi.

---

## Konvolucija — Intuicija

Konvolucija je osnovna operacija u Computer Vision-u. Razumi je na ovaj način:

### Filter (Kernel)

Filter je mala matrica (npr. 3×3) koja se "klizi" po slici. Za svaki položaj, množi filter sa odgovarajućim dijelom slike i sabira.

```
Original slika (dio):        Filter 3x3:
┌──┬──┬──┐                  ┌─────┬─────┬─────┐
│ 1│ 2│ 3│                  │ -1  │  0  │  1  │
│ 4│ 5│ 6│  ●  (konvolucija)│ -2  │  0  │  2  │
│ 7│ 8│ 9│                  │ -1  │  0  │  1  │
└──┴──┴──┘                  └─────┴─────┴─────┘

Rezultat = (1*-1 + 2*0 + 3*1) + (4*-2 + 5*0 + 6*2) + (7*-1 + 8*0 + 9*1) = 8
```

### Šta filteri detektuju?

Različiti filteri detektuju različite karakteristike:

- **Ivice (edges)**: Filter koji traži nagle promene brightness-a
- **Horizontalne linije**: Filter koji naglašava horizontalne strukture
- **Vertikalne linije**: Slično, ali vertikalno
- **Teksture**: Kompleksniji filteri

U CNN-u, model **sam uči** koji filteri su korisni za zadatak — ne programiraš ih ručno.

### Feature Map

Nakon primene filtera na cijelu sliku, dobijamo **feature map** — novu "sliku" koja prikazuje gdje je filter detektovao određenu karakteristiku.

```
Originalna slika → [Filter za ivice] → Feature map (mapa ivica)
```

---

## CNN — Konvoluciona Neuronska Mreža

CNN (Convolutional Neural Network) je arhitektura dubokog učenja posebno dizajnirana za slike.

### Osnovna ideja

Zamišljaj CNN kao niz slojeva koji postupno **ekstrahuju sve složenije karakteristike**:

```
Slika ulaz
    ↓
[Conv Layer 1] → Detektuje ivice, boje, jednostavne oblike
    ↓
[Pooling Layer] → Smanjuje veličinu (zadržava važne informacije)
    ↓
[Conv Layer 2] → Detektuje složenije oblike (krugovi, uglovi...)
    ↓
[Pooling Layer] → Smanjuje veličinu
    ↓
[Conv Layer 3] → Detektuje dijelove objekata (oko, kotač...)
    ↓
[Flatten] → Pretvori mapu u dugačak vektor
    ↓
[Fully Connected] → Kombinuje sve naučeno
    ↓
[Softmax izlaz] → Vjerovatnoće svake klase (npr. 90% mačka, 10% pas)
```

### Conv Layer (Konvolucioni sloj)

Primenjuje **mnogo filtera** na ulaz. Npr. 32 filtera → 32 feature mape.

Model uči **težine (weights)** ovih filtera tokom treninga.

### Pooling Layer

Smanjuje prostorne dimenzije. **Max Pooling** uzima maksimalnu vrednost iz svakog 2×2 bloka:

```
Ulaz (4x4):          Max Pooling 2x2:
┌─┬─┬─┬─┐           ┌───┬───┐
│1│3│2│4│           │ 4 │ 6 │
│5│4│1│6│    →      ├───┼───┤
│2│7│3│1│           │ 7 │ 9 │
│0│3│9│2│           └───┴───┘
└─┴─┴─┴─┘
```

Zašto? Smanjuje broj parametara, ali zadržava "da li je ta karakteristika tu negde".

### Fully Connected Layer

Na kraju, sve feature mape se "spljošte" u jedan vektor i prolaze kroz obične neuronske slojeve — kao u klasičnom ML-u.

### Transfer Learning — Ključan za IOAI

Trenirati CNN od nule zahteva milione slika i sate treninga. Na takmičenjima to nije moguće.

**Transfer Learning** znači: uzmi model koji je neko već istrenirao na **milijardama slika** (npr. ResNet, EfficientNet na ImageNet datasetu), i **prilagodi ga svom zadatku**.

Model je već naučio da detektuje ivice, oblike, teksture — samo ga "dotreiraš" na tvojim podacima.

Ovo je standardni pristup na IOAI CV zadacima.

---

## Rezime Ključnih Pojmova

| Pojam | Objašnjenje |
|---|---|
| Piksel | Jedan element slike, broj 0-255 |
| RGB | 3 kanala: Crvena, Zelena, Plava |
| Kernel/Filter | Mala matrica za konvoluciju |
| Feature Map | Rezultat primjene filtera |
| Pooling | Smanjenje prostornih dimenzija |
| CNN | Neuronska mreža za slike |
| Transfer Learning | Upotreba pre-treniranog modela |

---

## Sledeći Korak

Pređi na **cv_osnovna_obrada_slike.md** da naučiš kako da radiš sa slikama u Python-u (OpenCV), a zatim na **cnn_pytorch_primer.md** za implementaciju CNN-a.
