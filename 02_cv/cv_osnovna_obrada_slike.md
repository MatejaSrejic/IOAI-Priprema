# Osnovna Obrada Slike — OpenCV

OpenCV (Open Source Computer Vision Library) je najpopularnija biblioteka za obradu slika. Koristićeš je za učitavanje, promenu i analizu slika pre nego što ih daš modelu.

---

## Instalacija

```bash
pip install opencv-python
```

---

## Učitavanje i Prikaz Slike

```python
import cv2
import matplotlib.pyplot as plt

slika = cv2.imread("slika.jpg")

print("Shape:", slika.shape)
```

Rezultat `shape` je `(visina, sirina, 3)` za kolor sliku. Npr. `(480, 640, 3)`.

**Važna napomena:** OpenCV učitava slike u **BGR** redosledu (Blue, Green, Red), a ne RGB! Matplotlib koristi RGB. Uvek konvertuj pre prikaza:

```python
slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)

plt.imshow(slika_rgb)
plt.axis("off")
plt.show()
```

---

## Resize (Promjena Veličine)

Gotovo uvek moraš resize-ovati slike na istu veličinu pre unosa u model.

```python
nova_slika = cv2.resize(slika, (224, 224))

print("Nova shape:", nova_slika.shape)
```

`(224, 224)` je standardna veličina za mnoge CNN modele (ResNet, EfficientNet...).

Ako dataset ima slike različitih veličina, resize-uj sve na istu dimenziju.

---

## Grayscale Konverzija

Pretvori kolor sliku u crno-bijelu (jedan kanal umesto tri).

```python
siva_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

print("Gray shape:", siva_slika.shape)
```

Shape je sada `(visina, sirina)` — bez trećeg kanala.

Kada koristiti grayscale:
- Zadaci gdje boja nije bitna (npr. prepoznavanje znakova, medicinki snimci)
- Smanjuje veličinu podataka za 3x
- Ubrzava trening

---

## Normalizacija Piksela

Pikseli su u opsegu 0-255. Modeli bolje rade sa vrednostima u opsegu 0-1 (ili -1 do 1).

```python
import numpy as np

slika_float = slika.astype(np.float32)
slika_norm  = slika_float / 255.0

print("Min:", slika_norm.min(), "Max:", slika_norm.max())
```

Ovo je jedan od prvih koraka u ML pipeline-u za slike.

---

## Threshold (Binarizacija)

Pretvori sivkaste piksele u čisto crno (0) ili bijelo (255) na osnovu praga.

```python
siva = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

prag, binarna = cv2.threshold(siva, 127, 255, cv2.THRESH_BINARY)

plt.imshow(binarna, cmap="gray")
plt.title("Binarna slika")
plt.axis("off")
plt.show()
```

Piksel > 127 → bijelo (255), piksel ≤ 127 → crno (0).

Koristiti za: Segmentaciju objekata od pozadine, pripremu medicinskih slika.

### Adaptivni Threshold

Kada je osvetljenje neravnomerno, koristi adaptivni threshold:

```python
adaptivna = cv2.adaptiveThreshold(
    siva, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)
```

---

## Edge Detection (Detekcija Ivica)

Canny algoritam detektuje ivice na slici.

```python
siva = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

ivice = cv2.Canny(siva, threshold1=100, threshold2=200)

plt.imshow(ivice, cmap="gray")
plt.title("Ivice")
plt.show()
```

`threshold1` i `threshold2` kontrolišu osjetljivost. Manji = više ivica, veći = samo jake ivice.

---

## Rad sa Batch-om Slika (za ML pipeline)

Na takmičenjima ćeš procesirati stotine ili hiljade slika. Evo kako to raditi efikasno:

```python
import os

def ucitaj_i_pripremi(putanja_do_foldera, velicina=(224, 224)):
    slike = []
    labele = []
    
    klase = os.listdir(putanja_do_foldera)
    
    for i in range(len(klase)):
        klasa = klase[i]
        folder = os.path.join(putanja_do_foldera, klasa)
        
        if not os.path.isdir(folder):
            continue
        
        for ime_fajla in os.listdir(folder):
            putanja = os.path.join(folder, ime_fajla)
            slika = cv2.imread(putanja)
            
            if slika is None:
                continue
            
            slika = cv2.resize(slika, velicina)
            slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
            slika = slika.astype(np.float32) / 255.0
            
            slike.append(slika)
            labele.append(i)
    
    return np.array(slike), np.array(labele)
```

Ova funkcija pretpostavlja strukturu foldera:

```
dataset/
├── klasa_0/
│   ├── slika1.jpg
│   └── slika2.jpg
└── klasa_1/
    ├── slika3.jpg
    └── slika4.jpg
```

---

## Vizualizacija Više Slika Odjednom

```python
def prikazi_slike(slike, labele, nazivi_klasa, n=8):
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(slike[i])
        plt.title(nazivi_klasa[labele[i]])
        plt.axis("off")
    plt.show()
```

---

## Brzi Pregled Bitnih OpenCV Funkcija

| Funkcija | Šta radi |
|---|---|
| `cv2.imread("slika.jpg")` | Učitaj sliku |
| `cv2.resize(slika, (w, h))` | Promeni veličinu |
| `cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)` | Konvertuj BGR → RGB |
| `cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)` | Konvertuj u grayscale |
| `cv2.Canny(slika, t1, t2)` | Detekcija ivica |
| `cv2.threshold(slika, prag, 255, ...)` | Binarizacija |

---

## Sledeći Korak

Sada kada znaš kako da obradiš slike, pređi na **cnn_pytorch_primer.md** za implementaciju CNN-a sa PyTorch-om.
