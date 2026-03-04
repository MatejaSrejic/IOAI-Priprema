# CNN u PyTorch-u — Praktičan Primer

Ovaj fajl pokazuje kako da implementiraš i istreniraš CNN za klasifikaciju slika. Zadatak je klasifikacija medicinskih slika — **da li X-ray snimak pokazuje upalu pluća (pneumoniju) ili je zdrav** (binarni klasifikacioni zadatak, tipičan za IOAI).

---

## Instalacija

```bash
pip install torch torchvision
```

---

## Uvoz Biblioteka

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
```

---

## Struktura Podataka

Pretpostavljamo strukturu foldera:

```
data/
├── train/
│   ├── NORMAL/
│   │   ├── img001.jpg
│   │   └── ...
│   └── PNEUMONIA/
│       ├── img101.jpg
│       └── ...
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

Ako dataset dolazi kao CSV sa putanjama i labelama, prilagodi `Dataset` klasu ispod.

---

## Dataset Klasa

PyTorch zahteva da definišeš klasu koja zna kako da učita jedan primjer.

```python
class MedicinskaSlika(Dataset):
    def __init__(self, putanje, labele, transform=None):
        self.putanje   = putanje
        self.labele    = labele
        self.transform = transform
    
    def __len__(self):
        return len(self.putanje)
    
    def __getitem__(self, idx):
        slika = cv2.imread(self.putanje[idx])
        slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            slika = self.transform(slika)
        
        labela = self.labele[idx]
        return slika, labela
```

`__len__` vraća ukupan broj primjera. `__getitem__` vraća jedan primjer (sliku i labelu) po indeksu `idx`.

---

## Transformacije (Augmentacija)

Transformacije se primjenjuju na svaku sliku pri učitavanju. Resize i normalizacija su obavezni. Augmentacija (random flip, rotation) se koristi samo na train setu da poveća raznolikost.

```python
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Vrijednosti za `mean` i `std` su standardne za ImageNet — koristiti ih uvek kada radiš Transfer Learning.

---

## Kreiranje DataLoader-a

DataLoader automatski pravi batch-ove i miješa podatke.

```python
# Pripremi liste putanja i labela iz foldera
def ucitaj_dataset(root_folder):
    putanje = []
    labele  = []
    klase   = sorted(os.listdir(root_folder))
    
    for i in range(len(klase)):
        klasa  = klase[i]
        folder = os.path.join(root_folder, klasa)
        
        if not os.path.isdir(folder):
            continue
        
        for ime in os.listdir(folder):
            putanja = os.path.join(folder, ime)
            putanje.append(putanja)
            labele.append(i)
    
    return putanje, labele, klase


train_putanje, train_labele, klase = ucitaj_dataset("data/train")
val_putanje,   val_labele,   _     = ucitaj_dataset("data/val")

print("Klase:", klase)
print("Train primjera:", len(train_putanje))
print("Val primjera:  ", len(val_putanje))
```

```python
train_dataset = MedicinskaSlika(train_putanje, train_labele, transform=train_transform)
val_dataset   = MedicinskaSlika(val_putanje,   val_labele,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)
```

`batch_size=32` znači da model vidi 32 slike odjednom. `shuffle=True` miješa podatke svake epohe.

---

## Model — Transfer Learning sa ResNet18

Koristimo **ResNet18** koji je pre-treniran na ImageNet-u (1.2 miliona slika, 1000 klasa). Samo zadnji sloj prilagođavamo za naš zadatak.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Koristi:", device)

model = models.resnet18(weights="IMAGENET1K_V1")

# Zameni zadnji sloj za binarnu klasifikaciju (2 klase)
broj_klasa = 2
model.fc = nn.Linear(model.fc.in_features, broj_klasa)

model = model.to(device)
```

Zašto ResNet18?
- Dobra ravnoteža između tačnosti i brzine
- Pre-treniran na raznolikim slikama
- Lak za prilagođavanje

Ako imaš više GPU resursa ili puno podataka, probaj ResNet50 ili EfficientNet-B0.

---

## Loss Funkcija i Optimizer

```python
kriterij  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

`CrossEntropyLoss` je standardna loss funkcija za klasifikaciju.

`Adam` optimizer automatski prilagođava learning rate — dobar početni izbor.

---

## Trening Petlja

```python
def treniraj_jednu_epohu(model, loader, optimizer, kriterij, device):
    model.train()
    ukupan_gubitak = 0.0
    
    for slike, labele in loader:
        slike  = slike.to(device)
        labele = labele.to(device)
        
        optimizer.zero_grad()
        izlaz   = model(slike)
        gubitak = kriterij(izlaz, labele)
        gubitak.backward()
        optimizer.step()
        
        ukupan_gubitak = ukupan_gubitak + gubitak.item()
    
    return ukupan_gubitak / len(loader)
```

Objašnjenje svakog koraka:
- `optimizer.zero_grad()` — resetuje gradijente (obavezno!)
- `model(slike)` — forward pass (predikcija)
- `kriterij(izlaz, labele)` — računa grešku
- `gubitak.backward()` — računa gradijente (backpropagation)
- `optimizer.step()` — ažurira težine modela

---

## Evaluacija

```python
def evaluiraj(model, loader, device):
    model.eval()
    sve_predikcije = []
    sve_labele     = []
    
    with torch.no_grad():
        for slike, labele in loader:
            slike  = slike.to(device)
            izlaz  = model(slike)
            
            _, predikcije = torch.max(izlaz, dim=1)
            
            sve_predikcije.extend(predikcije.cpu().numpy())
            sve_labele.extend(labele.numpy())
    
    acc = accuracy_score(sve_labele, sve_predikcije)
    f1  = f1_score(sve_labele, sve_predikcije, average="weighted")
    return acc, f1
```

`torch.no_grad()` isključuje računanje gradijenta — nije potrebno pri evaluaciji i ubrzava kod.

---

## Glavna Trening Petlja

```python
EPOHE = 10

for epoha in range(EPOHE):
    train_gubitak = treniraj_jednu_epohu(model, train_loader, optimizer, kriterij, device)
    val_acc, val_f1 = evaluiraj(model, val_loader, device)
    
    print(f"Epoha {epoha+1}/{EPOHE} | Loss: {train_gubitak:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
```

Prati kako se accuracy i F1 mijenjaju. Ako validation metrika prestane da se poboljšava (ili se pogorša) — trening možeš zaustaviti.

---

## Čuvanje i Učitavanje Modela

```python
# Sačuvaj model
torch.save(model.state_dict(), "best_model.pth")
print("Model sačuvan.")

# Učitaj model (za predikciju)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
```

---

## Predikcija na Test Setu

```python
def predvidi(model, test_putanje, transform, device):
    model.eval()
    predikcije = []
    
    for putanja in test_putanje:
        slika = cv2.imread(putanja)
        slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
        
        tensor = transform(slika).unsqueeze(0).to(device)
        
        with torch.no_grad():
            izlaz     = model(tensor)
            _, pred   = torch.max(izlaz, dim=1)
            predikcije.append(pred.item())
    
    return predikcije
```

`unsqueeze(0)` dodaje batch dimenziju jer model očekuje batch, a mi dajemo jednu sliku.

---

## Generisanje Submission Fajla

```python
test_putanje_lista, test_id_lista = [], []

test_folder = "data/test"
for ime in os.listdir(test_folder):
    test_putanje_lista.append(os.path.join(test_folder, ime))
    test_id_lista.append(ime.replace(".jpg", ""))

predikcije = predvidi(model, test_putanje_lista, val_transform, device)

submission = pd.DataFrame({
    "id":    test_id_lista,
    "label": predikcije
})

submission.to_csv("submission.csv", index=False)
print("Submission sacuvan!")
```

---

## Prilagođavanje Drugom Zadatku

| Šta treba promjeniti | Kako |
|---|---|
| Broj klasa | Promijeni `broj_klasa = N` |
| Arhitektura | `models.resnet50()`, `models.efficientnet_b0()` |
| Veličina slike | Promijeni `(224, 224)` u transform-u |
| Metrika | Promijeni u `evaluiraj()` funkciji |
| Multiclass | `CrossEntropyLoss` već radi za N klasa |
| Regression | Promijeni zadnji sloj na `nn.Linear(in, 1)` i koristi `MSELoss` |

---

## Rezime Tipičnog CV Pipeline-a na IOAI

```
1. Učitaj slike + labele
2. Resize na (224, 224) + normalizuj
3. Podeli na train/val
4. Napravi DataLoader
5. Učitaj pre-treniran model (ResNet18)
6. Zameni zadnji sloj
7. Treniraj N epoha
8. Evaluiraj na val setu
9. Podesi learning_rate, batch_size, broj epoha
10. Predikcija na test setu → submission.csv
```
