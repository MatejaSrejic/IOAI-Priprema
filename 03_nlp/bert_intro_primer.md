# BERT — Fine-Tuning sa HuggingFace

Ovaj fajl pokazuje kako da koristiš **BERT** za NLP klasifikacione zadatke kroz HuggingFace biblioteku. Zadatak je isti kao u prethodnom fajlu — klasifikacija sentimenta — ali sada koristimo pre-trenirani transformer model za bolji rezultat.

---

## Zašto BERT i kada ga koristiti?

TF-IDF + XGBoost je brz i solidan pristup. BERT je spori, ali skoro uvek tačniji.

| Pristup | Prednosti | Mane |
|---|---|---|
| TF-IDF + XGBoost | Brz, lak za implementaciju | Ne razumije kontekst |
| BERT | Visoka tačnost, razumije kontekst | Sporiji trening, treba GPU |

**Preporuka:** Počni sa TF-IDF baseline-om. Ako treba bolje, pređi na BERT.

---

## Instalacija

```bash
pip install transformers torch
```

---

## Uvoz Biblioteka

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
```

---

## Odabir Modela

Na IOAI ćeš najčešće koristiti:

| Model | Veličina | Preporuka |
|---|---|---|
| `distilbert-base-uncased` | Mala | Brz, dobar za početak |
| `bert-base-uncased` | Srednja | Standardni izbor |
| `roberta-base` | Srednja | Generalno bolji od BERT |

Počni sa `distilbert` — brži je a skoro iste tačnosti kao `bert-base`.

```python
MODEL_NAME = "distilbert-base-uncased"
```

---

## Tokenizer

Tokenizer pretvara tekst u format koji BERT razumije (ID-ove tokena).

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Test tokenizacije
primjer = "This movie was fantastic!"
tokeni = tokenizer(primjer, return_tensors="pt")
print("Input IDs:", tokeni["input_ids"])
print("Dekodovano:", tokenizer.decode(tokeni["input_ids"][0]))
```

---

## Dataset Klasa

```python
class TekstDataset(Dataset):
    def __init__(self, tekstovi, labele, tokenizer, max_duzina=128):
        self.tekstovi   = tekstovi
        self.labele     = labele
        self.tokenizer  = tokenizer
        self.max_duzina = max_duzina
    
    def __len__(self):
        return len(self.tekstovi)
    
    def __getitem__(self, idx):
        enkodovano = self.tokenizer(
            self.tekstovi[idx],
            max_length=self.max_duzina,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids      = enkodovano["input_ids"].squeeze()
        attention_mask = enkodovano["attention_mask"].squeeze()
        labela         = torch.tensor(self.labele[idx], dtype=torch.long)
        
        return input_ids, attention_mask, labela
```

Objašnjenje:
- `max_length=128` — maksimalna dužina sekvence (duže se skraćuje, kraće se popunjava)
- `padding="max_length"` — sve sekvence su iste dužine (batch zahteva to)
- `truncation=True` — skrati tekst ako je duži od `max_length`
- `attention_mask` — bitmapa koja govori modelu koji tokeni su pravi, a koji su padding

---

## Priprema Podataka

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
    "Very bad movie. No plot, no character development."
]
labele = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

train_t, val_t, train_l, val_l = train_test_split(
    tekstovi, labele, test_size=0.2, random_state=42
)

train_dataset = TekstDataset(train_t, train_l, tokenizer, max_duzina=64)
val_dataset   = TekstDataset(val_t,   val_l,   tokenizer, max_duzina=64)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
```

---

## Učitavanje Pre-Treniranog BERT Modela

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Koristi:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model = model.to(device)
```

`AutoModelForSequenceClassification` automatski dodaje klasifikacioni sloj na vrh BERT-a. `num_labels=2` za binarnu klasifikaciju.

---

## Optimizer i Scheduler

```python
EPOHE      = 3
LR         = 2e-5
ukupno_koraka = len(train_loader) * EPOHE

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=ukupno_koraka
)
```

`AdamW` je preporučeni optimizer za BERT fine-tuning.

`get_linear_schedule_with_warmup` postepeno smanjuje learning rate tokom treninga — standardna praksa za transformere.

Tipični learning rate za BERT fine-tuning: `2e-5` do `5e-5`. Ne koristi veće vrijednosti — model će "zaboraviti" šta je naučio.

---

## Trening Petlja

```python
kriterij = nn.CrossEntropyLoss()

def treniraj_epohu(model, loader, optimizer, scheduler, kriterij, device):
    model.train()
    ukupan_gubitak = 0.0
    
    for input_ids, attention_mask, labele in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labele         = labele.to(device)
        
        optimizer.zero_grad()
        
        izlaz = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = izlaz.logits
        
        gubitak = kriterij(logits, labele)
        gubitak.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        ukupan_gubitak = ukupan_gubitak + gubitak.item()
    
    return ukupan_gubitak / len(loader)
```

`clip_grad_norm_(model.parameters(), 1.0)` je gradient clipping — sprečava eksploziju gradijenta, standardno za transformere.

---

## Evaluacija

```python
def evaluiraj(model, loader, device):
    model.eval()
    sve_pred   = []
    sve_labele = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labele in loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            izlaz = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = izlaz.logits
            
            predikcije = torch.argmax(logits, dim=1)
            
            sve_pred.extend(predikcije.cpu().numpy())
            sve_labele.extend(labele.numpy())
    
    acc = accuracy_score(sve_labele, sve_pred)
    f1  = f1_score(sve_labele, sve_pred, average="weighted")
    return acc, f1
```

---

## Pokretanje Treninga

```python
for epoha in range(EPOHE):
    gubitak = treniraj_epohu(model, train_loader, optimizer, scheduler, kriterij, device)
    acc, f1 = evaluiraj(model, val_loader, device)
    
    print(f"Epoha {epoha+1}/{EPOHE} | Loss: {gubitak:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
```

Tipično je dovoljno 2-5 epoha za fine-tuning. Previše epoha → overfitting.

---

## Čuvanje i Učitavanje

```python
# Sačuvaj
model.save_pretrained("moj_bert_model")
tokenizer.save_pretrained("moj_bert_model")
print("Model sačuvan.")

# Učitaj za predikciju
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("moj_bert_model")
model     = AutoModelForSequenceClassification.from_pretrained("moj_bert_model")
model     = model.to(device)
```

---

## Predikcija na Novom Tekstu

```python
def predvidi_tekst(tekst, model, tokenizer, device):
    model.eval()
    
    enkodovano = tokenizer(
        tekst,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids      = enkodovano["input_ids"].to(device)
    attention_mask = enkodovano["attention_mask"].to(device)
    
    with torch.no_grad():
        izlaz = model(input_ids=input_ids, attention_mask=attention_mask)
        predikcija = torch.argmax(izlaz.logits, dim=1).item()
    
    return predikcija


# Test
print(predvidi_tekst("This was absolutely amazing!", model, tokenizer, device))
print(predvidi_tekst("Terrible and boring movie.", model, tokenizer, device))
```

---

## Generisanje Submission Fajla

```python
test = pd.read_csv("test.csv")
test_dataset = TekstDataset(
    list(test["tekst"]), [0] * len(test), tokenizer, max_duzina=128
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
sve_pred = []

for input_ids, attention_mask, _ in test_loader:
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        izlaz      = model(input_ids=input_ids, attention_mask=attention_mask)
        predikcije = torch.argmax(izlaz.logits, dim=1)
        sve_pred.extend(predikcije.cpu().numpy())

submission = pd.DataFrame({
    "id":        test["id"],
    "sentiment": sve_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission sacuvan!")
```

---

## Prilagođavanje Drugom Zadatku

| Šta se mijenja | Kako |
|---|---|
| Broj klasa | Promijeni `num_labels=N` |
| Duži tekst | Povećaj `max_duzina` (max 512 za BERT) |
| Bolji model | Zamijeni `MODEL_NAME` (roberta-base, xlm-roberta...) |
| Višejezični tekst | Koristi `xlm-roberta-base` umjesto distilbert |
| Više epoha | Povećaj `EPOHE` (pazi na overfitting) |

---

## Brzi Savjeti za IOAI NLP

1. **Počni sa TF-IDF + XGBoost** — brz baseline, dobar ako imaš malo vremena
2. **Pređi na DistilBERT** za bolji score
3. **Koristiti GPU** ako je dostupan — BERT bez GPU-a je spor
4. **max_duzina = 128** je dovoljna za većinu kraćih tekstova
5. **Learning rate 2e-5** je siguran izbor
6. **3-4 epohe** obično dovoljno za fine-tuning
