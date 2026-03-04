# 00 — Uvod u Python

## Šta je Python?

Python je programski jezik koji se odlikuje čitljivom sintaksom i ogromnim brojem gotovih biblioteka. Upravo zbog toga je postao **lingua franca** u svetu mašinskog učenja, computer vision-a i obrade prirodnog jezika (NLP). Na takmičenjima poput IOAI-a, Python je jedini jezik koji ćeš koristiti.

Nekoliko ključnih osobina:
- **Interpretirani jezik** — pokrećeš kod liniju po liniju, idealno za eksperimentisanje
- **Dinamički tipovi** — ne moraš pisati tip promenljive
- **Bogat ekosistem** — `numpy`, `pandas`, `scikit-learn`, `xgboost`, `pytorch`, `transformers`...

---

## Instalacija Python-a i biblioteka

### Instaliraj Python
Preuzmi Python 3.10 ili noviji sa [python.org](https://www.python.org/downloads/).

Provjeri instalaciju:
```bash
python --version
```

### pip — menadžer paketa
`pip` dolazi uz Python i koristi se za instalaciju biblioteka.

Instaliraj sve biblioteke koje ćeš koristiti na IOAI-u jednom komandom:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter opencv-python torch torchvision transformers
```

Šta svaka biblioteka radi:

| Biblioteka | Namena |
|---|---|
| `numpy` | Rad sa matricama i brojevima |
| `pandas` | Učitavanje i analiza podataka (tabele) |
| `scikit-learn` | Gotovi ML alati (encoding, split, metrike) |
| `xgboost` | Glavni ML model koji ćeš koristiti |
| `matplotlib` / `seaborn` | Vizualizacija |
| `jupyter` | Interaktivni notebook za eksperimentisanje |
| `opencv-python` | Computer Vision |
| `torch` / `torchvision` | Deep Learning (PyTorch) |
| `transformers` | BERT i drugi NLP modeli |

---

## Jupyter Notebook — zašto ga koristiti?

**Jupyter Notebook** ti omogućava da pišeš i pokrećeš Python kod u malim blokovima (ćelijama), vidiš rezultat odmah ispod koda, i kombinuješ tekst i grafike sa kodom.

Ovo je idealno za takmičenje jer:
- Možeš pokrenuti samo jedan deo koda bez ponovnog pokretanja svega
- Odmah vidiš grafike i tabele
- Lako se vraćaš na prethodni korak i popravljaš

### Pokretanje Jupyter Notebook-a

```bash
jupyter notebook
```

Otvoriće se pretraživač. Klikni **New → Python 3** da napraviš novi notebook.

Alternativno, koristi **VS Code** sa instaliranim Jupyter ekstenzijom — možeš otvoriti `.ipynb` fajlove direktno.

### Osnovna upotreba

- **Shift + Enter** — pokreni ćeliju i pređi na sledeću
- **Ctrl + Enter** — pokreni ćeliju bez prelaska
- **B** — dodaj ćeliju ispod (dok si u Command Mode, tj. klikni van polja za unos)
- **M** — promeni ćeliju u Markdown (tekst)

---

## Kako pokrenuti Python skriptu

Ako imaš fajl `moj_kod.py`, pokreni ga ovako:

```bash
python moj_kod.py
```

Na takmičenjima ćeš uglavnom raditi u `.ipynb` (notebook) fajlovima, ali pokretanje skripti je korisno za testiranje.

---

## Preporučena struktura projekta na takmičenju

Kada dobiješ takmičarski zadatak, preporučena struktura foldera:

```
/moj_zadatak/
│
├── train.csv          ← podaci za trening
├── test.csv           ← podaci za predikciju
├── sample_submission.csv  ← primer kako treba da izgleda submission
│
├── resenje.ipynb      ← tvoj glavni notebook
│
└── submission.csv     ← tvoj finalni fajl sa predikcijama
```

Sve radi unutar jednog notebook-a. Na takmičenju nema potrebe za složenom strukturom.

---

## Sledeći korak

Sada kad imaš podešeno okruženje, pređi na **01_osnovna_sintaksa.md** da naučiš Python sintaksu koja ti je potrebna za ML zadatke.
