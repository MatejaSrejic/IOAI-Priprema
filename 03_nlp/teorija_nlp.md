# Teorija NLP — Obrada Prirodnog Jezika

> Ovaj fajl je teorija bez koda. Fokus je na razumevanju ključnih pojmova.

---

## Šta je NLP?

NLP (Natural Language Processing) je oblast veštačke inteligencije koja se bavi razumevanjem i generisanjem **ljudskog teksta i govora**.

Dok ljudi razumeju jezik intuitivno, kompjuteri vide samo niz karaktera. NLP algoritmi pretvaraju tekst u numeričke reprezentacije sa kojima modeli mogu da rade.

Primjeri primjene:
- **Klasifikacija teksta** — spam filter, analiza sentimenta (pozitivno/negativno)
- **Mašinsko prevođenje** — Google Translate
- **Odgovaranje na pitanja** — ChatGPT, Siri, Alexa
- **Sumarizacija** — automatsko rezimiranje dokumenata
- **Ekstrakcija informacija** — izvlačenje naziva kompanija iz teksta

**Na IOAI**, najčešći NLP zadaci su:
- Klasifikacija teksta (sentiment, kategorija, tema)
- Binarna klasifikacija (spam/nije spam, toksično/nije toksično)

---

## Tokenizacija

**Tokenizacija** je razbijanje teksta na manje jedinice — **tokene**.

### Word tokenization (na nivou reči)

```
"Mačka sjedi na jastuku."
→ ["Mačka", "sjedi", "na", "jastuku", "."]
```

### Subword tokenization

Moderni modeli poput BERT-a koriste subword tokenizaciju — razbijaju reči na manji delove:

```
"neobjašnjivo"
→ ["neo", "##bjašn", "##jivo"]
```

Prednost: manje vokabular, model može da "razume" neviđene reči kroz njihove dijelove.

### Character tokenization

Svaki karakter je token. Manje se koristi, ali korisno za kratke tekstove ili primjere sa puno grešaka u kucanju.

---

## Bag of Words (BoW)

**Bag of Words** je jednostavan ali iznenađujuće efikasan način da se tekst pretvori u vektor brojeva.

### Ideja

1. Napravi rječnik svih jedinstvenih reči u datasetu
2. Za svaki tekst, prebroj koliko puta se svaka reč pojavljuje
3. Rezultat je vektor dužine = veličina rječnika

### Primjer

Dokumenti:
- Doc 1: "Mačka sjedi na jastuku"
- Doc 2: "Pas trči po dvoru"
- Doc 3: "Mačka i pas su prijatelji"

Rječnik: `[mačka, sjedi, na, jastuku, pas, trči, po, dvoru, i, su, prijatelji]`

BoW reprezentacije:

| | mačka | sjedi | na | jastuku | pas | trči | po | dvoru | i | su | prijatelji |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Doc 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Doc 2 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| Doc 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |

### Mane BoW

- Ne čuva **redosled reči** — "pas ugrize čovjeka" i "čovjek ugrize psa" imaju isti BoW
- Česte reči ("i", "u", "na") dominiraju, a ne nose smisao → rješenje: **TF-IDF**
- Vektori su dugački i uglavnom nule (sparse)

---

## TF-IDF

**TF-IDF (Term Frequency — Inverse Document Frequency)** je poboljšanje BoW-a koje daje veći značaj rijetkim, informativnim riječima.

### TF (Term Frequency)

Koliko puta se reč pojavljuje u dokumentu (normalizovano dužinom):

```
TF("mačka", Doc1) = 1/4 = 0.25
```

### IDF (Inverse Document Frequency)

Penalizuje reči koje se pojavljuju u **mnogo** dokumenata (jer su manje informativne):

```
IDF("mačka") = log(3/2) ≈ 0.18   (pojavljuje se u 2 od 3 dokumenta)
IDF("sjedi")  = log(3/1) ≈ 0.48  (pojavljuje se samo u 1 dokumentu)
```

Reč "sjedi" je informativnija od "mačka" u ovom kontekstu.

### TF-IDF Score

```
TF-IDF = TF × IDF
```

Reči koje se često pojavljuju u jednom dokumentu, ali rijetko u ostalima, dobivaju visok score.

TF-IDF je **gotovo uvek bolji od BoW** za NLP klasifikaciju, posebno u kombinaciji sa XGBoost-om.

---

## Word Embeddings (Vektorske Reprezentacije)

BoW i TF-IDF tretiraju svaku reč kao nezavisnu. Ali "king" i "queen" su semantički slične, a ove metode to ne znaju.

**Word Embeddings** su naučene vektorske reprezentacije reči gdje su **semantički slične reči blizu** u vektorskom prostoru.

### Word2Vec i GloVe

Klasični embedding modeli koji uče iz velikih tekstualnih korpusa.

```
Vektor("kralj")  ≈ [0.52, -0.3, 0.88, ...]  (dimenzija: 100-300)
Vektor("kraljica") ≈ [0.49, -0.1, 0.91, ...]  # slično "kralj"!
Vektor("jabuka")  ≈ [-0.2, 0.75, -0.4, ...]   # potpuno drugačije
```

Čuvena analogija:
```
vektor("kralj") - vektor("muškarac") + vektor("žena") ≈ vektor("kraljica")
```

### Zašto embeddings bolje od BoW?

- Hvata **semantičku sličnost** između reči
- Dimenzionalniji prostor je kompaktniji (100-300 dimenzija vs. hiljade u BoW)
- Može "razumeti" sinonime

---

## Transformers — Intuicija

**Transformers** su arhitektura koja je revolucionisala NLP od 2017. godine. BERT, GPT, T5 — svi su bazirani na transformerima.

### Problem sa prethodnim pristupima

Ranije metode (LSTM, RNN) obrađivale su tekst **reč po reč** — kao čitanje s lijeva na desno. Problem: teško je pamtiti kontekst s početka dugačke rečenice.

### Ključna ideja: Self-Attention

Transformer gleda **sve reči odjednom** i za svaku reč računa koliko je "važna" svaka druga reč u kontekstu.

Primjer rečenice: *"Banka je bila pored rijeke, i bila je mokra."*

Za razumevanje reči "mokra" — model treba da shvati da se odnosi na "rijeku" (ne na "banku"). Self-attention mehanizam to automatski uči.

```
"mokra" → visoka pažnja na "rijeka"
"mokra" → niska pažnja na "banka"
```

### Zašto je ovo revolucionarno?

- Mogu se procesovati **paralelno** (brže treniranje)
- Bolje hvata **dugoročne zavisnosti** u tekstu
- Pre-treniranje na ogromnim tekstovima → fino podešavanje na malom datasetu

---

## BERT — Bidirectional Encoder Representations from Transformers

BERT je pre-trenirani transformer model koji je naučen na:
- Wikipedia-i (2.5 milijardi reči)
- BookCorpus (800 miliona reči)

Ključna osobina: **bidirectionalan** — razumije kontekst s obje strane reči.

```
"Uzeo sam [MASK] na plaži."
← BERT → kontekst s oba smjera → "sunčanje", "šetnju", "kupaće"
```

Za IOAI: Koristiti BERT (ili manje varijante poput `distilbert-base-uncased`) za zadatke sa tekstom. Transfer learning pristup — isto kao ResNet za slike.

---

## Rezime — Kada Koristiti Šta

| Pristup | Kada koristiti |
|---|---|
| TF-IDF + XGBoost | Kratak tekst, brzo rješenje, dobar baseline |
| Word Embeddings | Kada želiš bolje razumevanje semantike |
| BERT (fine-tuning) | Kada imaš malo podataka ali treba visoka tačnost |

Na IOAI NLP zadacima, **TF-IDF + XGBoost** je odličan baseline, a **BERT fine-tuning** je korak prema medalji.

---

## Sledeći Korak

Pređi na **text_classification_primer.md** za implementaciju klasifikacije teksta sa TF-IDF i XGBoost-om, a zatim na **bert_intro_primer.md** za BERT fine-tuning.
