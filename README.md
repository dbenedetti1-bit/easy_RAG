# Easy RAG

Un sistema RAG (Retrieval-Augmented Generation) didattico, progettato per il corso
di AI Engineering della LUMSA.

Questo progetto ti permette di **fare domande ai tuoi documenti PDF** usando un
modello di intelligenza artificiale che gira **interamente sul tuo computer**,
senza inviare dati a servizi esterni.

## Come funziona (in breve)

```
I tuoi PDF  →  Testo suddiviso in pezzi  →  Ogni pezzo diventa un vettore numerico
                                                        ↓
La tua domanda  →  Diventa un vettore  →  Confronto con i vettori dei pezzi
                                                        ↓
                                          I pezzi più simili + la domanda
                                                        ↓
                                             Il modello genera la risposta
```

## Prerequisiti

- **Python 3.10** o superiore → [Scarica Python](https://www.python.org/downloads/)
- **Ollama** → [Scarica Ollama](https://ollama.ai/download)
- **Git** → [Scarica Git](https://git-scm.com/downloads) (per scaricare il progetto)

## Installazione passo per passo

### 1. Scarica il progetto

Apri un terminale (Prompt dei comandi o PowerShell su Windows, Terminale su Mac) e scrivi:

```bash
git clone https://github.com/dbenedetti1-bit/easy_RAG.git
cd Easy_RAG
```

### 2. Crea l'ambiente virtuale

L'ambiente virtuale è una "scatola" isolata dove installare le librerie Python
senza interferire con il resto del sistema.

**Su Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Su Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Quando l'ambiente è attivo, vedrai `(venv)` all'inizio della riga nel terminale.

### 3. Installa le dipendenze Python

```bash
pip install -r requirements.txt
```

### 4. Installa e avvia Ollama

1. Scarica Ollama da [ollama.ai](https://ollama.ai/download) e installalo
2. Apri un **nuovo** terminale e scarica i due modelli necessari:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

Il primo modello (`mistral`) genera le risposte (~4 GB di download).
Il secondo (`nomic-embed-text`) crea gli embedding (~275 MB di download).

Dopo il download, Ollama resta in esecuzione in background. Non chiudere il terminale.

## Utilizzo

### Passo 1: Prepara i documenti

Copia i tuoi file PDF nella cartella `documenti/`.

### Passo 2: Indicizza i documenti

Assicurati che l'ambiente virtuale sia attivo (vedi il `(venv)` nel terminale), poi:

```bash
python 01_indicizza_documenti.py
```

Questo script legge i PDF, li divide in pezzi e crea il database vettoriale.
Ci può volere qualche minuto, a seconda della quantità di testo.

### Passo 3: Fai le tue domande

```bash
python 02_cerca_e_rispondi.py
```

Scrivi la tua domanda e premi Invio. Il sistema cercherà i pezzi di testo
più rilevanti e genererà una risposta. Scrivi `esci` per terminare.

## Struttura del progetto

```
Easy_RAG/
│
├── configurazione.txt           ← Parametri e prompt (modificabile!)
│
├── 01_indicizza_documenti.py    ← Script per creare il database vettoriale
├── 02_cerca_e_rispondi.py       ← Script per fare domande ai documenti
├── utilita.py                   ← Funzioni condivise tra i due script
│
├── documenti/                   ← Metti qui i tuoi PDF
│
├── vector_db/                   ← Qui viene salvato il database vettoriale
│   └── LEGGIMI_VECTOR_DB.md    ← Spiega cosa c'è nel database
│
├── requirements.txt             ← Elenco delle librerie Python necessarie
└── README.md                    ← Questo file
```

## Cosa puoi sperimentare

- **Cambia i prompt** in `configurazione.txt` e osserva come cambiano le risposte
- **Cambia la temperature**: valori bassi (0.1) → risposte più "secche"; valori alti (1.0) → più creative
- **Cambia la dimensione dei chunk**: pezzi più grandi danno più contesto ma meno precisione
- **Prova modelli diversi**: cambia `MODELLO_LLM` in `configurazione.txt` (es. `llama3.2:3b` per un modello più leggero)
- **Apri il database** (`vector_db/database.json`) per vedere come sono fatti i chunk e gli embedding

## Risoluzione problemi

| Problema | Soluzione |
|---|---|
| `ollama: command not found` | Ollama non è installato o non è nel PATH. Reinstalla da [ollama.ai](https://ollama.ai/download) |
| `Error: model not found` | Scarica il modello con `ollama pull nome_modello` |
| `Connection refused` | Ollama non è in esecuzione. Avvialo con `ollama serve` in un altro terminale |
| `No module named 'ollama'` | L'ambiente virtuale non è attivo. Riattivalo (vedi punto 2 dell'installazione) |
| Lo script è molto lento | Prova un modello più piccolo: `llama3.2:3b` al posto di `mistral` |

