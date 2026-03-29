# Cosa c'è nel Vector Database?

Dopo aver eseguito lo script `01_indicizza_documenti.py`, in questa cartella
troverai un file chiamato `database.json`. Questo file **è** il nostro
database vettoriale.

## Struttura del file

Puoi aprire `database.json` con qualsiasi editor di testo (VS Code, Notepad++,
anche il Blocco Note). Vedrai qualcosa del genere:

```json
{
  "descrizione": "Database vettoriale generato da Easy RAG",
  "modello_embedding": "nomic-embed-text",
  "dimensione_chunk": 800,
  "sovrapposizione_chunk": 150,
  "numero_chunks": 42,
  "chunks": [
    {
      "id": 0,
      "testo": "Il testo originale del primo pezzo...",
      "fonte": "dispensa_capitolo1.pdf",
      "pagina": 3,
      "embedding": [0.0234, -0.1567, 0.0891, ...]
    },
    {
      "id": 1,
      "testo": "Il testo originale del secondo pezzo...",
      "fonte": "dispensa_capitolo1.pdf",
      "pagina": 3,
      "embedding": [-0.0456, 0.2345, -0.0123, ...]
    }
  ]
}
```

## Spiegazione dei campi

### Metadati generali (in cima al file)

| Campo | Cosa contiene |
|---|---|
| `descrizione` | Una descrizione leggibile del database |
| `modello_embedding` | Il nome del modello usato per creare i vettori |
| `dimensione_chunk` | Quanti caratteri ha ciascun pezzo di testo |
| `sovrapposizione_chunk` | Quanti caratteri si sovrappongono tra pezzi consecutivi |
| `numero_chunks` | Il numero totale di pezzi nel database |

### Ogni chunk (dentro la lista "chunks")

| Campo | Cosa contiene |
|---|---|
| `id` | Un numero progressivo che identifica il chunk |
| `testo` | Il testo originale estratto dal PDF. È quello che poi viene mostrato al modello come "contesto" quando fai una domanda |
| `fonte` | Il nome del file PDF da cui proviene il testo |
| `pagina` | Il numero di pagina nel PDF originale |
| `embedding` | La rappresentazione numerica (vettore) del testo. È una lista di numeri decimali (di solito 768 numeri) che codifica il *significato* del testo |

## Come funziona la ricerca?

Quando fai una domanda:

1. La tua domanda viene trasformata in un vettore (embedding) dallo stesso modello
2. Questo vettore viene confrontato con **tutti** i vettori nel database
3. Il confronto usa la **similarità del coseno**: una formula che misura quanto
   due vettori "puntano nella stessa direzione"
4. I chunk con la similarità più alta sono quelli che parlano di argomenti
   più vicini alla tua domanda
5. Questi chunk vengono passati al modello LLM come contesto per generare
   la risposta

## Perché JSON e non un "vero" database?

In un progetto reale si userebbero database vettoriali specializzati
(ChromaDB, FAISS, Pinecone, ecc.) che sono molto più veloci.

Qui usiamo JSON perché:
- **Puoi aprirlo e leggerlo** con un qualsiasi editor di testo
- **Puoi vedere esattamente** cosa c'è dentro, senza strumenti speciali
- **Puoi capire la struttura** del database e come funziona la ricerca
- Con pochi documenti, la velocità non è un problema
