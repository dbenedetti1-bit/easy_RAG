"""
===========================================================
 SCRIPT 01 - INDICIZZAZIONE DEI DOCUMENTI
===========================================================
 Questo script fa la PRIMA FASE della RAG:
 prende i file PDF dalla cartella "documenti/" e li trasforma
 in un database vettoriale (vector database).

 I passaggi sono:
 1. Leggere il testo dai PDF
 2. Dividere il testo in pezzi (chunking)
 3. Trasformare ogni pezzo in un vettore numerico (embedding)
 4. Salvare tutto in un file JSON (il nostro vector database)

 COME USARLO:
 1. Metti uno o più file PDF nella cartella "documenti/"
 2. Esegui questo script: python 01_indicizza_documenti.py
 3. Il database verrà creato nella cartella "vector_db/"
===========================================================
"""

import os
import sys
import fitz  # PyMuPDF: libreria per leggere i PDF

# Importiamo le funzioni condivise dal nostro file utilita.py
from utilita import (
    leggi_configurazione,
    ottieni_valore_numerico,
    genera_embedding,
    salva_database,
    stampa_separatore,
)


# =============================================================
#  PASSO 1: LETTURA DEI PDF
# =============================================================

def estrai_testo_da_pdf(percorso_pdf):
    """
    Apre un file PDF e ne estrae il testo, pagina per pagina.

    Restituisce una lista di dizionari, uno per pagina:
    [
        {"pagina": 1, "testo": "contenuto della prima pagina..."},
        {"pagina": 2, "testo": "contenuto della seconda pagina..."},
        ...
    ]
    """

    # Apriamo il PDF con PyMuPDF
    documento = fitz.open(percorso_pdf)

    pagine = []

    # Scorriamo tutte le pagine del documento
    for numero_pagina in range(len(documento)):
        # Estraiamo il testo dalla pagina corrente
        pagina = documento[numero_pagina]
        testo = pagina.get_text()

        # Salviamo solo le pagine che contengono testo
        # (alcune pagine potrebbero essere solo immagini)
        if testo.strip():
            pagine.append({
                "pagina": numero_pagina + 1,  # Le pagine partono da 1, non da 0
                "testo": testo.strip()
            })

    documento.close()

    return pagine


# =============================================================
#  PASSO 2: CHUNKING (divisione del testo in pezzi)
# =============================================================

def dividi_in_chunk(testo, dimensione_chunk, sovrapposizione):
    """
    Divide un testo lungo in pezzi più piccoli (chunk), con sovrapposizione.

    Perché non passiamo tutto il testo al modello?
    - I modelli hanno un limite sulla quantità di testo che possono elaborare.
    - Pezzi più piccoli permettono di trovare le informazioni più rilevanti.
    - La sovrapposizione evita di "tagliare" un concetto a metà.

    Esempio con dimensione=10 e sovrapposizione=3:
    Testo:    "ABCDEFGHIJKLMNOPQRST"
    Chunk 1:  "ABCDEFGHIJ"      (caratteri 0-9)
    Chunk 2:  "HIJKLMNOPQ"      (caratteri 7-16, i primi 3 si sovrappongono)
    Chunk 3:  "OPQRST"          (caratteri 14-19, i primi 3 si sovrappongono)

    Parametri:
        testo: il testo da dividere
        dimensione_chunk: quanti caratteri per ogni pezzo
        sovrapposizione: quanti caratteri di sovrapposizione tra pezzi consecutivi

    Restituisce:
        Una lista di stringhe (i chunk)
    """

    chunks = []

    # Il "passo" è di quanti caratteri avanziamo ad ogni iterazione.
    # Se la dimensione è 800 e la sovrapposizione è 150,
    # avanziamo di 650 caratteri ogni volta.
    passo = dimensione_chunk - sovrapposizione

    # Partiamo dall'inizio del testo e avanziamo di "passo" caratteri
    posizione_iniziale = 0

    while posizione_iniziale < len(testo):
        # Prendiamo un pezzo di testo dalla posizione corrente
        posizione_finale = posizione_iniziale + dimensione_chunk
        chunk = testo[posizione_iniziale:posizione_finale]

        # Aggiungiamo il chunk solo se contiene testo significativo
        if chunk.strip():
            chunks.append(chunk.strip())

        # Avanziamo alla prossima posizione
        posizione_iniziale += passo

    return chunks


# =============================================================
#  PASSO 3 e 4: EMBEDDING E SALVATAGGIO
# =============================================================

def indicizza_documenti():
    """
    Funzione principale che orchestra tutto il processo di indicizzazione:
    1. Legge la configurazione
    2. Trova i PDF nella cartella "documenti/"
    3. Per ogni PDF: estrae il testo, lo divide in chunk
    4. Per ogni chunk: calcola l'embedding con Ollama
    5. Salva tutto nel database vettoriale (file JSON)
    """

    stampa_separatore()
    print("  EASY RAG - Indicizzazione documenti")
    stampa_separatore()

    # --- Leggiamo la configurazione ---
    print("\n1. Lettura della configurazione...")
    parametri, prompt = leggi_configurazione()

    modello_embedding = parametri.get("MODELLO_EMBEDDING", "nomic-embed-text")
    dimensione_chunk = ottieni_valore_numerico(parametri, "DIMENSIONE_CHUNK", 800)
    sovrapposizione = ottieni_valore_numerico(parametri, "SOVRAPPOSIZIONE_CHUNK", 150)

    print(f"   Modello embedding: {modello_embedding}")
    print(f"   Dimensione chunk:  {dimensione_chunk} caratteri")
    print(f"   Sovrapposizione:   {sovrapposizione} caratteri")

    # --- Cerchiamo i PDF nella cartella "documenti/" ---
    print("\n2. Ricerca dei file PDF...")

    cartella_documenti = "documenti"
    file_pdf = []

    # Scorriamo tutti i file nella cartella
    for nome_file in os.listdir(cartella_documenti):
        # Controlliamo che il file abbia estensione .pdf (ignorando maiuscole/minuscole)
        if nome_file.lower().endswith(".pdf"):
            percorso_completo = os.path.join(cartella_documenti, nome_file)
            file_pdf.append(percorso_completo)

    # Se non ci sono PDF, avvisiamo l'utente e usciamo
    if not file_pdf:
        print("   ERRORE: Nessun file PDF trovato nella cartella 'documenti/'.")
        print("   Inserisci almeno un PDF e riprova.")
        sys.exit(1)

    print(f"   Trovati {len(file_pdf)} file PDF:")
    for percorso in file_pdf:
        print(f"   - {percorso}")

    # --- Processiamo ogni PDF ---
    print("\n3. Elaborazione dei documenti...")

    # Questa lista conterrà tutti i chunk di tutti i PDF
    tutti_i_chunks = []

    # Contatore progressivo per assegnare un ID unico a ogni chunk
    contatore_id = 0

    for percorso_pdf in file_pdf:
        nome_file = os.path.basename(percorso_pdf)
        print(f"\n   Elaboro: {nome_file}")

        # PASSO 1: Estraiamo il testo dal PDF
        pagine = estrai_testo_da_pdf(percorso_pdf)
        print(f"   - Pagine con testo: {len(pagine)}")

        # PASSO 2: Dividiamo il testo di ogni pagina in chunk
        for info_pagina in pagine:
            chunks_pagina = dividi_in_chunk(
                testo=info_pagina["testo"],
                dimensione_chunk=dimensione_chunk,
                sovrapposizione=sovrapposizione
            )

            # Per ogni chunk, salviamo le informazioni necessarie
            for testo_chunk in chunks_pagina:
                tutti_i_chunks.append({
                    "id": contatore_id,
                    "testo": testo_chunk,
                    "fonte": nome_file,
                    "pagina": info_pagina["pagina"],
                })
                contatore_id += 1

        print(f"   - Chunk creati: {contatore_id}")

    print(f"\n   Totale chunk da tutti i PDF: {len(tutti_i_chunks)}")

    # --- PASSO 3: Calcoliamo gli embedding per ogni chunk ---
    print("\n4. Calcolo degli embedding (questa operazione può richiedere tempo)...")
    print(f"   Modello: {modello_embedding}")

    for indice, chunk in enumerate(tutti_i_chunks):
        # Mostriamo il progresso
        print(f"   Embedding {indice + 1}/{len(tutti_i_chunks)}...", end="\r")

        # Calcoliamo l'embedding del testo del chunk
        vettore = genera_embedding(
            testo=chunk["testo"],
            modello_embedding=modello_embedding
        )

        # Aggiungiamo il vettore al chunk
        chunk["embedding"] = vettore

    print(f"   Embedding {len(tutti_i_chunks)}/{len(tutti_i_chunks)} - Completato!")

    # --- PASSO 4: Salviamo il database vettoriale ---
    print("\n5. Salvataggio del database vettoriale...")

    # Creiamo la struttura del database
    database = {
        "descrizione": "Database vettoriale generato da Easy RAG",
        "modello_embedding": modello_embedding,
        "dimensione_chunk": dimensione_chunk,
        "sovrapposizione_chunk": sovrapposizione,
        "numero_chunks": len(tutti_i_chunks),
        "chunks": tutti_i_chunks
    }

    # Salviamo in formato JSON
    percorso_database = os.path.join("vector_db", "database.json")
    salva_database(database, percorso_database)

    # --- Riepilogo finale ---
    stampa_separatore()
    print("  INDICIZZAZIONE COMPLETATA!")
    stampa_separatore()
    print(f"  PDF elaborati:     {len(file_pdf)}")
    print(f"  Chunk totali:      {len(tutti_i_chunks)}")
    print(f"  Database salvato:  {percorso_database}")
    print()
    print("  Ora puoi fare domande con: python 02_cerca_e_rispondi.py")
    stampa_separatore()


# =============================================================
#  PUNTO DI INGRESSO
# =============================================================
# Questa è la parte che viene eseguita quando lanci lo script.
# La condizione "if __name__" verifica che il file sia stato
# eseguito direttamente (non importato da un altro file).

if __name__ == "__main__":
    indicizza_documenti()
