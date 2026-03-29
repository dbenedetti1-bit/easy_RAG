"""
===========================================================
 UTILITÀ CONDIVISE - Easy RAG
===========================================================
 Questo file contiene le funzioni usate sia dallo script di
 indicizzazione (01) che da quello di risposta (02).

 Separare le funzioni condivise evita di duplicare il codice
 e rende più facile modificarle in un solo punto.
===========================================================
"""

import os
import json


# =============================================================
#  LETTURA DELLA CONFIGURAZIONE
# =============================================================

def leggi_configurazione(percorso_file="configurazione.txt"):
    """
    Legge il file di configurazione e restituisce due dizionari:
    - parametri: coppie chiave-valore (es. TEMPERATURE = 0.3)
    - prompt: testi lunghi definiti tra [NOME] e la sezione successiva

    Il file di configurazione ha un formato semplice e leggibile,
    pensato per essere modificato anche da chi non programma.
    """

    # Dizionario per i parametri semplici (chiave = valore)
    parametri = {}

    # Dizionario per i prompt (testi lunghi)
    prompt = {}

    # Leggiamo tutto il contenuto del file
    with open(percorso_file, "r", encoding="utf-8") as file:
        contenuto = file.read()

    # --- FASE 1: Estraiamo i parametri semplici ---
    # Scorriamo il file riga per riga
    for riga in contenuto.split("\n"):

        # Ignoriamo le righe vuote e i commenti
        riga_pulita = riga.strip()
        if riga_pulita == "" or riga_pulita.startswith("#") or riga_pulita.startswith("["):
            continue

        # Se la riga contiene un "=", è un parametro
        if "=" in riga_pulita:
            # Dividiamo la riga in nome e valore, usando il primo "=" come separatore
            nome, valore = riga_pulita.split("=", 1)
            nome = nome.strip()
            valore = valore.strip()

            # Salviamo il parametro nel dizionario
            parametri[nome] = valore

    # --- FASE 2: Estraiamo i prompt (blocchi di testo tra [SEZIONI]) ---
    # Cerchiamo tutte le sezioni che iniziano con [NOME_SEZIONE]
    import re
    sezioni = re.split(r"\[([A-Z_]+)\]", contenuto)

    # re.split produce una lista alternata: [testo_prima, nome_sezione, testo_sezione, ...]
    # Partiamo dall'indice 1 e procediamo a coppie
    for i in range(1, len(sezioni), 2):
        nome_sezione = sezioni[i]           # Es: "PROMPT_DI_SISTEMA"
        testo_sezione = sezioni[i + 1]      # Il testo che segue

        # Rimuoviamo le righe vuote all'inizio e alla fine
        prompt[nome_sezione] = testo_sezione.strip()

    return parametri, prompt


def ottieni_valore_numerico(parametri, nome, valore_predefinito):
    """
    Legge un parametro dal dizionario e lo converte in numero.
    Se il parametro non esiste o non è un numero valido,
    restituisce il valore predefinito.

    Esempio: ottieni_valore_numerico(parametri, "TEMPERATURE", 0.3)
    """
    try:
        valore_testo = parametri.get(nome, str(valore_predefinito))
        # Se il valore contiene un punto, è un numero decimale (float)
        if "." in str(valore_testo):
            return float(valore_testo)
        else:
            return int(valore_testo)
    except ValueError:
        print(f"  Attenzione: il parametro '{nome}' non è un numero valido.")
        print(f"  Uso il valore predefinito: {valore_predefinito}")
        return valore_predefinito


# =============================================================
#  GESTIONE DEL VECTOR DATABASE (file JSON)
# =============================================================

def salva_database(database, percorso_file):
    """
    Salva il database vettoriale in un file JSON.

    Il file viene scritto in modo "leggibile" (indent=2), così puoi
    aprirlo con un qualsiasi editor di testo e vedere cosa contiene.

    Struttura del file:
    {
        "chunks": [
            {
                "id": 0,
                "testo": "il testo originale del pezzo...",
                "fonte": "nome_file.pdf",
                "pagina": 3,
                "embedding": [0.123, -0.456, 0.789, ...]
            },
            ...
        ]
    }
    """
    with open(percorso_file, "w", encoding="utf-8") as file:
        json.dump(database, file, ensure_ascii=False, indent=2)

    print(f"  Database salvato in: {percorso_file}")


def carica_database(percorso_file):
    """
    Carica il database vettoriale da un file JSON.
    Restituisce il dizionario con tutti i chunk e i loro embedding.
    """
    with open(percorso_file, "r", encoding="utf-8") as file:
        database = json.load(file)
    return database


# =============================================================
#  COMUNICAZIONE CON OLLAMA
# =============================================================

def genera_embedding(testo, modello_embedding):
    """
    Chiede a Ollama di trasformare un testo in un vettore numerico (embedding).

    Cosa succede dietro le quinte:
    1. Il testo viene inviato al modello di embedding
    2. Il modello lo "legge" e produce una lista di numeri (es. 768 numeri)
    3. Questa lista di numeri RAPPRESENTA il significato del testo
    4. Testi con significato simile avranno vettori simili

    Parametri:
        testo: la stringa da trasformare in vettore
        modello_embedding: il nome del modello Ollama per gli embedding

    Restituisce:
        Una lista di numeri decimali (il vettore embedding)
    """
    import ollama

    # Chiamiamo Ollama per ottenere l'embedding
    risposta = ollama.embed(
        model=modello_embedding,
        input=testo
    )

    # La risposta contiene una lista di embedding (uno per ogni input).
    # Noi ne abbiamo passato uno solo, quindi prendiamo il primo.
    vettore = risposta["embeddings"][0]

    return vettore


def calcola_similarita_coseno(vettore_a, vettore_b):
    """
    Calcola quanto due vettori sono "simili" usando la similarità del coseno.

    Come funziona (intuizione):
    - Immagina due frecce che partono dallo stesso punto.
    - Se puntano nella stessa direzione → similarità = 1 (molto simili)
    - Se sono perpendicolari → similarità = 0 (nessuna relazione)
    - Se puntano in direzioni opposte → similarità = -1 (significati opposti)

    La formula matematica:
                     A · B            (prodotto scalare)
    similarità = -----------
                  |A| × |B|          (prodotto delle lunghezze)

    Parametri:
        vettore_a: prima lista di numeri
        vettore_b: seconda lista di numeri

    Restituisce:
        Un numero tra -1 e 1 (più è alto, più i testi sono simili)
    """
    import numpy as np

    # Convertiamo le liste in array numpy per poter fare calcoli vettoriali
    a = np.array(vettore_a)
    b = np.array(vettore_b)

    # Prodotto scalare: moltiplichiamo elemento per elemento e sommiamo
    prodotto_scalare = np.dot(a, b)

    # Calcoliamo la "lunghezza" (norma) di ciascun vettore
    lunghezza_a = np.linalg.norm(a)
    lunghezza_b = np.linalg.norm(b)

    # Evitiamo la divisione per zero (caso raro ma possibile)
    if lunghezza_a == 0 or lunghezza_b == 0:
        return 0.0

    # Applichiamo la formula della similarità del coseno
    similarita = prodotto_scalare / (lunghezza_a * lunghezza_b)

    return float(similarita)


def stampa_separatore(carattere="=", lunghezza=60):
    """Stampa una riga di separazione per rendere l'output più leggibile."""
    print(carattere * lunghezza)
