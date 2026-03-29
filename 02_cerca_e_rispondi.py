"""
===========================================================
 SCRIPT 02 - CERCA E RISPONDI
===========================================================
 Questo script fa la SECONDA FASE della RAG:
 prende la domanda dell'utente, cerca i pezzi di testo
 più rilevanti nel database vettoriale, e li usa come
 contesto per generare una risposta con il modello LLM.

 I passaggi per OGNI domanda sono:
 1. Trasformare la domanda in un vettore (embedding)
 2. Confrontare il vettore con quelli nel database (ricerca)
 3. Prendere i pezzi di testo più simili (contesto)
 4. Inviare contesto + domanda al modello LLM (generazione)
 5. Mostrare la risposta all'utente

 COME USARLO:
 1. Assicurati di aver prima eseguito 01_indicizza_documenti.py
 2. Esegui: python 02_cerca_e_rispondi.py
 3. Scrivi le tue domande e premi Invio
 4. Scrivi "esci" per terminare
===========================================================
"""

import os
import sys
import ollama

# Importiamo le funzioni condivise dal nostro file utilita.py
from utilita import (
    leggi_configurazione,
    ottieni_valore_numerico,
    genera_embedding,
    carica_database,
    calcola_similarita_coseno,
    stampa_separatore,
)


# =============================================================
#  PASSO 1 e 2: RICERCA NEL DATABASE VETTORIALE
# =============================================================

def cerca_chunks_simili(domanda_embedding, database, numero_risultati):
    """
    Cerca nel database i chunk il cui embedding è più simile
    a quello della domanda.

    Come funziona:
    1. Prendiamo il vettore della domanda
    2. Lo confrontiamo con il vettore di OGNI chunk nel database
    3. Calcoliamo la similarità del coseno per ogni confronto
    4. Restituiamo i chunk con la similarità più alta

    È come cercare in una biblioteca: la domanda è il "tema"
    che cerchiamo, e la similarità ci dice quali pagine
    parlano dello stesso tema.

    Parametri:
        domanda_embedding: il vettore della domanda dell'utente
        database: il database vettoriale caricato dal JSON
        numero_risultati: quanti chunk restituire

    Restituisce:
        Una lista dei chunk più simili, ordinati dal più al meno simile
    """

    # Lista dove salveremo ogni chunk con il suo punteggio di similarità
    risultati = []

    # Scorriamo TUTTI i chunk nel database
    for chunk in database["chunks"]:
        # Calcoliamo quanto il chunk è simile alla domanda
        similarita = calcola_similarita_coseno(
            vettore_a=domanda_embedding,
            vettore_b=chunk["embedding"]
        )

        # Salviamo il chunk insieme al suo punteggio
        risultati.append({
            "chunk": chunk,
            "similarita": similarita
        })

    # Ordiniamo i risultati dal più simile al meno simile
    # (reverse=True perché vogliamo i valori più ALTI prima)
    risultati.sort(
        key=lambda risultato: risultato["similarita"],
        reverse=True
    )

    # Restituiamo solo i primi N risultati
    return risultati[:numero_risultati]


# =============================================================
#  PASSO 3: COSTRUZIONE DEL CONTESTO
# =============================================================

def costruisci_contesto(risultati_ricerca):
    """
    Prende i chunk trovati dalla ricerca e li assembla in un
    unico blocco di testo (il "contesto") da passare al modello.

    Per ogni chunk includiamo anche la fonte (nome file e pagina),
    così il modello può citarla nella risposta.
    """

    pezzi_di_contesto = []

    for i, risultato in enumerate(risultati_ricerca):
        chunk = risultato["chunk"]
        similarita = risultato["similarita"]

        # Costruiamo un blocco di testo per ogni chunk
        pezzo = (
            f"[Fonte: {chunk['fonte']}, Pagina {chunk['pagina']}]"
            f" (rilevanza: {similarita:.2f})\n"
            f"{chunk['testo']}"
        )
        pezzi_di_contesto.append(pezzo)

    # Uniamo tutti i pezzi separandoli con una riga vuota
    contesto_completo = "\n\n".join(pezzi_di_contesto)

    return contesto_completo


# =============================================================
#  PASSO 4: GENERAZIONE DELLA RISPOSTA
# =============================================================

def genera_risposta(domanda, contesto, parametri, prompt):
    """
    Invia la domanda e il contesto al modello LLM e restituisce
    la risposta generata.

    Qui avviene la "magia" della RAG:
    - Il modello NON risponde dalla sua conoscenza generale
    - Il modello risponde USANDO SOLO il contesto che gli forniamo
    - Questo riduce le "allucinazioni" (risposte inventate)
    """

    # Leggiamo i parametri dalla configurazione
    modello = parametri.get("MODELLO_LLM", "mistral")
    temperature = ottieni_valore_numerico(parametri, "TEMPERATURE", 0.3)
    top_k = ottieni_valore_numerico(parametri, "TOP_K", 40)
    top_p = ottieni_valore_numerico(parametri, "TOP_P", 0.9)

    # Leggiamo i prompt dalla configurazione
    prompt_di_sistema = prompt.get("PROMPT_DI_SISTEMA", "Rispondi basandoti sul contesto fornito.")
    template_domanda = prompt.get("TEMPLATE_DOMANDA", "Contesto: {contesto}\nDomanda: {domanda}")

    # Costruiamo il messaggio finale sostituendo i segnaposto {contesto} e {domanda}
    messaggio_utente = template_domanda.replace("{contesto}", contesto)
    messaggio_utente = messaggio_utente.replace("{domanda}", domanda)

    # Inviamo la richiesta a Ollama con il formato "chat"
    # Il modello riceve due messaggi:
    #   - "system": le regole generali (prompt di sistema)
    #   - "user": la domanda specifica con il contesto
    risposta = ollama.chat(
        model=modello,
        messages=[
            {
                "role": "system",
                "content": prompt_di_sistema
            },
            {
                "role": "user",
                "content": messaggio_utente
            }
        ],
        options={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    )

    # La risposta di Ollama contiene vari campi;
    # il testo generato si trova in risposta["message"]["content"]
    testo_risposta = risposta["message"]["content"]

    return testo_risposta


# =============================================================
#  CICLO PRINCIPALE: DOMANDA → RISPOSTA
# =============================================================

def avvia_chat():
    """
    Funzione principale che avvia il ciclo interattivo di domanda/risposta.
    """

    stampa_separatore()
    print("  EASY RAG - Cerca e Rispondi")
    stampa_separatore()

    # --- Carichiamo la configurazione ---
    print("\nCaricamento configurazione...")
    parametri, prompt = leggi_configurazione()

    modello_embedding = parametri.get("MODELLO_EMBEDDING", "nomic-embed-text")
    modello_llm = parametri.get("MODELLO_LLM", "mistral")
    numero_risultati = ottieni_valore_numerico(parametri, "NUMERO_RISULTATI", 3)

    print(f"  Modello LLM:      {modello_llm}")
    print(f"  Modello embedding: {modello_embedding}")
    print(f"  Risultati per ricerca: {numero_risultati}")

    # --- Carichiamo il database vettoriale ---
    print("\nCaricamento del database vettoriale...")

    percorso_database = os.path.join("vector_db", "database.json")

    # Verifichiamo che il database esista
    if not os.path.exists(percorso_database):
        print("  ERRORE: Il database vettoriale non è stato trovato!")
        print("  Esegui prima: python 01_indicizza_documenti.py")
        sys.exit(1)

    database = carica_database(percorso_database)
    print(f"  Chunk nel database: {database['numero_chunks']}")

    # --- Ciclo di domanda/risposta ---
    stampa_separatore()
    print("  Pronto! Scrivi la tua domanda e premi Invio.")
    print("  Scrivi 'esci' per terminare.")
    stampa_separatore()

    while True:
        # Chiediamo la domanda all'utente
        print()
        domanda = input("Tu: ").strip()

        # Controlliamo se l'utente vuole uscire
        if domanda.lower() in ["esci", "exit", "quit", "q"]:
            print("\nArrivederci!")
            break

        # Ignoriamo le domande vuote
        if not domanda:
            print("  (Scrivi una domanda o 'esci' per terminare)")
            continue

        # PASSO 1: Trasformiamo la domanda in un vettore
        print("\n  Cerco nei documenti...")
        domanda_embedding = genera_embedding(
            testo=domanda,
            modello_embedding=modello_embedding
        )

        # PASSO 2: Cerchiamo i chunk più simili nel database
        risultati = cerca_chunks_simili(
            domanda_embedding=domanda_embedding,
            database=database,
            numero_risultati=numero_risultati
        )

        # Mostriamo quali fonti sono state trovate
        print("  Fonti trovate:")
        for r in risultati:
            chunk = r["chunk"]
            sim = r["similarita"]
            print(f"    - {chunk['fonte']}, pag. {chunk['pagina']} (rilevanza: {sim:.2f})")

        # PASSO 3: Costruiamo il contesto
        contesto = costruisci_contesto(risultati)

        # PASSO 4: Generiamo la risposta
        print("  Genero la risposta...\n")
        risposta = genera_risposta(
            domanda=domanda,
            contesto=contesto,
            parametri=parametri,
            prompt=prompt
        )

        # Mostriamo la risposta
        stampa_separatore("-")
        print(f"\n{risposta}\n")
        stampa_separatore("-")


# =============================================================
#  PUNTO DI INGRESSO
# =============================================================

if __name__ == "__main__":
    avvia_chat()
