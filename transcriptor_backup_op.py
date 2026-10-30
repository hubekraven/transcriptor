import sys
import os
import torch
from faster_whisper import WhisperModel
import ollama
import time





# Indiquez ici le dossier exact contenant ffmpeg.exe
dossier_ffmpeg = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + dossier_ffmpeg

# Extensions audio prises en charge
AUDIO_EXTENSIONS = ('.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.m4v')

def verifier_et_telecharger_modele_ollama(modele_ollama):
    """Vérifie si le modèle Ollama est disponible localement, sinon le télécharge."""
    try:
        modeles_locaux = [m.model for m in ollama.list().models]
        if not any(modele_ollama in m for m in modeles_locaux):
            print(f"Modèle Ollama '{modele_ollama}' non trouvé localement. Téléchargement en cours...")
            ollama.pull(modele_ollama)
            print(f"Modèle '{modele_ollama}' téléchargé avec succès.")
    except Exception as e:
        print(f"Avertissement lors de la vérification du modèle Ollama : {e}")



def transcrire_et_nettoyer_dossier(
    dossier_source, 
    dossier_sortie = None, 
    model_whisper = WhisperModel("small", device="cuda", compute_type="float16"),
    modele_ollama="qwen2.5:1.5b" # or use mistral
):
    if not os.path.exists(dossier_source):
        print(f"Erreur : Le dossier source '{dossier_source}' n'existe pas.")
        return

    if not dossier_sortie: 
        dossier_sortie = dossier_source

    # Création du dossier d'extrants s'il n'existe pas
    os.makedirs(dossier_sortie, exist_ok=True)

    # Recuperation des fichiers audio
    fichiers_audio = [
        f for f in os.listdir(dossier_source)
        if os.path.isfile(os.path.join(dossier_source, f)) and f.lower().endswith(AUDIO_EXTENSIONS)
    ]

    if not fichiers_audio:
        print(f"Aucun fichier audio supporté trouvé dans '{dossier_source}'.")
        print(f"Extensions acceptées : {', '.join(AUDIO_EXTENSIONS)}")
        return


    print(f"Dossier source : {dossier_source}")
    print(f"Dossier de sortie : {dossier_sortie}")
    print(f"Nombre de fichiers audio détectés : {len(fichiers_audio)}")

    # # Vérification du modèle Ollama
    verifier_et_telecharger_modele_ollama(modele_ollama)

    # 1. Utilisation du GPU (GTX 970) pour Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/3] Chargement du modèle Whisper ('{model_whisper}') sur : {device.upper()}...")
    
    total = len(fichiers_audio)
    temps_total_start = time.perf_counter()

    for idx, fichier in enumerate(fichiers_audio, 1):
        chemin_audio = os.path.join(dossier_source, fichier)  
        nom_de_base= os.path.splitext(fichier)[0]
        temps_whisper_end = 0.0

        # Noms des fichiers de sortie .txt
        fichier_brut_out = os.path.join(dossier_sortie, f"{nom_de_base}_brut.txt")
        fichier_nettoye_out = os.path.join(dossier_sortie, f"{nom_de_base}.txt")

        print("\n" + "=" * 60)
        print(f"[{idx}/{total}] Traitement de : {fichier}")
        print("=" * 60)

    # A. Transcription Whisper
        try:
            print(" -> Transcription avec Whisper...")
            temps_whisper_start = time.perf_counter()

            # Transcription du fichier
            segments, info = model_whisper.transcribe(chemin_audio, language="fr", beam_size=5)
            
            # Reconstruction du texte ligne par ligne à partir des segments
            if segments:
                phrases = [seg.text.strip() for seg in segments if seg.text.strip()]
                
                # Regroupement des phrases en paragraphes de 3-4 phrases
                paragraphes = []
                for i in range(0, len(phrases), 3):
                    paragraphes.append(" ".join(phrases[i:i+3]))
                texte_brut = "\n\n".join(paragraphes)
            else:
                texte_brut = "\n\n".join(paragraphes)

            with open(fichier_brut_out, "w", encoding="utf-8") as f:
                f.write(texte_brut)

            temps_whisper_end = time.perf_counter() - temps_whisper_start

            print(f" -> Transcription brute enregistrée : {fichier_brut_out}")
            print(f" -> Durée de l'audio: {info.duration}")
            print(f" -> Langue de l'audio: {info.language}")
            print(f" -> Temps de traitement: {temps_whisper_end:.2f}s")

        except Exception as e:
            print(f" Erreur lors de la transcription Whisper de '{fichier}' : {e}")
            continue

   # B. Nettoyage Ollama
        # try:
        #     print(f" -> Nettoyage avec Ollama ({modele_ollama})...")
        #     # Forcer Ollama à utiliser le CPU pour éviter le bug CUDA sur GTX 970
        #     os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
        #     prompt = (
        #         "Tu es un simple outil de retranscription et de mise en page. "
        #         "Ta seule tâche est de nettoyer et découper le texte ci-dessous en paragraphes.\n\n"
        #         "RÈGLES STRICTES :\n"
        #         "- Ne génère AUCUN titre, AUCUNE introduction, AUCUNE conclusion.\n"
        #         "- Ne génère AUCUN conseil, remarque ou commentaire (ex: 'Conseils de rédaction', 'Correction', etc.).\n"
        #         "- Ne modifie pas le sens du texte.\n"
        #         "- Retourne EXCLUSIVEMENT le texte corrigé et structuré en paragraphes.\n\n"
        #         "Texte brut à traiter :\n"
        #         "---\n"
        #         f"{texte_brut}\n"
        #         "---"
        #     )

        #     # temps_ollama_start = time.perf_counter()
        #     reponse = ollama.chat(
        #         model=modele_ollama,
        #         messages=[{"role": "user", "content": prompt}],
        #         options={"num_gpu": 0 } # L'option num_gpu=0 force Ollama à charger le modèle uniquement en RAM/CPU
        #     )
            
        #     texte_nettoye = reponse["message"]["content"].strip()

        #     # Sauvegarde de la transcription nettoyée
        #     with open(fichier_nettoye_out, "w", encoding="utf-8") as f:
        #         f.write(texte_nettoye)
        #    temps_ollama_end = time.perf_counter() - temps_ollama_start
        #     print(f" -> Transcription nettoyée enregistrée : {fichier_nettoye_out}")
        #     print(f" -> Temp de traitement : {temps_ollama_end:.2f}s")    
        # except Exception as e:
        #     print(f" Erreur lors du nettoyage Ollama pour '{fichier}' : {e}")
        #     print(f" Conservation uniquement de la version brute pour ce fichier.")
    
    #Calcule temps de traitement total
    temps_total_end = time.perf_counter() - temps_total_start
    minutes, secondes = divmod(temps_total_end, 60)

    print("\n" + "=" * 60)
    print(" Traitement terminé pour l'ensemble des fichiers !")
    print(f"Temps total de traitement: {int(minutes)}m {secondes:.2f}s")
    print(f"Les transcriptions finales sont disponibles dans : {os.path.abspath(dossier_sortie)}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python transcriptor.py <dossier_audio> [dossier_output]")
        print("Exemple : python transcriptor.py ./mes_audios")
        print("Exemple : python transcriptor.py ./mes_audios ./output")
    else:
        dossier_source = sys.argv[1]
        dossier_sortie = sys.argv[2] if len(sys.argv) > 2 else dossier_source
        transcrire_et_nettoyer_dossier(dossier_source, dossier_sortie)