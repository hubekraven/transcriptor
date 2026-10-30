import sys
import os
import time
import torch
import whisper
import ollama

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
    dossier_sortie=None, 
    model_whisper="small", 
    modele_ollama="qwen2.5:1.5b"
):
    if not os.path.exists(dossier_source):
        print(f"Erreur : Le dossier source '{dossier_source}' n'existe pas.")
        return

    if not dossier_sortie:
        dossier_sortie = dossier_source

    os.makedirs(dossier_sortie, exist_ok=True)

    fichiers_audio = [
        f for f in os.listdir(dossier_source)
        if os.path.isfile(os.path.join(dossier_source, f)) and f.lower().endswith(AUDIO_EXTENSIONS)
    ]

    if not fichiers_audio:
        print(f"Aucun fichier audio supporté trouvé dans '{dossier_source}'.")
        return

    print(f"Dossier source : {dossier_source}")
    print(f"Dossier de sortie : {dossier_sortie}")
    print(f"Nombre de fichiers audio détectés : {len(fichiers_audio)}")

    verifier_et_telecharger_modele_ollama(modele_ollama)

    # 1. Utilisation du GPU pour Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/3] Chargement du modèle Whisper ('{model_whisper}') sur : {device.upper()}...")
    
    model_whisper = whisper.load_model(model_whisper, device=device)

    total = len(fichiers_audio)
    temps_debut_total = time.perf_counter()

    for idx, fichier in enumerate(fichiers_audio, 1):
        temps_debut_fichier = time.perf_counter()
        duree_whisper = 0.0
        duree_ollama = 0.0

        chemin_audio = os.path.join(dossier_source, fichier)   
        nom_de_base = os.path.splitext(fichier)[0]

        fichier_brut_out = os.path.join(dossier_sortie, f"{nom_de_base}_brut.txt")
        fichier_nettoye_out = os.path.join(dossier_sortie, f"{nom_de_base}.txt")

        print("\n" + "=" * 60)
        print(f"[{idx}/{total}] Traitement de : {fichier}")
        print("=" * 60)

        # A. Transcription Whisper
        try:
            print(" -> Transcription avec Whisper...")
            t0_whisper = time.perf_counter()
            resultat = model_whisper.transcribe(chemin_audio, language="fr")
            duree_whisper = time.perf_counter() - t0_whisper
            
            segments = resultat.get("segments", [])
            if segments:
                phrases = [seg["text"].strip() for seg in segments if seg["text"].strip()]
                paragraphes = []
                for i in range(0, len(phrases), 3):
                    paragraphes.append(" ".join(phrases[i:i+3]))
                texte_brut = "\n\n".join(paragraphes)
            else:
                texte_brut = resultat["text"].strip()

            with open(fichier_brut_out, "w", encoding="utf-8") as f:
                f.write(texte_brut)
            
            print(f" -> Transcription brute enregistrée en {duree_whisper:.2f}s : {fichier_brut_out}")

        except Exception as e:
            print(f" Erreur lors de la transcription Whisper de '{fichier}' : {e}")
            continue

        # B. Nettoyage Ollama
        try:
            print(f" -> Nettoyage avec Ollama ({modele_ollama})...")
            t0_ollama = time.perf_counter()

            reponse = ollama.chat(
                model=modele_ollama,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un transformateur de texte strict. Tu ne dois renvoyer QUE le texte traité. "
                            "Aucun titre, aucun conseil, aucun préambule, aucune remarque d'introduction ou de conclusion."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Nettoie la grammaire, la ponctuation et découpe en paragraphes clairs le texte suivant :\n\n{texte_brut}"
                    }
                ],
                options={"num_gpu": 0}
            )
            
            duree_ollama = time.perf_counter() - t0_ollama
            texte_nettoye = reponse["message"]["content"].strip()

            with open(fichier_nettoye_out, "w", encoding="utf-8") as f:
                f.write(texte_nettoye)
            
            print(f" -> Nettoyage enregistré en {duree_ollama:.2f}s : {fichier_nettoye_out}")

        except Exception as e:
            duree_ollama = time.perf_counter() - t0_ollama
            print(f" Erreur lors du nettoyage Ollama pour '{fichier}' ({duree_ollama:.2f}s) : {e}")
            print(" Conservation uniquement de la version brute pour ce fichier.")

        # Bilan de durée pour ce fichier
        duree_totale_fichier = time.perf_counter() - temps_debut_fichier
        print(f" [Chrono] Fichier traité en {duree_totale_fichier:.2f}s (Whisper: {duree_whisper:.2f}s | Ollama: {duree_ollama:.2f}s)")

    temps_total = time.perf_counter() - temps_debut_total
    minutes, secondes = divmod(temps_total, 60)

    print("\n" + "=" * 60)
    print(" Traitement terminé pour l'ensemble des fichiers !")
    print(f" Temps total écoulé : {int(minutes)}m {secondes:.2f}s")
    print(f" Les transcriptions finales sont disponibles dans : {os.path.abspath(dossier_sortie)}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python transcriptor.py <dossier_audio> [dossier_output]")
    else:
        dossier_source = sys.argv[1]
        dossier_sortie = sys.argv[2] if len(sys.argv) > 2 else "output"
        transcrire_et_nettoyer_dossier(dossier_source, dossier_sortie)