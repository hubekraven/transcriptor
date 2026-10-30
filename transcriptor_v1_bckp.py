import sys
import os
import torch
import whisper
import ollama

# Indiquez ici le dossier exact contenant ffmpeg.exe
dossier_ffmpeg = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + dossier_ffmpeg
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Extensions audio prises en charge
AUDIO_EXTENSIONS = ('.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.m4v')

def transcrire_et_nettoyer(chemin_audio, modele_whisper="small", modele_ollama="mistral"):
    if not os.path.exists(chemin_audio):
        print(f"Erreur : Le fichier '{chemin_audio}' n'existe pas.")
        return

    # 1. Utilisation du GPU (GTX 970) pour Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/3] Chargement du modèle Whisper ('{modele_whisper}') sur : {device.upper()}...")
    
    model = whisper.load_model(modele_whisper, device=device)
    
    print(f"[2/3] Transcription en cours de : {chemin_audio}")
    resultat = model.transcribe(chemin_audio, language="fr")
    texte_brut = resultat["text"]
    
    # Enregistrement de la transcription brute
    with open("transcription_brute.txt", "w", encoding="utf-8") as f:
        f.write(texte_brut)
    print(" -> Fichier 'transcription_brute.txt' généré.")

    # 2. Nettoyage avec Ollama (s'exécute sur le CPU/RAM pour préserver la VRAM)
    print(f"\n[3/3] Nettoyage avec Ollama ({modele_ollama})...")

    # Masquer le GPU temporairement pour le client Ollama s'il communique avec une instance locale
    client_ollama = ollama.Client()

    prompt = f"""Tu es un assistant éditorial expert.
Voici la transcription brute d'un fichier audio :
---
{texte_brut}
---

Consignes de correction :
1. Corrige l'orthographe, la grammaire et la ponctuation.
2. Supprime les hésitations (ex: "euh", "bon", répétitions inutiles).
3. Structure le texte en paragraphes clairs tout en conservant scrupuleusement le sens original et les mots importants.
4. Réponds UNIQUEMENT avec le texte corrigé et structuré, sans aucun commentaire d'introduction ni de conclusion.
"""

    reponse = client_ollama.chat(
        model=modele_ollama,
        messages=[{"role": "user", "content": prompt}]
    )
    
    texte_nettoye = reponse["message"]["content"]

    # Enregistrement du texte nettoyé
    with open("transcription_nettoyee.txt", "w", encoding="utf-8") as f:
        f.write(texte_nettoye)
    
    print(" -> Fichier 'transcription_nettoyee.txt' généré.")
    print("\n--- Aperçu du texte nettoyé ---\n")
    print(texte_nettoye)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcrire.py <fichier_audio>")
        print("Exemple: python transcrire.py enregistrement.mp3")
    else:
        transcrire_et_nettoyer(sys.argv[1])