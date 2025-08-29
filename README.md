# 🚀 EMANET VOXTRAL - Générateur de Sous-titres B200

## 🏆 Vue d'ensemble

Ce système a été **complètement refactorisé et optimisé** pour la génération de sous-titres multilingues à partir de sources audio/vidéo, avec une optimisation spécifique pour les GPU NVIDIA B200.

## 📋 Prérequis Système

Avant de lancer l'application, assurez-vous d'avoir les dépendances système suivantes installées :

- **Python**: Version 3.9+
- **Git**: Pour cloner le dépôt.
- **ffmpeg**: Une dépendance critique pour tout traitement audio et vidéo.

Sur les systèmes basés sur Debian (comme Ubuntu ou l'environnement Runpod), vous pouvez installer `ffmpeg` avec :
`sudo apt-get update && sudo apt-get install ffmpeg`

## 🛠️ Installation

Pour installer le projet et toutes ses dépendances :

```bash
# 1. Cloner le projet
git clone https://github.com/Grandgousier1/emanet_voxtral.git
cd emanet_voxtral

# 2. Lancer la configuration complète
make setup
```

## 🚀 Démarrage Rapide

Une fois l'installation terminée, vous pouvez lancer un traitement :

```bash
# Traitement d'une URL YouTube ou d'un fichier local
make start URL='https://www.youtube.com/watch?v=VOTRE_VIDEO_ID' OUTPUT='votre_fichier.srt'

# Traitement par lot (liste d'URLs/chemins dans un fichier videos.txt)
make batch LIST='videos.txt' DIR='./resultats'
```

## ✅ Validation & Diagnostic

Pour vérifier l'état de votre environnement et des dépendances :

```bash
# Validation complète de l'environnement
make validate

# Vérification des conflits de dépendances Python
make check-deps
```

## 🔑 Configuration du Token Hugging Face

Pour utiliser les modèles Voxtral, vous devez configurer votre token Hugging Face :

```bash
make setup-token
```

## 📚 Documentation Complète

Pour des informations détaillées sur les modes d'exécution avancés, le monitoring, le débogage et les optimisations, veuillez consulter la documentation complète du projet.

---
