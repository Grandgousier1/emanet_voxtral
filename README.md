# Projet de Sous-titrage Automatique de la Série YouTube Turque *Emanet*

## Présentation

Ce projet vise à générer automatiquement des sous-titres en français parfaitement synchronisés pour la série YouTube turque *Emanet*, en exploitant exclusivement des modèles d’intelligence artificielle fonctionnant en local sur un environnement GPU dédié (RunPod B200).  
L’objectif est d’assurer une qualité optimale de transcription et traduction contextuelle, tout en minimisant les coûts liés à l’utilisation d’API externes. Le pipeline complet gère la récupération audio, la segmentation, la reconnaissance vocale, la traduction contextuelle, la synchronisation temporelle précise et la production de fichiers `.srt` utilisables directement dans VLC ou tout autre lecteur.

---

## Architecture & Fonctionnalités

- **Téléchargement audio YouTube** : Extraction fiable et robuste via `yt-dlp` avec gestion des erreurs fréquentes liées aux mesures anti-bot de YouTube.  
- **Détection vocale (VAD)** : Identification intelligente des segments parlés pour éviter le traitement des silences et de la musique, optimisant ainsi le temps de calcul et la consommation GPU.  
- **Reconnaissance vocale (ASR)** :  
  - Modèle principal : *Voxtral Small 24B* en local, garantissant un excellent compromis entre vitesse, qualité et usage mémoire.  
  - Fallback automatique vers *Voxtral Mini 3B* si manque de ressources ou erreurs.  
  - Dernier recours : *Faster Whisper* (open source, léger) pour robustesse maximale.  
- **Traduction contextuelle locale** : Utilisation exclusive du modèle LLM *Mistral Small* en local pour traduire précisément chaque segment turc en français naturel, adapté aux dialogues dramatiques.  
- **Génération et synchronisation SRT** : Production des fichiers `.srt` avec horodatage millimétrique, compatible tous lecteurs vidéo.  
- **Batch Processing** : Gestion séquentielle automatisée de playlists entières, avec nettoyage et libération mémoire GPU pour éviter tout crash.  
- **Debug & Logs** : Interface CLI user-friendly avec barres de progression détaillées, logs explicites, gestion d’erreurs améliorée, vérifications pré-exécution de toutes dépendances et fonctions critiques.  
- **Configuration GPU** : Adaptation automatique des paramètres (batch size, mémoire) selon capacité détectée.  

---

## Installation & Prérequis

### Environnement

- OS : Ubuntu (compatible Fedora, testé sur RunPod B200 GPU instance)  
- GPU : minimum 12 Go VRAM recommandé (RunPod B200 idéal)  
- Python 3.10+ recommandé

### Installation

1. Cloner le projet :  
   ```bash
   git clone <URL_DU_PROJET>
   cd <PROJET>
````

2. Installer les dépendances Python :

   ```bash
   pip install -r requirements.txt
   ```

3. Préparer l’environnement GPU et modèles (exécuté automatiquement via `Makefile`) :

   ```bash
   make setup
   ```

---

## Usage

### Lancer la génération d’un seul épisode

```bash
python main.py --youtube_url <URL_YOUTUBE> --output <FICHIER_SORTIE.srt>
```

Options importantes :

* `--use_voxtral_mini` : Forcer l’usage du modèle Voxtral Mini 3B (moins lourd mais moins rapide)
* `--debug` : Activer logs détaillés pour diagnostic
* `--batch_size <N>` : Ajuster la taille du batch (mémoire GPU)

### Traitement en batch

Lister plusieurs URLs dans un fichier texte, une URL par ligne, puis lancer :

```bash
make batch FILE=liste_videos.txt
```

Le pipeline traitera chaque épisode séquentiellement, libérant la mémoire GPU entre les traitements.

---

## Structure du projet

```
.
├── Makefile               # commandes d’installation, debug, exécution batch
├── requirements.txt       # dépendances Python optimisées
├── setup_runpod.sh        # script d’environnement et préchargement modèles
├── main.py                # script principal pipeline complet
├── utils/
│   └── gpu_utils.py       # fonctions utilitaires gestion GPU
└── README.md              # documentation projet
```

---

## Notes & Conseils

* **Choix des modèles** :
  Le modèle Voxtral Small 24B est un excellent compromis qualité/rapidité.
  Le fallback automatique vers Voxtral Mini 3B et Faster Whisper garantit la robustesse même sur GPU moins puissants ou en cas de surcharge.
* **VAD** : Limiter le traitement aux segments vocaux réduit significativement le coût en calcul et le temps.
* **Synchronisation** : L’utilisation combinée de WhisperX-like alignement et VAD garantit des sous-titres parfaitement synchronisés.
* **Optimisation GPU** : Le script analyse la mémoire disponible et ajuste ses batchs automatiquement pour ne jamais saturer la mémoire GPU, évitant ainsi plantages et ralentissements.
* **Robustesse YouTube** : `yt-dlp` avec options spécifiques et gestion des erreurs assure un téléchargement stable, même face aux protections anti-bot.
* **Débogage** : Toujours lancer en mode debug pour les premières exécutions afin de vérifier intégrité et performance.

---

## Perspectives d’amélioration

* Intégration d’un module de post-édition automatique pour améliorer la fluidité des sous-titres.
* Ajout de diarisation et reconnaissance multi-locuteurs.
* Interface graphique simple pour utilisateurs non techniques.
* Déploiement conteneurisé Docker pour portabilité maximale.

---

## Licence

Projet open-source, libre d’utilisation et modification sous licence MIT.

```
```
