# Projet de Pipeline de Sous-titrage Automatisé

## 1. Présentation

Ce projet fournit un pipeline de sous-titrage automatisé, robuste et configurable. Il a été conçu pour transformer des vidéos (notamment depuis YouTube) en fichiers de sous-titres (`.srt`) de haute qualité.

Le système est entièrement basé sur des modèles d'intelligence artificielle open-source fonctionnant en local. Il gère l'ensemble du processus : téléchargement, transcription, traduction, et génération de fichiers de sous-titres synchronisés. L'objectif est d'offrir une solution flexible et puissante, sans dépendre d'API externes payantes.

---

## 2. Architecture & Fonctionnalités

Le pipeline a été entièrement refactorisé pour garantir la stabilité et la flexibilité.

- **Traitement par lots (Batch Processing)** : Le script est conçu pour traiter une liste d'URLs de vidéos fournie dans un fichier texte. Les modèles d'IA ne sont chargés qu'une seule fois par session, optimisant considérablement les performances pour les traitements en série.

- **Configuration Hybride** :
  - Un fichier central `config.yaml` définit tous les paramètres par défaut (modèles, langues, etc.).
  - Chaque paramètre peut être surchargé dynamiquement via des arguments en ligne de commande, offrant une flexibilité maximale pour des exécutions spécifiques.

- **Chaîne de Transcription (ASR) avec Fallback** : Pour garantir la robustesse, le système utilise une chaîne de modèles de reconnaissance vocale. Si le premier échoue, il passe automatiquement au suivant :
  1. **Voxtral Small** (par défaut, haute qualité)
  2. **Voxtral Mini** (modèle de secours plus léger)
  3. **Faster-Whisper (small)** (dernier recours, très robuste)

- **Traduction Neuronale** : La traduction est gérée par le modèle **NLLB (No Language Left Behind)** de Meta, permettant de traduire depuis et vers des centaines de langues.

- **Robustesse du téléchargement YouTube** : Pour contourner les mesures anti-bot de YouTube, le script intègre une option `--cookies` permettant de passer un fichier de cookies à `yt-dlp` pour s'authentifier.

- **Gestion de la Mémoire** : Des utilitaires de gestion de la mémoire GPU sont intégrés pour libérer les ressources après chaque étape critique, assurant la stabilité lors de longs traitements.

- **Gestion des Dépendances** : Toutes les dépendances Python sont figées (`pinned`) dans des fichiers `requirements.txt` (pour GPU) et `requirements-cpu.txt` (pour CPU seulement), garantissant une reproductibilité parfaite de l'environnement.

---

## 3. Installation & Prérequis

### Prérequis

- **OS** : Linux (testé sur Ubuntu)
- **Python** : 3.10 ou supérieur
- **GPU** : Recommandé pour des performances optimales (NVIDIA, >12Go VRAM). Une installation CPU est également supportée.
- **ffmpeg** : Doit être installé et accessible dans le PATH (`sudo apt update && sudo apt install ffmpeg`).

### Installation

1.  **Cloner le projet :**
    ```bash
    git clone <URL_DU_PROJET>
    cd <NOM_DU_DOSSIER>
    ```

2.  **Créer l'environnement virtuel et installer les dépendances :**

    Le `Makefile` automatise la création de l'environnement. Choisissez l'une des deux options :

    - **Pour un environnement GPU (recommandé) :**
      ```bash
      make install
      ```
    - **Pour un environnement CPU uniquement :**
      ```bash
      make install-cpu
      ```

---

## 4. Usage

Le script est conçu pour être lancé via le `Makefile` pour plus de simplicité.

### Commande principale (Traitement par lots)

La méthode d'utilisation principale consiste à créer un fichier texte (ex: `videos.txt`) contenant une liste d'URLs de vidéos, une par ligne.

**Exemple de `videos.txt` :**
```
https://www.youtube.com/watch?v=xxxxxxxxxxx
https://www.youtube.com/watch?v=yyyyyyyyyyy
```

Lancez ensuite le traitement avec la commande `make batch` :
```bash
make batch BATCH_FILE=videos.txt
```
Les fichiers de sortie (audio, vidéo, `.srt`) seront sauvegardés dans le dossier `output/`.

### Arguments et Surcharges

Vous pouvez surcharger les paramètres du `config.yaml` directement depuis la ligne de commande.

**Exemple : Lancer le traitement en changeant la langue de traduction en espagnol :**
```bash
# La commande est passée au script python via la variable ARGS
make batch BATCH_FILE=videos.txt ARGS="--target-lang spa_Latn"
```

**Exemple : Utiliser un fichier de cookies pour le téléchargement :**
```bash
make batch BATCH_FILE=videos.txt ARGS="--cookies /chemin/vers/cookies.txt"
```

---

## 5. Configuration (`config.yaml`)

Le fichier `config.yaml` est au cœur du pipeline. Il permet de définir les paramètres par défaut de manière claire et centralisée.

- **`model_paths`**: Spécifie les identifiants Hugging Face ou les chemins locaux pour tous les modèles (Voxtral, faster-whisper, NLLB).
- **`transcription_options`**: Options pour la phase de transcription, comme le `device` (`cuda` ou `cpu`).
- **`translation_options`**: Options pour la traduction, incluant le `device`, la langue source (`source_lang`) et la langue cible (`target_lang`).
- **`youtube_dl_options`**: Options pour `yt-dlp`, comme le dossier de sortie (`output_dir`) ou l'activation/désactivation du téléchargement vidéo/audio.

---

## 6. Structure du projet

```
.
├── Makefile               # Commandes pour installer, formater, et lancer le pipeline
├── README.md              # Cette documentation
├── config.yaml            # Fichier de configuration central
├── main.py                # Script principal du pipeline
├── requirements.txt       # Dépendances Python pour environnement GPU
├── requirements-cpu.txt   # Dépendances Python pour environnement CPU
└── utils/
    └── gpu_utils.py       # Fonctions utilitaires pour la gestion de la mémoire GPU
```

---

## 7. Notes & Conseils

- **Stabilité** : Le traitement par lots et la gestion de la mémoire ont été conçus pour des heures d'exécution sans interruption.
- **Flexibilité** : La combinaison de `config.yaml` et des surcharges par ligne de commande permet d'adapter facilement le pipeline à différents besoins sans modifier le code.
- **Dépendances** : L'utilisation de versions figées garantit que le script fonctionnera de la même manière dans le futur. Si vous rencontrez des problèmes d'installation, cela peut être dû à une incompatibilité de votre version de Python avec les versions des paquets.

---

## 8. Licence

Ce projet est open-source et distribué sous la licence MIT.
```
