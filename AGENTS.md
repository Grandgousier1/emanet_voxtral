# Documentation des Agents et Modules - Projet Emanet RunPod

Ce document décrit les différents agents (modules, scripts, fonctions majeures) constituant le pipeline de génération automatique de sous-titres pour la série turque *Emanet*. Il présente leurs responsabilités, interactions et spécificités.

---

## 1. `main.py` — Agent principal du pipeline

- **Rôle** :  
  Coordonne l’ensemble du processus de bout en bout, depuis le téléchargement de la vidéo YouTube jusqu’à la production du fichier `.srt` final en français.  
- **Fonctions clés** :  
  - Parsing des arguments CLI (URL YouTube, options de modèle, debug...)  
  - Lancement du téléchargement audio avec `yt-dlp` et gestion des erreurs  
  - Application de la détection d’activité vocale (VAD) pour segmenter l’audio  
  - Appel successif aux modèles de reconnaissance vocale (ASR) : Voxtral Small, fallback Mini, puis Faster Whisper  
  - Traduction locale avec le LLM Mistral Small, segment par segment  
  - Génération finale du fichier `.srt` avec synchronisation précise  
  - Gestion automatique de la mémoire GPU et nettoyage entre épisodes (batch processing)  
- **Interactions** : Utilise `gpu_utils.py` pour contrôle GPU, interagit avec les modèles locaux via API ou bindings Python.

---

## 2. `utils/gpu_utils.py` — Agent de gestion GPU

- **Rôle** :  
  Fournit des fonctions utilitaires pour :  
  - Vérifier la mémoire GPU disponible avant traitement  
  - Adapter la taille des batchs selon la VRAM détectée  
  - Forcer la libération de mémoire GPU après chaque étape critique  
  - Surveiller la consommation GPU pendant l’exécution (logs debug)  
- **Importance** : Assure la stabilité du pipeline sur un GPU unique limité (RunPod B200), évitant crashes et lenteurs.

---

## 3. Modèles ASR locaux (Voxtral Small, Mini, Faster Whisper)

- **Rôle** :  
  Reconnaissance vocale en turc des segments audio.  
- **Voxtral Small (24B)** : Modèle principal, haute qualité et rapidité optimisée sur GPU.  
- **Voxtral Mini (3B)** : Modèle fallback plus léger, utilisé en cas d’insuffisance de ressources.  
- **Faster Whisper** : Modèle open source léger, dernier recours pour robustesse maximale.  
- **Intégration** : Chargement et exécution via PyTorch localement dans le pipeline.

---

## 4. Modèle LLM de traduction locale (Mistral Small)

- **Rôle** :  
  Traduction contextuelle segment par segment, du turc vers un français naturel et fluide adapté à un dialogue dramatique.  
- **Exigences** : Fonctionne exclusivement en local, minimisant coûts et dépendances à des APIs externes.  
- **Intégration** : Interfacé via pipeline PyTorch, appelle la génération texte par segment.

---

## 5. Module de détection d’activité vocale (VAD)

- **Rôle** :  
  Identification des segments pertinents contenant de la parole pour réduire le volume de données à traiter.  
- **Bénéfices** : Optimisation du temps et ressources GPU, meilleure qualité finale en évitant le bruit/musique/silence.  
- **Méthode** : Utilisation d’algorithmes VAD open-source ou intégrés selon disponibilité.

---

## 6. Makefile

- **Rôle** :  
  Faciliter l’installation, l’exécution, le batch processing, le debug et le nettoyage via commandes simples et documentées.  
- **Commandes principales** :  
  - `make setup` : Prépare l’environnement, installe dépendances, télécharge modèles.  
  - `make run` : Exécution d’un traitement simple sur un URL donné.  
  - `make batch FILE=liste.txt` : Traitement batch séquentiel avec nettoyage GPU.  
  - `make debug` : Active logs détaillés.  
  - `make clean` : Nettoyage des fichiers temporaires.

---

## 7. Gestion des erreurs et logs

- **Rôle** :  
  - Capturer et rapporter toutes erreurs critiques (dépendances, GPU, téléchargement, modélisation).  
  - Fournir des messages clairs et instructifs pour un debugging rapide.  
  - Afficher des barres de progression et compte-rendu d’étapes.  
- **Implémentation** : Intégré dans `main.py` et `gpu_utils.py` avec sorties colorées et timestampées.

---

## Conclusion

Chaque agent/module est conçu pour garantir robustesse, efficacité et facilité d’usage dans un environnement local GPU contraint. L’interopérabilité entre ces agents permet d’assurer un workflow fluide et maintenable pour la génération automatisée de sous-titres.

---
