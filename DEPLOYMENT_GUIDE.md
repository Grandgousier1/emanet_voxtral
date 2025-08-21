
# Guide de Déploiement Critique - Projet Voxtral sur RunPod B200

Ce document fournit la procédure et la checklist finale pour le déploiement de l'application Voxtral sur une infrastructure haute performance. Le respect de ces étapes est crucial pour garantir une exécution réussie et éviter les coûts liés à un échec.

## 1. Contexte : Déploiement "One-Shot"

L'objectif est une exécution sans erreur du début à la fin. L'application a été renforcée avec plusieurs mécanismes de validation et de sécurité pour atteindre cet objectif.

- **Validation de l'environnement** : Le `Makefile` contient une cible `validate-all-b200` qui vérifie les dépendances critiques, l'API B200 et l'empreinte mémoire.
- **Validation des données** : Le traitement des tâches inclut désormais une vérification de la corruption des fichiers audio.
- **Robustesse du traitement parallèle** : Le processeur `asyncio` intègre des timeouts pour prévenir les blocages (deadlocks).
- **Vérifications système** : Le script principal vérifie maintenant l'espace disque et la validité du token Hugging Face avant de démarrer.

## 2. Procédure de Déploiement Standard

1.  **Préparation du Pod** : Lancez un pod RunPod avec l'image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`.

2.  **Clonage du Projet** : Clonez le dépôt sur le pod.
    ```bash
    git clone <URL_DU_DEPOT>
    cd voxtral
    ```

3.  **Installation** : Installez les dépendances. Le projet est configuré pour utiliser la version de PyTorch de l'image.
    ```bash
    make install
    ```

4.  **Validation Complète** : Exécutez la suite de validation complète. Le nouveau script d'orchestration vous fournira une expérience guidée et un résumé clair.
    ```bash
    make validate-all-b200
    ```
    Si cette commande échoue, n'allez pas plus loin. Le script vous indiquera l'étape qui a échoué.

5.  **Exécution** : Lancez le script principal avec les arguments souhaités.

## 3. Checklist de Déploiement Finale

À vérifier manuellement dans le terminal du pod, **juste avant de lancer `main.py`** :

-   `[ ]` **Token Hugging Face** : Exécutez `echo $HF_TOKEN`. Assurez-vous que la variable d'environnement est définie si vous utilisez des modèles privés.
-   `[ ]` **Accès au Modèle** : Exécutez `curl https://huggingface.co/api/models/mistralai/Voxtral-Small-24B-2507`. Vérifiez que la réponse n'est pas une erreur 404 ou 401.
-   `[ ]` **Espace Disque** : Exécutez `df -h /`. Assurez-vous qu'il y a au moins 25 Go d'espace libre.
-   `[ ]` **Configuration** : Vérifiez que le fichier `config.py` ou les variables d'environnement correspondent à la configuration souhaitée (modèle, langue, etc.).
-   `[ ]` **Dry-Run (Optionnel mais recommandé)** : Effectuez un test à blanc avec un seul fichier pour vérifier le pipeline de bout en bout.
    ```bash
    python main.py --url <URL_TEST> --output test.srt --dry-run
    ```
