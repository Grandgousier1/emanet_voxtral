# 🚀 EMANET VOXTRAL - Guide Utilisateur Complet

## 🎯 Introduction

EMANET VOXTRAL est un générateur de sous-titres intelligent optimisé pour GPU B200, utilisant les modèles Voxtral de Mistral AI pour une traduction de haute qualité.

## 🚦 Démarrage Rapide

### Pour les Débutants 

```bash
# Interface simplifiée guidée
python quick_start.py
```

### Pour les Utilisateurs Expérimentés

```bash
# Assistant interactif complet
python main_enhanced.py --wizard

# Traitement direct
python main.py --url "https://youtube.com/watch?v=..." --output sous_titres.srt
```

## 📋 Prérequis

### Obligatoires
- **Python 3.8+**
- **Token Hugging Face** (gratuit) → [Créer un compte](https://huggingface.co)
- **25GB espace disque libre** minimum
- **Connexion Internet** pour téléchargement des modèles

### Recommandés
- **GPU B200** avec 180GB VRAM (optimal)
- **188GB RAM** pour performances maximales
- **28+ vCPU** pour traitement parallèle optimal

## 🔧 Installation et Configuration

### 1. Installation des Dépendances

```bash
# Installation automatique
pip install -r requirements.txt

# Ou installation manuelle des composants critiques
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers vllm mistral-common rich soundfile
```

### 2. Configuration Token Hugging Face

**Méthode 1: Via l'interface**
```bash
python quick_start.py  # Suivre le guide intégré
```

**Méthode 2: Manuelle**
```bash
# Créer fichier .env
echo "HF_TOKEN=votre_token_ici" > .env

# Ou variable d'environnement
export HF_TOKEN=votre_token_ici
```

### 3. Vérification de l'Installation

```bash
# Diagnostic complet
python validator.py

# Ou diagnostic simple
python quick_start.py  # Option 5: Diagnostic système
```

## 🎬 Modes d'Utilisation

### Mode 1: Traitement Simple

**YouTube:**
```bash
python main.py \
  --url "https://youtube.com/watch?v=dQw4w9WgXcQ" \
  --output "rick_roll_fr.srt" \
  --target-lang fr \
  --quality best
```

**Fichier Local:**
```bash
python main.py \
  --url "/chemin/vers/video.mp4" \
  --output "sous_titres.srt" \
  --target-lang fr
```

### Mode 2: Traitement en Lot

**1. Créer un fichier de lot (batch.txt):**
```
https://youtube.com/watch?v=video1
https://youtube.com/watch?v=video2
/chemin/vers/fichier_local.mp4
/chemin/vers/audio.mp3
```

**2. Lancer le traitement:**
```bash
python main.py \
  --batch-list batch.txt \
  --output-dir ./resultats/ \
  --target-lang fr \
  --max-workers 4
```

### Mode 3: Assistant Interactif (Recommandé)

```bash
# Interface complète avec guidance
python main_enhanced.py --wizard

# Interface simplifiée
python quick_start.py
```

## 🌍 Langues Supportées

| Code | Langue | Code | Langue |
|------|--------|------|--------|
| `fr` | 🇫🇷 Français | `en` | 🇺🇸 English |
| `es` | 🇪🇸 Español | `de` | 🇩🇪 Deutsch |
| `it` | 🇮🇹 Italiano | `pt` | 🇵🇹 Português |
| `ru` | 🇷🇺 Русский | `zh` | 🇨🇳 中文 |
| `ja` | 🇯🇵 日本語 | `ar` | 🇸🇦 العربية |

## ⚙️ Configuration Avancée

### Paramètres de Performance

```bash
# Optimisation GPU B200
python main.py \
  --url "..." \
  --gpu-memory-limit 0.9 \
  --batch-size 64 \
  --precision bf16 \
  --max-workers 8

# Mode haute qualité
python main.py \
  --url "..." \
  --quality best \
  --model voxtral-small \
  --min-quality-score 0.8 \
  --retry-failed-segments
```

### Paramètres Audio

```bash
# Contrôle détection vocale
python main.py \
  --url "..." \
  --vad-threshold 0.3 \
  --min-segment-duration 1.0

# Qualité audio élevée
python main.py \
  --url "..." \
  --audio-quality high
```

### Monitoring et Debug

```bash
# Avec monitoring temps réel
python main.py --url "..." --monitor --telemetry

# Mode debug complet
python main.py --url "..." --debug --verbose --log-level DEBUG
```

## 🎛️ Options Complètes

### Sources d'Entrée
- `--url URL` : YouTube ou fichier local
- `--batch-list FILE` : Fichier contenant liste URLs/chemins

### Sortie
- `--output FILE` : Fichier SRT de sortie (mode simple)
- `--output-dir DIR` : Répertoire de sortie (mode batch)

### Langue et Qualité
- `--target-lang LANG` : Langue cible (défaut: fr)
- `--quality LEVEL` : fast/balanced/best (défaut: balanced)
- `--model MODEL` : voxtral-small/voxtral-mini

### Performance
- `--max-workers N` : Workers parallèles (défaut: 4)
- `--batch-size N` : Taille lot GPU (défaut: 32)
- `--gpu-memory-limit FLOAT` : Limite mémoire GPU 0.1-0.95
- `--precision TYPE` : fp16/bf16/fp32

### Contrôle Qualité
- `--min-quality-score FLOAT` : Score qualité minimal (0.0-1.0)
- `--retry-failed-segments` : Réessayer segments échoués
- `--quality-check` : Validation post-traitement

### Modes Spéciaux
- `--wizard` : Assistant interactif
- `--setup` : Configuration système
- `--validate` : Diagnostic complet
- `--tutorial` : Guide d'utilisation
- `--dry-run` : Simulation sans traitement

### Debug et Monitoring
- `--debug` : Mode debug détaillé
- `--monitor` : Interface monitoring temps réel
- `--telemetry` : Métriques avancées
- `--verbose` : Sortie détaillée
- `--log-level LEVEL` : DEBUG/INFO/WARNING/ERROR

## 📊 Monitoring et Performances

### Interface Monitoring Temps Réel

```bash
# Activer le monitoring
python main.py --url "..." --monitor
```

L'interface affiche:
- 🎮 Utilisation GPU (mémoire, température, puissance)
- 💾 Ressources système (RAM, disque, CPU)
- ⚡ Métriques de traitement en temps réel
- 🔄 Progression des tâches

### Métriques Avancées

Les métriques sont exportées vers `metrics/` au format JSON pour analyse ultérieure:
- Durées de traitement par segment
- Utilisation ressources GPU/CPU
- Qualité des traductions
- Erreurs et reprises

## 🚨 Résolution de Problèmes

### Erreurs Courantes

**"CUDA out of memory"**
```bash
# Solutions:
--gpu-memory-limit 0.7  # Réduire limite mémoire
--batch-size 16         # Réduire taille lot
# Fermer autres applications GPU
```

**"Token HF invalide"**
```bash
# Solutions:
python quick_start.py   # Reconfigurer token
# Ou régénérer token sur https://huggingface.co
```

**"Espace disque insuffisant"**
```bash
# Libérer espace:
rm -rf ~/.cache/huggingface/  # Nettoyer cache
# Ou changer répertoire de cache
export HF_HOME=/autre/repertoire/
```

**"Erreur téléchargement YouTube"**
```bash
# Solutions:
pip install -U yt-dlp              # Mettre à jour
python main.py --url "..." --debug # Mode debug
```

### Diagnostic Automatique

```bash
# Diagnostic complet avec fixes automatiques
python main_enhanced.py --validate

# Diagnostic simple
python quick_start.py  # Option 5

# Validation détaillée
python validator.py
```

### Logs et Debug

```bash
# Logs détaillés
python main.py --url "..." --debug --log-level DEBUG

# Fichier de log
tail -f emanet.log

# Mode verbose
python main.py --url "..." --verbose
```

## 📈 Optimisation des Performances

### Pour GPU B200

```bash
python main.py \
  --url "..." \
  --gpu-memory-limit 0.95 \
  --batch-size 128 \
  --precision bf16 \
  --max-workers 12
```

### Pour GPU Standard

```bash
python main.py \
  --url "..." \
  --gpu-memory-limit 0.7 \
  --batch-size 16 \
  --precision fp16 \
  --max-workers 4
```

### Pour CPU Uniquement

```bash
python main.py \
  --url "..." \
  --model voxtral-mini \
  --quality fast \
  --max-workers 2
```

## 🔗 Intégrations

### Scripts Personnalisés

```python
#!/usr/bin/env python3
import subprocess

# Traitement automatisé
def process_video(url, output_lang="fr"):
    cmd = [
        "python", "main.py",
        "--url", url,
        "--target-lang", output_lang,
        "--quality", "best",
        "--monitor"
    ]
    return subprocess.run(cmd)

# Utilisation
process_video("https://youtube.com/watch?v=...", "fr")
```

### CI/CD

```yaml
# GitHub Actions exemple
- name: Generate Subtitles
  run: |
    python main.py \
      --batch-list videos.txt \
      --output-dir ./results/ \
      --quality balanced \
      --continue-on-error
```

## 💡 Conseils et Astuces

### Optimiser la Qualité

1. **Utilisez `--quality best`** pour qualité maximale
2. **Modèle voxtral-small** pour meilleure précision
3. **`--min-quality-score 0.8`** pour filtrer résultats faibles
4. **`--retry-failed-segments`** pour traiter les échecs

### Optimiser la Vitesse

1. **`--quality fast`** pour traitement rapide
2. **Modèle voxtral-mini** plus léger
3. **Augmenter `--batch-size`** si GPU le permet
4. **Augmenter `--max-workers`** pour parallélisme

### Traitement en Lot Efficace

1. **Grouper par langue** dans les fichiers batch
2. **Utiliser `--skip-existing`** pour reprises
3. **`--continue-on-error`** pour robustesse
4. **Surveiller logs** avec `--monitor`

## 📚 Exemples Pratiques

### Conférence Multilingue

```bash
# Générer sous-titres en plusieurs langues
for lang in fr en es de; do
  python main.py \
    --url "conference.mp4" \
    --output "conference_${lang}.srt" \
    --target-lang $lang \
    --quality best
done
```

### Playlist YouTube

```bash
# Créer liste avec yt-dlp
yt-dlp --get-id "playlist_url" > playlist.txt
sed 's/^/https:\/\/youtube.com\/watch?v=/' playlist.txt > batch.txt

# Traiter en lot
python main.py --batch-list batch.txt --output-dir ./playlist_subs/
```

### Surveillance Qualité

```bash
# Avec validation qualité
python main.py \
  --url "..." \
  --quality best \
  --min-quality-score 0.9 \
  --quality-check \
  --retry-failed-segments
```

## 🆘 Support

- **GitHub Issues**: Repository emanet_voxtral
- **Documentation**: README.md et fichiers CLAUDE.md
- **Logs**: emanet.log pour diagnostic
- **Debug**: `--debug --verbose` pour détails complets

## 📄 Licence et Crédits

- **Modèles Voxtral**: Mistral AI
- **Framework**: PyTorch, Transformers, vLLM
- **GPU**: Optimisé pour NVIDIA B200
- **Interface**: Rich CLI

---

*Pour une expérience optimale, utilisez l'assistant interactif avec `python quick_start.py` ou `python main_enhanced.py --wizard`*