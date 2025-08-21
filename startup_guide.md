# 🚀 Guide de Démarrage EMANET VOXTRAL

## 🎯 Première Utilisation - C'est Parti !

### Étape 1: Lancer l'Interface Guidée

```bash
make
```
ou
```bash
make start
```

**Cela lance automatiquement l'interface utilisateur qui te guidera étape par étape !**

### Étape 2: Choisir ton Niveau

L'interface te proposera automatiquement :

#### 🟢 **Débutant** (Recommandé)
- Interface simple sans dépendances
- Configuration automatique du token HF
- Détection des problèmes avec solutions
- Guidance complète pour chaque action

#### 🟡 **Intermédiaire** 
```bash
make wizard
```
- Assistant complet avec toutes les options
- Configuration avancée des paramètres
- Prévisualisation avant lancement

#### 🔴 **Expert**
```bash
make process URL="https://youtube.com/..." OUTPUT="sous_titres.srt"
```
- Commandes directes avec contrôle total

## 🛠️ Configuration Automatique

### Le système détecte et configure automatiquement :

✅ **Token Hugging Face** - Te guide pour créer et configurer ton token gratuit  
✅ **Espace disque** - Vérifie que tu as assez d'espace (25GB min)  
✅ **GPU** - Détecte ton GPU et optimise les paramètres  
✅ **Dépendances** - Valide que tout est installé correctement  

### Si problème détecté :
- **Solutions automatiques** proposées et appliquées
- **Instructions claires** pour résolution manuelle
- **Diagnostic complet** disponible à tout moment

## 🎮 Exemples d'Utilisation Guidée

### Cas 1: Traiter une Vidéo YouTube
```bash
make start
# → Choisir "1. Traiter une vidéo YouTube"
# → Coller l'URL
# → Choisir la langue
# → Lancer !
```

### Cas 2: Traiter un Fichier Local
```bash
make start
# → Choisir "2. Traiter un fichier local"
# → Sélectionner dans la liste ou saisir le chemin
# → Configurer les options
# → Lancer !
```

### Cas 3: Traitement en Lot
```bash
make start
# → Choisir "3. Traiter plusieurs fichiers"
# → Créer ou sélectionner fichier de lot
# → Configurer les paramètres
# → Lancer !
```

## 🔧 Commandes Rapides

Une fois que tu maîtrises, tu peux utiliser les commandes directes :

```bash
# Configuration système
make setup-interactive

# Diagnostic complet
make validate-interactive

# Guide d'utilisation
make tutorial

# Démonstration
make demo

# Traitement direct YouTube
make process URL="https://youtube.com/watch?v=dQw4w9WgXcQ" OUTPUT="sous_titres.srt"

# Traitement en lot
make batch LIST="videos.txt" DIR="resultats"
```

## 🚨 En Cas de Problème

### 1. Diagnostic Automatique
```bash
make validate-interactive
```
- Analyse complète du système
- Détection des problèmes
- Solutions automatiques proposées

### 2. Guide de Résolution
```bash
make tutorial
```
- Section "Résolution de problèmes"
- Solutions pour erreurs courantes
- Conseils d'optimisation

### 3. Mode Debug
```bash
# Dans l'interface, active le mode debug pour plus d'informations
# Ou utilise directement :
python main.py --url "..." --debug --verbose
```

## 💡 Conseils pour Démarrer

### ✅ **À Faire**
- Commence par `make start` - l'interface te guidera
- Configure ton token HF dès le début
- Teste avec une courte vidéo d'abord
- Active le monitoring pour voir les performances

### ❌ **À Éviter**
- Ne pas ignorer les warnings du diagnostic
- Ne pas traiter de gros fichiers sans tester avant
- Ne pas forcer (`--force`) sans comprendre les erreurs

## 🎯 Workflow Typique

1. **`make`** → Voir l'aide et les options
2. **`make start`** → Interface guidée pour configurer
3. **Tester** avec une petite vidéo
4. **`make tutorial`** → Apprendre les options avancées
5. **Utiliser** normalement pour tes projets

## 🏁 Tu es Prêt !

**Maintenant lance simplement :**
```bash
make start
```

**Et suis les instructions à l'écran. Le système te guidera pour tout !**

---

*Si tu veux revenir à ce guide : `cat startup_guide.md`*