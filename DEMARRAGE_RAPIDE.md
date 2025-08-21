# 🚀 DÉMARRAGE RAPIDE - EMANET VOXTRAL

## 1️⃣ Configuration initiale (OBLIGATOIRE)

```bash
# Installer les dépendances
make install-dev

# Configuration token + interface complète
make start
```

**C'est tout !** Cette séquence :
- ✅ Installe toutes les dépendances nécessaires
- 🔑 Configure votre token HuggingFace **une seule fois**
- 🎯 Lance l'interface interactive complète avec menu

## 2️⃣ Utilisation quotidienne

Après la configuration initiale, juste :

```bash
make start
```

Ou directement :

```bash
python main.py --url "https://youtube.com/watch?v=..."
```

## 🔧 Si vous avez des problèmes

```bash
# Diagnostic complet
make validate

# Réinstaller proprement
make clean
make install-dev
make start
```

## 📝 Séquence complète pour nouveau projet

```bash
git clone https://github.com/Grandgousier1/emanet_voxtral.git
cd emanet_voxtral
make install-dev    # Installe tout
make start          # Configure + interface
```

**Promesse** : Après `make install-dev` + `make start`, tout fonctionne parfaitement !