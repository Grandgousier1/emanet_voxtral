# 🚀 EMANET VOXTRAL - Utilisation Ultra-Simple

## Configuration en 1 étape

```bash
make start
```

**C'est tout !** Le système :

1. ✅ Détecte automatiquement si vous avez un token
2. 🔑 Si pas de token → vous demande de le coller **UNE SEULE FOIS**
3. 💾 Sauvegarde automatiquement pour les prochaines fois
4. 🎯 Plus jamais besoin de reconfigurer

## Utilisation après configuration

```bash
# Traiter une vidéo YouTube
python main.py --url "https://youtube.com/watch?v=..."

# Voir toutes les options
python main.py --help
```

## Où récupérer votre token HuggingFace ?

1. Allez sur : https://huggingface.co/settings/tokens
2. Créez un nouveau token (lecture seule suffit)
3. Copiez-le
4. Lancez `make start` et collez-le quand demandé

## Résolution de problèmes

Si vous avez des problèmes :

```bash
# Test rapide de votre config
python test_token.py

# Configuration manuelle directe
export HF_TOKEN="votre_token_ici"

# Version de secours ultra-simple
python quick_start_ultra.py
```

## Garanties

- ✅ **Une seule configuration** : Le token est sauvé automatiquement
- ✅ **Pas de redémarrage terminal** : Fonctionne immédiatement
- ✅ **Aucune dépendance compliquée** : Système de secours intégré
- ✅ **Persistant** : Marche à chaque fois après la première config

**Plus aucune manipulation particulière nécessaire après la première configuration !**