#!/usr/bin/env python3
"""
simple_token_manager.py - Token manager simple et robuste
Remplace auth_manager.py quand il y a des problèmes
"""

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cli_feedback import CLIFeedback


def get_hf_token() -> Optional[str]:
    """Récupère le token HuggingFace de manière simple et fiable."""
    
    # 1. Variable d'environnement (priorité haute)
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        return token.strip()
    
    # 2. Fichier .env
    env_file = Path('.env')
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        token = line.split('=', 1)[1].strip('\'"')
                        if token:
                            return token
                    elif line.startswith('HUGGINGFACE_HUB_TOKEN='):
                        token = line.split('=', 1)[1].strip('\'"')
                        if token:
                            return token
        except Exception:
            pass
    
    # 3. Fichier ~/.cache/huggingface/token (HuggingFace CLI)
    try:
        hf_token_file = Path.home() / '.cache' / 'huggingface' / 'token'
        if hf_token_file.exists():
            token = hf_token_file.read_text().strip()
            if token:
                return token
    except Exception:
        pass
    
    return None


def check_hf_token() -> bool:
    """Vérifie si un token HuggingFace est disponible."""
    return bool(get_hf_token())


def save_hf_token(token: str) -> bool:
    """Sauvegarde le token HuggingFace dans .env et la session."""
    if not token:
        return False
    
    try:
        # Sauvegarder dans .env
        env_file = Path('.env')
        
        # Lire le contenu existant
        existing_lines = []
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                existing_lines = [line for line in f 
                                if not line.strip().startswith(('HF_TOKEN=', 'HUGGINGFACE_HUB_TOKEN='))]
        
        # Écrire le nouveau token
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(f'HF_TOKEN={token}\n')
            f.writelines(existing_lines)
        
        # Configurer pour la session actuelle
        os.environ['HF_TOKEN'] = token
        
        return True
    except Exception:
        return False


class SimpleTokenManager:
    """Version simple du TokenManager sans dépendances compliquées."""
    
    def __init__(self, feedback: Optional['CLIFeedback'] = None):
        self.feedback = feedback
    
    def get_hf_token(self) -> Optional[str]:
        """Récupère le token HuggingFace."""
        return get_hf_token()
    
    def check_token(self) -> bool:
        """Vérifie si le token est disponible."""
        return check_hf_token()


# Fonction de secours pour remplacer TokenManager en cas de problème
def get_simple_token_manager(feedback=None):
    """Retourne un TokenManager simple."""
    return SimpleTokenManager(feedback)


if __name__ == "__main__":
    token = get_hf_token()
    if token:
        print(f"✅ Token trouvé: {token[:10]}...")
    else:
        print("❌ Token non trouvé")