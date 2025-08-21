#!/usr/bin/env python3
"""
simple_auth.py - Simple authentication module without heavy dependencies
Fallback for when cryptography or other modules are not available
"""

import os
import getpass
from pathlib import Path


def setup_hf_token_simple():
    """Simple HF token setup without dependencies."""
    print("\n🔑 Configuration Token Hugging Face")
    print("=" * 50)
    print("1. Allez sur https://huggingface.co")
    print("2. Créez un compte (gratuit)")
    print("3. Settings → Access Tokens → Create new token")
    print("4. Copiez le token")
    print("")
    
    token = getpass.getpass("🔑 Entrez votre token HF (input masqué): ").strip()
    
    if token:
        # Save to .env file
        env_file = Path(".env")
        
        # Read existing content to avoid duplicates
        existing_lines = []
        if env_file.exists():
            with open(env_file, "r") as f:
                existing_lines = [line for line in f.readlines() 
                                if not line.strip().startswith("HF_TOKEN=")]
        
        # Write token to .env
        with open(env_file, "w") as f:
            f.write(f"HF_TOKEN={token}\n")
            f.writelines(existing_lines)
        
        print("✅ Token sauvegardé dans .env")
        print("⚠️  Redémarrez le terminal pour prendre en compte le token")
        
        # Also set for current session
        os.environ['HF_TOKEN'] = token
        print("✅ Token configuré pour cette session")
        return True
    else:
        print("❌ Token non configuré")
        return False


def get_hf_token():
    """Get HF token from environment or .env file."""
    # Check environment first
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        return token
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        return line.split('=', 1)[1].strip('\'"')
        except Exception:
            pass
    
    return None


def check_hf_token():
    """Check if HF token is available."""
    return bool(get_hf_token())


if __name__ == "__main__":
    if check_hf_token():
        print("✅ Token HF trouvé")
    else:
        print("❌ Token HF manquant")
        setup_hf_token_simple()