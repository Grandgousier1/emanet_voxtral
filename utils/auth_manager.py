#!/usr/bin/env python3
"""
utils/auth_manager.py - Authentication management for HuggingFace and other services
Secure token handling with CLI interaction and .env management
"""

import os
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
import json
import base64
import hashlib

# Optional advanced security features
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    keyring = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text

from cli_feedback import CLIFeedback, get_feedback


class TokenManager:
    """Secure token management with multiple storage options."""
    
    def __init__(self, feedback: Optional[CLIFeedback] = None):
        self.feedback = feedback or get_feedback()
        self.console = Console()
        self.env_file = Path('.env')
        self.keyring_service = "voxtral-b200"
    
    def _get_machine_key(self) -> bytes:
        """Generate a machine-specific key for local encryption."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "Cryptography library required for secure token storage. "
                "Install with: pip install cryptography>=42.0.0"
            )
        
        import platform
        machine_info = f"{platform.node()}{platform.machine()}{platform.platform()}"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'voxtral-b200-salt',
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(machine_info.encode()))
    
    def _encrypt_token(self, token: str) -> str:
        """Encrypt token with machine-specific key."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "Cryptography library required for secure token storage. "
                "Install with: pip install cryptography>=42.0.0"
            )
        
        try:
            key = self._get_machine_key()
            f = Fernet(key)
            encrypted = f.encrypt(token.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise RuntimeError(f"Token encryption failed: {e}") from e
    
    def _decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt token with machine-specific key."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "Cryptography library required for secure token storage. "
                "Install with: pip install cryptography>=42.0.0"
            )
        
        try:
            key = self._get_machine_key()
            f = Fernet(key)
            decoded = base64.urlsafe_b64decode(encrypted_token.encode())
            return f.decrypt(decoded).decode()
        except Exception as e:
            raise RuntimeError(f"Token decryption failed: {e}") from e
    
    def get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from various sources in priority order."""
        
        # 1. Environment variable (highest priority)
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if token:
            self.feedback.debug("Using HF token from environment variable")
            return token
        
        # 2. .env file
        if self.env_file.exists():
            token = self._get_token_from_env_file()
            if token:
                self.feedback.debug("Using HF token from .env file")
                return token
        
        # 3. System keyring (secure storage)
        if KEYRING_AVAILABLE:
            try:
                token = keyring.get_password(self.keyring_service, "hf_token")
                if token:
                    # Decrypt if needed
                    token = self._decrypt_token(token)
                    self.feedback.debug("Using HF token from system keyring")
                    return token
            except Exception as e:
                self.feedback.debug(f"Keyring access failed: {e}")
        else:
            self.feedback.debug("System keyring not available")
        
        # 4. Interactive prompt
        return self._prompt_for_token()
    
    def _get_token_from_env_file(self) -> Optional[str]:
        """Extract HF token from .env file."""
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        token = line.split('=', 1)[1].strip('\'"')
                        return self._decrypt_token(token) if len(token) > 50 else token
                    elif line.startswith('HUGGINGFACE_HUB_TOKEN='):
                        token = line.split('=', 1)[1].strip('\'"')
                        return self._decrypt_token(token) if len(token) > 50 else token
        except Exception as e:
            self.feedback.debug(f"Failed to read .env file: {e}")
        return None
    
    def _prompt_for_token(self) -> Optional[str]:
        """Interactive prompt for HF token with security features."""
        
        self.console.print(Panel.fit(
            Text("🤗 HuggingFace Authentication Required", style="bold blue"),
            border_style="blue"
        ))
        
        self.console.print("\n[yellow]To access Mistral Voxtral models, you need a HuggingFace token.[/yellow]")
        self.console.print("[dim]Get your token at: https://huggingface.co/settings/tokens[/dim]\n")
        
        # Check if user wants to continue
        if not Confirm.ask("Do you have a HuggingFace token to enter?", default=True):
            self.console.print("[yellow]⚠️ Proceeding without token - only public models will be available[/yellow]")
            return None
        
        # Secure token input
        token = getpass.getpass("🔑 Enter your HuggingFace token (input hidden): ").strip()
        
        if not token:
            self.console.print("[red]No token entered[/red]")
            return None
        
        # Basic token validation
        if not self._validate_hf_token(token):
            self.console.print("[red]❌ Invalid token format[/red]")
            return None
        
        # Ask how to store the token
        storage_choice = self._prompt_storage_method()
        
        if storage_choice:
            self._store_token(token, storage_choice)
            self.console.print(f"[green]✅ Token stored securely using {storage_choice}[/green]")
        
        return token
    
    def _validate_hf_token(self, token: str) -> bool:
        """Basic validation of HF token format."""
        if not token:
            return False
        
        # HF tokens usually start with 'hf_' and are around 37+ characters
        if token.startswith('hf_') and len(token) >= 20:
            return True
        
        # Legacy tokens might be different
        if len(token) >= 20 and token.replace('_', '').replace('-', '').isalnum():
            return True
        
        return False
    
    def _prompt_storage_method(self) -> Optional[str]:
        """Ask user how they want to store the token."""
        
        self.console.print("\n[bold]How would you like to store your token?[/bold]")
        
        choices = []
        
        # Check keyring availability
        if KEYRING_AVAILABLE:
            try:
                keyring.get_keyring()
                choices.append(("keyring", "System keyring (most secure)"))
            except (ImportError, RuntimeError) as e:
                self.feedback.debug(f"Keyring not available: {e}")
        
        choices.extend([
            ("env_file", ".env file (encrypted, project-specific)"),
            ("session", "This session only (not saved)")
        ])
        
        for i, (key, desc) in enumerate(choices, 1):
            self.console.print(f"  [bold]{i}[/bold]. {desc}")
        
        try:
            choice_num = int(Prompt.ask(
                "Choose storage method", 
                choices=[str(i) for i in range(1, len(choices) + 1)],
                default="1"
            ))
            return choices[choice_num - 1][0]
        except (ValueError, IndexError):
            return None
    
    def _store_token(self, token: str, method: str) -> bool:
        """Store token using specified method."""
        
        try:
            if method == "keyring":
                encrypted_token = self._encrypt_token(token)
                keyring.set_password(self.keyring_service, "hf_token", encrypted_token)
                return True
            
            elif method == "env_file":
                return self._store_in_env_file(token)
            
            elif method == "session":
                # Store in environment for current session
                os.environ['HF_TOKEN'] = token
                return True
                
        except Exception as e:
            self.feedback.error(f"Failed to store token: {e}")
            return False
        
        return False
    
    def _store_in_env_file(self, token: str) -> bool:
        """Store encrypted token in .env file."""
        try:
            encrypted_token = self._encrypt_token(token)
            
            # Read existing .env content
            existing_lines = []
            if self.env_file.exists():
                with open(self.env_file, 'r') as f:
                    existing_lines = [line.rstrip() for line in f if not line.startswith('HF_TOKEN=')]
            
            # Write updated content
            with open(self.env_file, 'w') as f:
                f.write(f'HF_TOKEN="{encrypted_token}"\n')
                for line in existing_lines:
                    if line.strip():
                        f.write(line + '\n')
            
            return True
            
        except Exception as e:
            self.feedback.error(f"Failed to write .env file: {e}")
            return False
    
    def remove_stored_token(self) -> bool:
        """Remove stored HF token from all locations."""
        removed = False
        
        # Remove from keyring
        try:
            keyring.delete_password(self.keyring_service, "hf_token")
            removed = True
            self.console.print("[green]✅ Removed token from system keyring[/green]")
        except (ImportError, RuntimeError, OSError) as e:
            self.feedback.debug(f"Keyring removal failed: {e}")
        
        # Remove from .env file
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    lines = [line for line in f if not line.strip().startswith('HF_TOKEN=')]
                
                with open(self.env_file, 'w') as f:
                    f.writelines(lines)
                
                removed = True
                self.console.print("[green]✅ Removed token from .env file[/green]")
            except Exception as e:
                self.feedback.error(f"Failed to update .env file: {e}")
        
        # Remove from current session
        if 'HF_TOKEN' in os.environ:
            del os.environ['HF_TOKEN']
            removed = True
            self.console.print("[green]✅ Removed token from current session[/green]")
        
        if 'HUGGINGFACE_HUB_TOKEN' in os.environ:
            del os.environ['HUGGINGFACE_HUB_TOKEN']
            removed = True
        
        return removed
    
    def show_token_status(self) -> Dict[str, Any]:
        """Show current token status and sources."""
        status = {
            'environment': bool(os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')),
            'env_file': False,
            'keyring': False,
            'current_token': None
        }
        
        # Check .env file
        if self.env_file.exists():
            token = self._get_token_from_env_file()
            status['env_file'] = bool(token)
        
        # Check keyring
        try:
            token = keyring.get_password(self.keyring_service, "hf_token")
            status['keyring'] = bool(token)
        except (ImportError, RuntimeError, OSError) as e:
            self.feedback.debug(f"Keyring status check failed: {e}")
        
        # Get current active token
        status['current_token'] = bool(self.get_hf_token())
        
        return status


def cli_auth_setup():
    """CLI command for authentication setup."""
    console = Console()
    feedback = get_feedback()
    token_manager = TokenManager(feedback)
    
    console.print(Panel.fit(
        Text("🔐 Voxtral Authentication Manager", style="bold blue"),
        border_style="blue"
    ))
    
    # Show current status
    status = token_manager.show_token_status()
    
    console.print("\n[bold]Current Authentication Status:[/bold]")
    console.print(f"  Environment variable: {'✅' if status['environment'] else '❌'}")
    console.print(f"  .env file: {'✅' if status['env_file'] else '❌'}")
    console.print(f"  System keyring: {'✅' if status['keyring'] else '❌'}")
    console.print(f"  Active token: {'✅' if status['current_token'] else '❌'}")
    
    if status['current_token']:
        console.print("\n[green]✅ Authentication is configured and working[/green]")
        
        if Confirm.ask("Do you want to update or remove your stored token?", default=False):
            action = Prompt.ask(
                "Choose action", 
                choices=["update", "remove"], 
                default="update"
            )
            
            if action == "remove":
                if token_manager.remove_stored_token():
                    console.print("[green]✅ Token removed successfully[/green]")
                else:
                    console.print("[yellow]⚠️ No stored tokens found to remove[/yellow]")
            else:
                # Update token
                token_manager._prompt_for_token()
    else:
        console.print("\n[yellow]⚠️ No authentication configured[/yellow]")
        token_manager.get_hf_token()  # This will trigger the interactive setup


if __name__ == "__main__":
    cli_auth_setup()