"""
utils/security_utils.py - Security related utilities, including token validation.
"""
import logging
import urllib.parse
from pathlib import Path
import os

logger = logging.getLogger(__name__)

def validate_hf_token() -> bool:
    """Performs a pre-flight check to validate the Hugging Face token."""
    try:
        from huggingface_hub import HfApi, HfFolder
        from huggingface_hub.errors import HfHubError

        token = HfFolder.get_token()
        if not token:
            logger.warning("Hugging Face token not found. Only public models will be accessible.")
            return True

        logger.info("Validating Hugging Face token...")
        api = HfApi()
        user = api.whoami()
        logger.info(f"Hugging Face token is valid. Authenticated as: {user.get('name')}")
        return True
    except ImportError:
        logger.warning("huggingface_hub is not installed. Cannot validate token.")
        return True # Non-blocking if library is not present
    except HfHubError as e:
        logger.critical(f"Hugging Face token is invalid or expired. Please login again. Error: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Hugging Face token validation: {e}")
        return False

def validate_url_security(url: str) -> bool:
    """Enhanced URL security validation with comprehensive protections."""
    try:
        if len(url) > 2048:
            return False
        
        parsed = urllib.parse.urlparse(url)
        
        if parsed.scheme not in ['https', 'http']:
            return False
        
        dangerous_schemes = ['file', 'ftp', 'data', 'javascript', 'vbscript']
        if any(scheme in url.lower() for scheme in dangerous_schemes):
            return False
        
        domain = parsed.netloc.lower()
        if 'localhost' in domain:
            if os.getenv('ENVIRONMENT', 'production') == 'production':
                return False
        
        allowed_domains = [
            'youtube.com', 'www.youtube.com', 'm.youtube.com',
            'youtu.be', 'music.youtube.com',
            'soundcloud.com', 'www.soundcloud.com',
            'vimeo.com', 'www.vimeo.com',
            'dailymotion.com', 'www.dailymotion.com',
        ]
        
        if not any(domain == allowed or domain.endswith('.' + allowed) for allowed in allowed_domains):
            if 'localhost' not in domain and '127.0.0.1' not in domain:
                return False
        
        dangerous_patterns = ['..', '//', '\\', '%2e%2e', '%2f%2f', '....']
        if any(pattern in parsed.path.lower() for pattern in dangerous_patterns):
            return False
        
        if parsed.query:
            dangerous_query_patterns = ['javascript:', 'data:', 'file:', '<script', 'eval(']
            if any(pattern in parsed.query.lower() for pattern in dangerous_query_patterns):
                return False
        
        if parsed.port:
            blocked_ports = list(range(1, 1024)) + [1080, 3128, 8080]
            allowed_ports = [80, 443, 8443]
            if parsed.port in blocked_ports and parsed.port not in allowed_ports:
                return False
        
        import ipaddress
        try:
            ip = ipaddress.ip_address(domain)
            if ip.is_private and os.getenv('ENVIRONMENT', 'production') == 'production':
                return False
        except ValueError:
            pass
        
        return True
        
    except Exception as e:
        return False

def validate_local_path_security(path_str: str) -> bool:
    """Enhanced local file path security validation."""
    try:
        path = Path(path_str)
        
        dangerous_patterns = [
            '..', '..\\', '../', '..\\\\',
            '%2e%2e', '%2e%2e%2f', '%2e%2e%5c',
            '....', '..../', '....\\\\',
            '.\\.. ',
            './/..',
            # Additional security patterns
            '\\\\?\\', '\\\\UNC\\', '\\\\localhost\\',
            '%5c%5c', '%2f%2f', '\\.\\',
            'aux', 'con', 'prn', 'nul',  # Windows reserved names
            'com1', 'com2', 'lpt1', 'lpt2'  # Windows device names
        ]
        
        # Additional length check to prevent buffer overflow attempts
        if len(path_str) > 4096:  # Reasonable maximum path length
            return False
        
        # Check for null bytes (path injection)
        if '\x00' in path_str:
            return False
        
        path_str_lower = path_str.lower()
        for pattern in dangerous_patterns:
            if pattern in path_str_lower:
                return False
        
        sensitive_dirs = [
            '/etc', '/proc', '/sys', '/dev', '/root',
            '/boot', '/usr/bin', '/bin', '/sbin',
            'c:\\windows', 'c:\\program files', 'c:\\users\\administrator'
        ]
        
        # Safely resolve path with additional checks
        try:
            resolved_path = path.resolve(strict=False)  # Don't require path to exist
            resolved_str = str(resolved_path).lower()
        except (OSError, ValueError, RuntimeError):
            # Path resolution failed - likely malicious
            return False
        
        for sensitive in sensitive_dirs:
            if resolved_str.startswith(sensitive.lower()):
                return False
        
        # Additional check: ensure resolved path doesn't escape allowed areas
        try:
            # Check if path tries to escape via symlinks or other means
            current_dir = Path.cwd().resolve()
            if not path.is_absolute():
                # For relative paths, ensure they stay within current directory tree
                if current_dir not in resolved_path.parents and resolved_path != current_dir:
                    # Check if it's a child of current directory
                    try:
                        resolved_path.relative_to(current_dir)
                    except ValueError:
                        return False
        except (OSError, ValueError):
            return False
        
        if path.is_absolute():
            safe_absolute_prefixes = [
                '/tmp', '/var/tmp', '/home', '/workspace',
                str(Path.cwd()), str(Path.home()),
            ]
            
            import os
            safe_absolute_prefixes.extend([
                os.getcwd(),
                os.path.expanduser('~'),
                '/app', 
                '/data',
            ])
            
            if not any(resolved_str.startswith(safe.lower()) for safe in safe_absolute_prefixes):
                return False
        
        if path.name and any(char in path.name for char in ['<', '>', ':', '"', '|', '?', '*']):
            return False
        
        if path.name and '.' in path.name and 'ssh' in path.name.lower():
            return False
        
        return True
            
    except Exception as e:
        return False


class SecureSubprocess:
    """Secure subprocess wrapper for safe command execution."""
    
    def __init__(self, feedback):
        self.feedback = feedback
    
    def run_secure_command(self, command, timeout=30, **kwargs):
        """Execute a command securely with validation and timeout."""
        import subprocess
        import shlex
        
        try:
            # Basic command validation
            if not command:
                self.feedback.error("Empty command not allowed")
                return None
            
            # Convert to list if string
            if isinstance(command, str):
                command_list = shlex.split(command)
            else:
                command_list = command
            
            # Basic security check - no shell injection patterns
            dangerous_patterns = ['&', '|', ';', '`', '$', '>', '<', '&&', '||']
            command_str = ' '.join(command_list)
            if any(pattern in command_str for pattern in dangerous_patterns):
                self.feedback.error("Potentially dangerous command patterns detected")
                return None
            
            # Execute with timeout
            result = subprocess.run(
                command_list,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False,
                **kwargs
            )
            
            return result
            
        except subprocess.TimeoutExpired:
            self.feedback.error(f"Command timed out after {timeout}s")
            return None
        except Exception as e:
            self.feedback.error(f"Command execution failed: {e}")
            return None