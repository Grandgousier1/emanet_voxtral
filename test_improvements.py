#!/usr/bin/env python3
"""
test_improvements.py - Test script for the improved anti-bot and B200 features
"""

import subprocess
from pathlib import Path

def test_import_improvements():
    """Test that all new imports work correctly."""
    print("🧪 Testing import improvements...")
    
    try:
        from utils.antibot_utils import (
            get_random_user_agent, find_browser_cookies, 
            validate_yt_dlp_version, get_antibot_download_args,
            check_antibot_error
        )
        from config import get_adaptive_batch_size
        print("✅ All new imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_antibot_utilities():
    """Test anti-bot protection utilities."""
    print("🛡️ Testing anti-bot utilities...")
    
    try:
        from utils.antibot_utils import (
            get_random_user_agent, validate_yt_dlp_version,
            get_antibot_download_args, check_antibot_error
        )
        
        # Test user agent generation
        ua = get_random_user_agent()
        assert len(ua) > 50, "User agent too short"
        print(f"✅ User agent: {ua[:50]}...")
        
        # Test yt-dlp version check
        is_valid, msg = validate_yt_dlp_version()
        print(f"✅ yt-dlp version check: {msg}")
        
        # Test download args generation
        args = get_antibot_download_args("https://youtube.com/watch?v=test", 0)
        assert '--user-agent' in args, "Missing user agent in args"
        print(f"✅ Download args generated ({len(args)} arguments)")
        
        # Test error detection
        is_antibot, desc = check_antibot_error("sign in to confirm you're not a bot")
        assert is_antibot, "Failed to detect anti-bot error"
        print(f"✅ Anti-bot error detection: {desc}")
        
        return True
    except Exception as e:
        print(f"❌ Anti-bot utilities test failed: {e}")
        return False

def test_config_improvements():
    """Test configuration improvements."""
    print("⚙️ Testing config improvements...")
    
    try:
        from config import detect_hardware
        
        # Test hardware detection
        hw = detect_hardware()
        assert 'gpu_count' in hw, "Hardware detection missing gpu_count"
        print(f"✅ Hardware detected: {hw['gpu_count']} GPUs")
        
        return True
    except Exception as e:
        print(f"❌ Config improvements test failed: {e}")
        return False

def test_cli_interface():
    """Test improved CLI interface."""
    print("🖥️ Testing CLI interface...")
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, 'main.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        help_output = result.stdout
        assert '--cookies' in help_output, "Missing --cookies option in help"
        assert 'anti-bot protection' in help_output, "Missing anti-bot description"
        print("✅ CLI help includes new --cookies option")
        
        return True
    except subprocess.TimeoutExpired:
        print("❌ CLI help command timed out")
        return False
    except Exception as e:
        print(f"❌ CLI interface test failed: {e}")
        return False

def test_requirements_updates():
    """Test that requirements.txt has been updated correctly."""
    print("📦 Testing requirements updates...")
    
    try:
        req_file = Path('requirements.txt')
        if not req_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        content = req_file.read_text()
        
        # Check for updated versions
        checks = [
            ('yt-dlp>=2025.08.11', 'Updated yt-dlp version'),
            ('transformers>=4.53.0,<5.0', 'Fixed transformers version'),
            ('fake-useragent>=1.4.0', 'Anti-bot user agents'),
            ('resampy>=0.4.2', 'Audio processing robustness'),
        ]
        
        for check, description in checks:
            if check in content:
                print(f"✅ {description}: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

