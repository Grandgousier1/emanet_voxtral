#!/usr/bin/env python3
"""
antibot_utils.py - Anti-bot protection utilities for YouTube downloads
"""

import random
import time
import glob
from pathlib import Path
from typing import Optional, List

def get_random_user_agent() -> str:
    """Get a random realistic User-Agent string."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    ]
    return random.choice(user_agents)

def find_browser_cookies() -> Optional[Path]:
    """Try to find browser cookies automatically.
    
    WARNING: This function accesses sensitive user data (browser cookies).
    Ensure that this is done securely and with user consent.
    """
    # Common browser cookie locations
    cookie_paths = [
        # Chrome/Chromium cookies
        Path.home() / '.config/google-chrome/Default/Cookies',
        Path.home() / '.config/chromium/Default/Cookies',
        Path.home() / 'Library/Application Support/Google/Chrome/Default/Cookies',  # macOS
        Path.home() / 'AppData/Local/Google/Chrome/User Data/Default/Cookies',      # Windows
        
        # Firefox cookies  
        Path.home() / '.mozilla/firefox/*/cookies.sqlite',
        Path.home() / 'Library/Application Support/Firefox/Profiles/*/cookies.sqlite',  # macOS
        Path.home() / 'AppData/Roaming/Mozilla/Firefox/Profiles/*/cookies.sqlite',      # Windows
        
        # Manual cookie files (exported)
        Path('./cookies.txt'),
        Path('./youtube_cookies.txt'),
        Path.home() / 'Downloads/cookies.txt',
    ]
    
    for cookie_path in cookie_paths:
        if '*' in str(cookie_path):
            # Handle glob patterns for Firefox
            matches = glob.glob(str(cookie_path))
            if matches:
                return Path(matches[0])
        elif cookie_path.exists():
            return cookie_path
    
    return None

def get_smart_sleep_interval(attempt: int = 0) -> tuple[int, int]:
    """Get smart sleep intervals based on attempt number."""
    base_sleep = [1, 2, 3, 5, 8][min(attempt, 4)]  # Fibonacci-like progression
    max_sleep = base_sleep + random.randint(1, 3)  # Add randomness
    return base_sleep, max_sleep

def create_cookie_instructions() -> str:
    """Create instructions for manual cookie export."""
    return """
🍪 TO EXPORT BROWSER COOKIES FOR ANTI-BOT PROTECTION:

📌 CHROME/EDGE:
1. Install extension: "Get cookies.txt LOCALLY" 
2. Go to youtube.com and login
3. Click extension icon → Export → Save as cookies.txt

📌 FIREFOX:
1. Install addon: "cookies.txt"
2. Go to youtube.com and login  
3. Click addon icon → Export cookies.txt

📌 MANUAL METHOD:
1. Open browser Developer Tools (F12)
2. Go to Application/Storage → Cookies → https://youtube.com
3. Copy all cookies to cookies.txt format

💡 PLACE COOKIES FILE IN:
- ./cookies.txt (current directory)
- ~/Downloads/cookies.txt
- Or specify with --cookies path/to/cookies.txt

⚠️ IMPORTANT: Keep cookies file secure and don't share it!
"""

def validate_yt_dlp_version() -> tuple[bool, str]:
    """Check if yt-dlp version is recent enough for anti-bot protection."""
    try:
        from .security_utils import SecureSubprocess
        secure_proc = SecureSubprocess()
        result = secure_proc.run_secure(['yt-dlp', '--version'])
        version = result.stdout.strip()
        
        # Extract version number (format: YYYY.MM.DD)
        if len(version) >= 10:
            version_parts = version.split('.')
            if len(version_parts) >= 3:
                year = int(version_parts[0])
                month = int(version_parts[1])
                
                # Check if version is recent enough (6 months from current date)
                from datetime import datetime, timedelta
                current_date = datetime.now()
                min_required_date = current_date - timedelta(days=180)  # 6 months
                
                try:
                    version_date = datetime(year, month, 1)
                    if version_date >= min_required_date:
                        return True, f"✅ yt-dlp version {version} is up to date"
                    else:
                        return False, f"❌ yt-dlp version {version} is outdated. Need version from last 6 months"
                except ValueError:
                    return False, f"❌ Invalid yt-dlp version format: {version}"
    except Exception as e:
        return False, f"❌ Cannot check yt-dlp version: {e}"
    
    return False, "❌ Invalid yt-dlp version format"

def get_antibot_download_args(url: str, attempt: int = 0) -> List[str]:
    """Get optimized yt-dlp arguments for anti-bot protection."""
    base_sleep, max_sleep = get_smart_sleep_interval(attempt)
    user_agent = get_random_user_agent()
    
    args = [
        '--user-agent', user_agent,
        '--sleep-interval', str(base_sleep),
        '--max-sleep-interval', str(max_sleep),
        '--retries', str(3 + attempt),  # Increase retries with attempts
        '--fragment-retries', str(5 + attempt),
        '--skip-unavailable-fragments',
        '--no-check-certificate',  # WARNING: Disables SSL certificate validation. Use with caution and only if absolutely necessary.
    ]
    
    # Add YouTube-specific optimizations
    if 'youtube.com' in url or 'youtu.be' in url:
        args.extend([
            '--extractor-args', 'youtube:player_client=web',
            '--add-header', 'Accept-Language:en-US,en;q=0.9',
            '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        ])
        
        # For higher attempt numbers, use more stealth
        if attempt >= 2:
            args.extend([
                '--extractor-args', 'youtube:player_client=web,tv',
                '--add-header', 'DNT:1',
                '--add-header', 'Upgrade-Insecure-Requests:1',
            ])
    
    return args

def check_antibot_error(error_output: str) -> tuple[bool, str]:
    """Check if error is anti-bot related and return appropriate solution."""
    error_lower = error_output.lower()
    
    antibot_indicators = [
        ('sign in to confirm', "YouTube requires sign-in verification"),
        ('not a bot', "Anti-bot verification triggered"),
        ('captcha', "CAPTCHA protection active"),
        ('rate limit', "Rate limiting detected"),
        ('too many requests', "Request limit exceeded"),
        ('temporarily unavailable', "Temporary blocking active"),
    ]
    
    for indicator, description in antibot_indicators:
        if indicator in error_lower:
            return True, description
    
    return False, "Unknown error"

if __name__ == '__main__':
    # Test utilities
    print("🔍 Testing anti-bot utilities...")
    
    print(f"Random User-Agent: {get_random_user_agent()}")
    
    cookies = find_browser_cookies()
    if cookies:
        print(f"Found cookies: {cookies}")
    else:
        print("No cookies found")
    
    is_valid, message = validate_yt_dlp_version()
    print(message)
    
    if not is_valid:
        print(create_cookie_instructions())