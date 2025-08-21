#!/usr/bin/env python3
"""
auth_setup.py - Standalone authentication setup for Voxtral
Simple CLI tool for managing HuggingFace authentication
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.auth_manager import cli_auth_setup

if __name__ == "__main__":
    print("🔐 Voxtral Authentication Setup")
    print("=" * 40)
    try:
        cli_auth_setup()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)