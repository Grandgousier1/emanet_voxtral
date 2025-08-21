#!/usr/bin/env python3
"""
cli_help.py - Enhanced CLI help and examples for Voxtral
Interactive guide for users to understand all CLI options
"""

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown

console = Console()

def show_quick_start():
    """Show quick start guide."""
    
    console.print(Panel.fit(
        Text("🚀 Quick Start Guide", style="bold green"),
        border_style="green"
    ))
    
    examples = [
        ("Basic Usage", "python main.py --url 'https://youtube.com/watch?v=abc123' --output subtitle.srt"),
        ("Batch Processing", "python main.py --batch-list videos.txt --output-dir ./subtitles/"),
        ("Setup Authentication", "python main.py --setup-auth"),
        ("High Quality", "python main.py --url 'video.mp4' --quality best --output result.srt"),
        ("Debug Mode", "python main.py --url 'video.mp4' --debug --output result.srt")
    ]
    
    for title, command in examples:
        console.print(f"\n[bold blue]{title}:[/bold blue]")
        console.print(f"[dim]$ {command}[/dim]")

def show_authentication_help():
    """Show authentication setup help."""
    
    console.print(Panel.fit(
        Text("🔐 Authentication Setup", style="bold blue"),
        border_style="blue"
    ))
    
    markdown_text = """
## HuggingFace Token Setup

Voxtral models require a HuggingFace token for access. Here's how to set it up:

### Method 1: Interactive Setup (Recommended)
```bash
python main.py --setup-auth
```
or
```bash
python auth_setup.py
```

### Method 2: Command Line
```bash
python main.py --hf-token "hf_your_token_here" --url "video.mp4"
```

### Method 3: Environment Variable
```bash
export HF_TOKEN="hf_your_token_here"
python main.py --url "video.mp4"
```

### Getting Your Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Copy the token (starts with `hf_`)

### Security Features
- Tokens are encrypted when stored locally
- Multiple storage options: system keyring, .env file, or session-only
- Automatic token validation
"""
    
    console.print(Markdown(markdown_text))

def show_advanced_options():
    """Show advanced CLI options."""
    
    table = Table(title="Advanced Options", show_header=True)
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Example", style="dim")
    
    advanced_options = [
        ("--quality", "Quality/speed tradeoff", "fast|balanced|best"),
        ("--format", "Output format", "srt|vtt|json"),
        ("--max-workers", "Parallel workers", "4"),
        ("--gpu-memory-limit", "GPU memory limit", "0.8"),
        ("--timeout", "Processing timeout", "3600"),
        ("--no-cache", "Disable caching", "flag"),
        ("--log-level", "Logging verbosity", "DEBUG|INFO|WARNING"),
        ("--quiet", "Minimal output", "flag"),
        ("--force", "Skip validation", "flag"),
        ("--dry-run", "Test without processing", "flag"),
    ]
    
    for option, desc, example in advanced_options:
        table.add_row(option, desc, example)
    
    console.print("\n")
    console.print(table)

def show_troubleshooting():
    """Show troubleshooting guide."""
    
    console.print(Panel.fit(
        Text("🔧 Troubleshooting", style="bold yellow"),
        border_style="yellow"
    ))
    
    troubleshooting_md = """
## Common Issues and Solutions

### 1. Authentication Errors
```bash
# Reset and reconfigure authentication
python main.py --setup-auth
```

### 2. Out of Memory Errors
```bash
# Reduce GPU memory usage
python main.py --gpu-memory-limit 0.7 --url "video.mp4"
```

### 3. Model Loading Fails
```bash
# Use smaller model
python main.py --use-voxtral-mini --url "video.mp4"
```

### 4. Network/Download Issues
```bash
# Use local file instead
python main.py --url "/path/to/audio.mp4" --output result.srt
```

### 5. Validation Errors
```bash
# Run diagnostics
python main.py --validate-only

# Force skip validation (risky)
python main.py --force --url "video.mp4"
```

### 6. Debug Information
```bash
# Enable detailed logging
python main.py --debug --log-level DEBUG --url "video.mp4"
```
"""
    
    console.print(Markdown(troubleshooting_md))

def show_b200_optimization():
    """Show B200-specific optimization help."""
    
    console.print(Panel.fit(
        Text("🚀 NVIDIA B200 Optimizations", style="bold magenta"),
        border_style="magenta"
    ))
    
    b200_md = """
## B200 GPU Optimization Features

### Automatic Detection
The system automatically detects B200 hardware and applies optimizations:

- **180GB VRAM utilization**: Up to 95% memory usage
- **bfloat16 precision**: Optimized for Tensor Cores
- **Async processing**: 28 vCPU parallel utilization
- **Batch optimization**: Large batch sizes for throughput

### Manual Configuration
```bash
# Adjust GPU memory usage
python main.py --gpu-memory-limit 0.95 --url "video.mp4"

# Set parallel workers
python main.py --max-workers 14 --url "video.mp4"

# Best quality for B200
python main.py --quality best --url "video.mp4"
```

### Monitoring
```bash
# Check B200 status
make monitor-b200

# Validate B200 setup
make validate-all-b200
```
"""
    
    console.print(Markdown(b200_md))

def show_full_help():
    """Show comprehensive help."""
    
    console.print(Panel.fit(
        Text("📚 Voxtral CLI Complete Guide", style="bold white"),
        border_style="white"
    ))
    
    sections = [
        ("1. Quick Start", show_quick_start),
        ("2. Authentication", show_authentication_help),
        ("3. Advanced Options", show_advanced_options),
        ("4. B200 Optimization", show_b200_optimization),
        ("5. Troubleshooting", show_troubleshooting),
    ]
    
    for title, func in sections:
        console.print(f"\n{'='*50}")
        func()
        console.input("\n[dim]Press Enter to continue...[/dim]")
        console.clear()
    
    console.print("\n[bold green]✅ Help guide complete![/bold green]")
    console.print("[dim]Use 'python main.py --help' for quick reference.[/dim]")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        section = sys.argv[1].lower()
        if section == "auth":
            show_authentication_help()
        elif section == "advanced":
            show_advanced_options()
        elif section == "b200":
            show_b200_optimization()
        elif section == "troubleshooting":
            show_troubleshooting()
        elif section == "quick":
            show_quick_start()
        else:
            console.print(f"[red]Unknown section: {section}[/red]")
            console.print("[dim]Available sections: auth, advanced, b200, troubleshooting, quick[/dim]")
    else:
        show_full_help()