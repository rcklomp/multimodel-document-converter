#!/usr/bin/env python3
"""
Interactive Ingestion Query Tool
=================================
A user-friendly semantic search interface for querying ingestion.jsonl files.

Features:
- File browser to select ingestion.jsonl file
- Interactive query loop (keeps running until quit)
- Rich formatted output with colors and tables
- Shows top N results with scores, page numbers, and content previews
- Supports multiple search modes

Usage:
    python tests/test_ingestion_query.py

    Or from the project root:
    conda run -p ./env python tests/test_ingestion_query.py

Author: Claude (Architect)
Date: 2025-01-03
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Check for required dependencies
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Install with: pip install sentence-transformers scikit-learn numpy")
    sys.exit(1)


# Define stub classes for fallback
class Console:
    def print(self, *args, **kwargs):
        print(*args, flush=True)

    def clear(self):
        import os
        import sys

        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()


class Table:
    def __init__(self, **kwargs):
        self.title = kwargs.get("title", "")
        self.columns = []

    def add_column(self, *args, **kwargs):
        self.columns.append(args[0] if args else "")

    def add_row(self, *args, **kwargs):
        pass


class Panel:
    def __init__(self, *args, **kwargs):
        pass


class Markdown:
    def __init__(self, *args, **kwargs):
        pass


class Prompt:
    @staticmethod
    def ask(prompt_text="", default=""):
        try:
            user_input = input(prompt_text + " " if prompt_text else "")
            return user_input if user_input else default
        except (EOFError, KeyboardInterrupt):
            return None


class Confirm:
    @staticmethod
    def ask(prompt_text="", default=False):
        try:
            response = input(prompt_text + " (y/n): " if prompt_text else "(y/n): ")
            return response.lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return default


def rprint(*args, **kwargs):
    print(*args)


# Try to import and override with rich components
try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich.panel import Panel as _RichPanel
    from rich.markdown import Markdown as _RichMarkdown
    from rich.prompt import Prompt as _RichPrompt
    from rich.prompt import Confirm as _RichConfirm
    from rich import print as rprint

    Console = _RichConsole  # type: ignore
    Table = _RichTable  # type: ignore
    Panel = _RichPanel  # type: ignore
    Markdown = _RichMarkdown  # type: ignore
    Prompt = _RichPrompt  # type: ignore
    Confirm = _RichConfirm  # type: ignore
except ImportError:
    print("⚠️ Rich library not found. Using basic fallback...")
    print("   For best experience: pip install rich")


# Initialize console
console = Console()


def select_file_with_browser() -> Optional[Path]:
    """
    Open a file browser dialog to select an ingestion.jsonl file.

    Returns:
        Path to selected file, or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes("-topmost", True)  # Bring dialog to front

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select ingestion.jsonl file",
            filetypes=[
                ("JSONL files", "*.jsonl"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
            initialdir=Path.cwd() / "output",
        )

        root.destroy()

        if file_path:
            return Path(file_path)
        return None

    except Exception as e:
        console.print(f"[yellow]⚠️ Could not open file browser: {e}[/yellow]")
        console.print("[dim]Falling back to manual path entry...[/dim]")
        return None


def load_ingestion_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load chunks from an ingestion.jsonl file.

    Args:
        file_path: Path to the ingestion.jsonl file

    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Warning: Skipped invalid JSON on line {line_num}[/yellow]")
    return chunks


def display_welcome_banner(file_path: Path, num_chunks: int):
    """Display a welcome banner with file info."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║            🔍  MMRAG V2 - Interactive Semantic Search  🔍            ║
╠══════════════════════════════════════════════════════════════════════╣
║  File: {file_path.name:<61} ║
║  Chunks: {num_chunks:<59} ║
║  Model: all-MiniLM-L6-v2                                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


def display_help():
    """Display help information."""
    help_text = """
## Commands

| Command | Description |
|---------|-------------|
| `exit`, `quit`, `q` | Exit the application |
| `help`, `?` | Show this help message |
| `stats` | Show statistics about the loaded data |
| `top N` | Set number of results to show (default: 3) |
| `page N` | Show all chunks from page N |
| `list` | List first 10 chunks |
| `clear` | Clear the screen |

## Query Examples

- `Mauser logo` - Find content about Mauser branding
- `maintenance tools` - Find tool-related content
- `assembly instructions` - Find assembly/disassembly info

**Tip:** Just type your query and press Enter to search!
"""
    console.print(Markdown(help_text))


def display_stats(chunks: List[Dict[str, Any]]):
    """Display statistics about the loaded chunks."""
    # Count by modality
    modalities = {}
    pages = set()
    total_chars = 0

    for chunk in chunks:
        modality = chunk.get("modality", "unknown")
        modalities[modality] = modalities.get(modality, 0) + 1

        page = chunk.get("metadata", {}).get("page_number")
        if page:
            pages.add(page)

        content = chunk.get("content", "")
        total_chars += len(content)

    # Create stats table
    table = Table(title="📊 Ingestion Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(len(chunks)))
    table.add_row("Total Pages", str(len(pages)))
    table.add_row("Total Characters", f"{total_chars:,}")
    table.add_row("Avg Chars/Chunk", f"{total_chars // len(chunks):,}" if chunks else "0")

    console.print(table)
    console.print()

    # Modality breakdown
    mod_table = Table(title="📦 Chunks by Modality", show_header=True)
    mod_table.add_column("Modality", style="cyan")
    mod_table.add_column("Count", style="green")
    mod_table.add_column("Percentage", style="yellow")

    for modality, count in sorted(modalities.items()):
        pct = (count / len(chunks)) * 100 if chunks else 0
        mod_table.add_row(modality, str(count), f"{pct:.1f}%")

    console.print(mod_table)


def display_results(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_n: int = 3,
):
    """
    Search and display results for a query.

    Args:
        query: Search query
        chunks: List of chunk dictionaries
        embeddings: Pre-computed embeddings
        model: SentenceTransformer model
        top_n: Number of results to display
    """
    # Compute query embedding
    query_embedding = model.encode([query], show_progress_bar=False)

    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top N indices
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # Display results
    console.print()
    console.print(Panel(f"[bold]Query:[/bold] {query}", title="🔍 Search", border_style="blue"))
    console.print()

    # Create results table
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="green", width=8)
    table.add_column("Page", style="cyan", width=6)
    table.add_column("Type", style="yellow", width=8)
    table.add_column("Content Preview", style="white")

    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        chunk = chunks[idx]

        # Extract metadata
        metadata = chunk.get("metadata", {})
        page_num = metadata.get("page_number") or metadata.get("page") or "?"
        modality = chunk.get("modality", "?")
        content = chunk.get("content", "")[:150]

        # Add ellipsis if truncated
        if len(chunk.get("content", "")) > 150:
            content += "..."

        # Color score based on relevance
        if score >= 0.7:
            score_str = f"[bold green]{score:.4f}[/bold green]"
        elif score >= 0.5:
            score_str = f"[yellow]{score:.4f}[/yellow]"
        else:
            score_str = f"[dim]{score:.4f}[/dim]"

        table.add_row(
            str(rank),
            score_str,
            str(page_num),
            modality[:7],
            content,
        )

    console.print(table)
    console.print()


def show_page_chunks(chunks: List[Dict[str, Any]], page_num: int):
    """Show all chunks from a specific page."""
    page_chunks = [c for c in chunks if c.get("metadata", {}).get("page_number") == page_num]

    if not page_chunks:
        console.print(f"[yellow]No chunks found for page {page_num}[/yellow]")
        return

    console.print(f"\n[bold cyan]📄 Page {page_num} - {len(page_chunks)} chunks[/bold cyan]\n")

    for i, chunk in enumerate(page_chunks, 1):
        modality = chunk.get("modality", "unknown")
        content = chunk.get("content", "")[:200]
        asset_ref = chunk.get("asset_ref", {})

        console.print(f"[bold]Chunk {i}[/bold] ([yellow]{modality}[/yellow])")
        if asset_ref:
            console.print(f"  [dim]Asset: {asset_ref.get('file_path', 'N/A')}[/dim]")
        console.print(f"  {content}{'...' if len(chunk.get('content', '')) > 200 else ''}")
        console.print()


def list_chunks(chunks: List[Dict[str, Any]], limit: int = 10):
    """List first N chunks."""
    table = Table(title=f"📋 First {min(limit, len(chunks))} Chunks", show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Page", style="cyan", width=6)
    table.add_column("Type", style="yellow", width=8)
    table.add_column("Content Preview", style="white")

    for i, chunk in enumerate(chunks[:limit], 1):
        metadata = chunk.get("metadata", {})
        page = metadata.get("page_number") or "?"
        modality = chunk.get("modality", "?")
        content = chunk.get("content", "")[:80]
        if len(chunk.get("content", "")) > 80:
            content += "..."

        table.add_row(str(i), str(page), modality[:7], content)

    console.print(table)


def main():
    """Main entry point for the interactive query tool."""
    console.print("\n[bold blue]🚀 MMRAG V2 - Interactive Ingestion Query Tool[/bold blue]\n")

    # Step 1: Select file
    console.print("[dim]Opening file browser...[/dim]")
    file_path = select_file_with_browser()

    if not file_path:
        # Fallback to manual entry
        console.print("\n[yellow]No file selected. Enter path manually:[/yellow]")
        path_str = Prompt.ask(
            "Path to ingestion.jsonl", default="./output/Firearms_test/ingestion.jsonl"
        )
        if path_str is None:
            console.print("[yellow]No path provided. Exiting.[/yellow]")
            sys.exit(0)
        file_path = Path(path_str)

    # Validate file exists
    if not file_path.exists():
        console.print(f"[red]❌ File not found: {file_path}[/red]")
        sys.exit(1)

    # Step 2: Load data
    console.print(f"\n[dim]Loading {file_path.name}...[/dim]")
    chunks = load_ingestion_file(file_path)

    if not chunks:
        console.print("[red]❌ No chunks found in file[/red]")
        sys.exit(1)

    console.print(f"[green]✅ Loaded {len(chunks)} chunks[/green]")

    # Step 3: Load model and compute embeddings
    console.print("\n[dim]Loading embedding model (all-MiniLM-L6-v2)...[/dim]")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    console.print("[dim]Computing embeddings for all chunks...[/dim]")
    content_list = [c.get("content", "") for c in chunks]
    embeddings = model.encode(content_list, show_progress_bar=False)
    console.print(f"[green]✅ Embeddings computed ({embeddings.shape})[/green]")

    # Display welcome banner
    console.print()
    display_welcome_banner(file_path, len(chunks))

    # Settings
    top_n = 3

    # Help hint
    console.print("[dim]Type 'help' for commands, or just enter a query to search.[/dim]")
    console.print("[dim]Type 'exit' or 'quit' to close.[/dim]\n")

    # Step 4: Interactive query loop
    while True:
        try:
            # Get user input
            query = Prompt.ask("[bold cyan]🔍 Query[/bold cyan]")
            if query is None:
                continue
            query = query.strip()

            if not query:
                continue

            # Handle commands
            query_lower = query.lower()

            if query_lower in ("exit", "quit", "q"):
                console.print("\n[bold green]👋 Goodbye![/bold green]\n")
                break

            elif query_lower in ("help", "?"):
                display_help()
                continue

            elif query_lower == "stats":
                display_stats(chunks)
                continue

            elif query_lower == "list":
                list_chunks(chunks)
                continue

            elif query_lower == "clear":
                console.clear()
                display_welcome_banner(file_path, len(chunks))
                continue

            elif query_lower.startswith("top "):
                try:
                    new_top = int(query_lower.split()[1])
                    if 1 <= new_top <= 20:
                        top_n = new_top
                        console.print(f"[green]✅ Now showing top {top_n} results[/green]")
                    else:
                        console.print("[yellow]Please use a number between 1 and 20[/yellow]")
                except (ValueError, IndexError):
                    console.print("[yellow]Usage: top N (e.g., 'top 5')[/yellow]")
                continue

            elif query_lower.startswith("page "):
                try:
                    page_num = int(query_lower.split()[1])
                    show_page_chunks(chunks, page_num)
                except (ValueError, IndexError):
                    console.print("[yellow]Usage: page N (e.g., 'page 5')[/yellow]")
                continue

            # Regular search query
            display_results(query, chunks, embeddings, model, top_n)

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]⚠️ Interrupted. Type 'exit' to quit.[/bold yellow]\n")
            continue

        except EOFError:
            console.print("\n[bold green]👋 Goodbye![/bold green]\n")
            break

        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            continue


if __name__ == "__main__":
    main()
