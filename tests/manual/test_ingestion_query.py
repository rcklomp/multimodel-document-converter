#!/usr/bin/env python3
"""
MMRAG V3 - PROFESSIONAL MULTIMODAL SEARCH ENGINE
================================================
Architectuur: Multimodale Vector Indexering
Ondersteunt: Tekst, Beeldomschrijvingen en Refined Content
"""

import json
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Dependencies laden
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    print(
        "❌ Dependencies missen. Installeer met: pip install sentence-transformers scikit-learn numpy rich"
    )
    sys.exit(1)

console = Console()


def get_file_path():
    """Opent een TKInter browser voor selectie van het JSONL bestand."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Selecteer Ingestion JSONL bestand",
        filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def load_and_index(file_path):
    """Laadt data en bouwt een verzwaarde multimodale index."""
    chunks = []
    with console.status(f"[bold green]Laden van {file_path.name}..."):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # We bouwen een verzwaarde zoekindex
    indexing_content = []
    for c in chunks:
        meta = c.get("metadata", {})
        raw = c.get("content", "") or ""
        refined = meta.get("refined_content", "") or ""
        visual = meta.get("visual_description", "") or ""

        # De architectuur van de index: we herhalen metadata om de focus daarop te leggen
        # Dit zorgt dat beelden en opgeschoonde tekst boven de ruwe OCR uitstijgen
        weighted_string = f"{refined} {visual} {visual} {raw}".strip()
        indexing_content.append(weighted_string)

    with console.status("[bold blue]Berekenen van Multimodale Embeddings..."):
        embeddings = model.encode(indexing_content, show_progress_bar=False)

    return chunks, embeddings, model


def display_results(query, chunks, embeddings, model, top_n=5):
    """Zoekt en presenteert resultaten visueel aantrekkelijk."""
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    table = Table(title=f"\n🔍 Resultaten voor: '{query}'", expand=True, border_style="cyan")
    table.add_column("Score", width=8, justify="right", style="green")
    table.add_column("Page", width=6, justify="center", style="yellow")
    table.add_column("Type", width=10, style="magenta")
    table.add_column("Content Preview (AI Optimized)", style="white")

    for idx in top_indices:
        score = similarities[idx]
        if score < 0.2:
            continue  # Filter irrelevante ruis

        chunk = chunks[idx]
        meta = chunk.get("metadata", {})

        # Bepaal de beste content om te tonen
        visual = meta.get("visual_description")
        refined = meta.get("refined_content")
        raw = chunk.get("content", "")
        modality = chunk.get("modality", "text").upper()

        if visual:
            display_text = f"[bold italic magenta]📸 BEELD:[/bold italic magenta] {visual}"
        elif refined:
            display_text = f"[bold green]✨ REFINED:[/bold green] {refined}"
        else:
            display_text = raw

        preview = display_text[:250].replace("\n", " ") + "..."

        table.add_row(f"{score:.4f}", str(meta.get("page_number", "?")), modality, preview)

    console.print(table)


def main():
    console.clear()
    console.print(
        Panel.fit("🚀 MMRAG SEARCH ARCHITECT V3\nMultimodale Vector Zoekmachine", style="bold blue")
    )

    file_path = get_file_path()
    if not file_path:
        console.print("[red]Geen bestand geselecteerd. Afsluiten.[/red]")
        return

    chunks, embeddings, model = load_and_index(file_path)
    console.print(f"[green]✅ {len(chunks)} chunks succesvol geïndexeerd![/green]")
    console.print("[dim]Typ 'exit' of 'stop' om te beëindigen.[/dim]\n")

    while True:
        query = Prompt.ask("[bold cyan]Wat wil je zoeken?[/bold cyan]")

        if query.lower() in ("exit", "stop", "quit", "q"):
            console.print("[yellow]Applicatie gestopt. Tot ziens![/yellow]")
            break

        if not query.strip():
            continue

        display_results(query, chunks, embeddings, model)


if __name__ == "__main__":
    main()
