"""
CLI - Typer-based Command Line Interface for V2 Document Processor
====================================================================
ENGINE_USE: Claude 4.5 Opus (Architect)

This module provides the CLI entrypoint for the V2 Document Processor with
configurable vision providers for VLM-based image enrichment.

Usage:
    mmrag-v2 process [FILE] --vision-provider [ollama|openai|anthropic|none]
    mmrag-v2 process document.pdf --vision-provider ollama --output-dir ./output
    mmrag-v2 process large.pdf --batch-size 10 --vision-provider ollama
    mmrag-v2 batch ./docs --vision-provider none

REQ Compliance:
- REQ-PDF-04: Uses Docling v2.66.0 with high-fidelity rendering
- REQ-PDF-05: Memory hygiene via gc.collect() between batches
- REQ-CHUNK-03: VLM descriptions truncated to 400 chars
- REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png
- REQ-STATE: Hierarchical breadcrumb tracking with batch continuity

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
Updated: 2025-12-29 (Batch Processing Engine)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# TYPE_CHECKING imports (no runtime cost)
if TYPE_CHECKING:
    from .batch_processor import BatchProcessor
    from .processor import V2DocumentProcessor

# Rich console for pretty output
console = Console()


def _print_startup_banner() -> None:
    """Print immediate startup banner so user knows CLI is responding."""
    console.print("[dim]🚀 MMRAG V2 CLI starting...[/dim]")


def _lazy_import_processor():
    """Lazy import V2DocumentProcessor to defer Docling load."""
    console.print("[dim]📦 Loading Docling engine...[/dim]")
    from .processor import V2DocumentProcessor

    return V2DocumentProcessor


def _lazy_import_batch_processor():
    """Lazy import BatchProcessor to defer Docling load."""
    console.print("[dim]📦 Loading batch processor...[/dim]")
    from .batch_processor import BatchProcessor

    return BatchProcessor


# Create Typer app
app = typer.Typer(
    name="mmrag-v2",
    help="Multimodal RAG Document Processor V2.0 - High-fidelity ETL with VLM enrichment",
    add_completion=False,
)


# ============================================================================
# ENUMS
# ============================================================================


class VisionProviderType(str, Enum):
    """Available vision provider types."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HAIKU = "haiku"
    NONE = "none"


class OCREngine(str, Enum):
    """Available OCR engines."""

    TESSERACT = "tesseract"
    EASYOCR = "easyocr"


class OCRMode(str, Enum):
    """OCR processing modes for scanned documents."""

    LEGACY = "legacy"  # Current shadow extraction
    LAYOUT_AWARE = "layout-aware"  # New layout-aware OCR cascade
    AUTO = "auto"  # Automatically detect based on document diagnostics


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=verbose,
            )
        ],
    )

    # Suppress noisy libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# API KEY RESOLUTION
# ============================================================================


def resolve_api_key(
    api_key: Optional[str],
    provider: VisionProviderType,
) -> Optional[str]:
    """Resolve API key from argument or environment variable."""
    if api_key:
        return api_key

    env_vars = {
        VisionProviderType.OPENAI: "OPENAI_API_KEY",
        VisionProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
        VisionProviderType.HAIKU: "ANTHROPIC_API_KEY",
    }

    env_var = env_vars.get(provider)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            console.print(f"[dim]Using API key from ${env_var}[/dim]")
            return key

    return None


# ============================================================================
# COMMANDS
# ============================================================================


@app.command("process")
def process_document(
    input_file: Path = typer.Argument(
        ...,
        help="Path to the document to process (PDF, EPUB, HTML, DOCX)",
        exists=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir",
        "-o",
        help="Directory for output files (JSONL and assets)",
    ),
    vision_provider: VisionProviderType = typer.Option(
        VisionProviderType.OLLAMA,
        "--vision-provider",
        "-v",
        help="Vision provider for image enrichment (default: ollama)",
        case_sensitive=False,
    ),
    vision_model: Optional[str] = typer.Option(
        None,
        "--vision-model",
        "-m",
        help="Vision model name for Ollama (optional - auto-detects loaded model if not specified)",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key for cloud providers",
        envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        "-b",
        help="Pages per batch for large PDFs (0=disable batching)",
    ),
    vlm_timeout: int = typer.Option(
        90,
        "--vlm-timeout",
        help="VLM read timeout in seconds (default: 90)",
    ),
    sensitivity: float = typer.Option(
        0.5,
        "--sensitivity",
        "-s",
        help="Vision extraction sensitivity (0.1=strict/large figures only, 1.0=maximum recall)",
        min=0.1,
        max=1.0,
    ),
    allow_fullpage_shadow: bool = typer.Option(
        False,
        "--allow-fullpage-shadow/--no-fullpage-shadow",
        help="Override Full-Page Guard to allow full-page shadow assets (use with caution)",
    ),
    strict_qa: bool = typer.Option(
        False,
        "--strict-qa/--no-strict-qa",
        help="Enable strict QA-CHECK-01 mode (fail on token validation errors)",
    ),
    semantic_overlap: bool = typer.Option(
        True,
        "--semantic-overlap/--no-semantic-overlap",
        help="Enable Dynamic Semantic Overlap (DSO) chunking for natural boundaries (Gap #3)",
    ),
    vlm_context_depth: int = typer.Option(
        3,
        "--vlm-context-depth",
        help="Number of previous text chunks to include as VLM context (Gap #3 semantic anchoring)",
        min=0,
        max=10,
    ),
    enable_ocr: bool = typer.Option(
        False,
        "--enable-ocr/--no-ocr",
        help="Enable OCR for scanned documents (disabled by default to avoid EasyOCR warnings)",
    ),
    ocr_engine: OCREngine = typer.Option(
        OCREngine.EASYOCR,
        "--ocr-engine",
        help="OCR engine to use",
        case_sensitive=False,
    ),
    ocr_mode: OCRMode = typer.Option(
        OCRMode.AUTO,
        "--ocr-mode",
        help="OCR processing mode: 'auto' (smart detection), 'legacy' (shadow extraction), or 'layout-aware' (3-layer OCR cascade)",
        case_sensitive=False,
    ),
    ocr_confidence_threshold: float = typer.Option(
        0.7,
        "--ocr-confidence-threshold",
        help="Minimum OCR confidence for layout-aware mode (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    enable_doctr: bool = typer.Option(
        True,
        "--enable-doctr/--no-doctr",
        help="Enable Doctr Layer 3 for layout-aware OCR (slower but more accurate)",
    ),
    enable_cache: bool = typer.Option(
        True,
        "--enable-cache/--no-cache",
        help="Enable vision cache for repeated images",
    ),
    pages: Optional[str] = typer.Option(
        None,
        "--pages",
        help="Specific pages to process (comma-separated, e.g., '6,21,169,241') or max count (e.g., '10')",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Process a single document with VLM-based image enrichment.

    For large PDFs, use --batch-size to enable memory-efficient batch processing.

    Examples:

        # Process with local Ollama (default)
        mmrag-v2 process document.pdf

        # Process large PDF with batch splitting (10 pages per batch)
        mmrag-v2 process large.pdf --batch-size 10

        # Process with OpenAI
        mmrag-v2 process document.pdf -v openai -k sk-xxx

        # Process without VLM
        mmrag-v2 process document.pdf --vision-provider none
    """
    setup_logging(verbose)

    # Validate API key for cloud providers
    resolved_key = resolve_api_key(api_key, vision_provider)

    cloud_providers = (
        VisionProviderType.OPENAI,
        VisionProviderType.ANTHROPIC,
        VisionProviderType.HAIKU,
    )
    if vision_provider in cloud_providers and not resolved_key:
        console.print(
            f"[red]Error:[/red] {vision_provider.value} requires an API key. "
            f"Use --api-key or set the appropriate environment variable.",
            style="bold red",
        )
        raise typer.Exit(code=1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir if enable_cache else None

    # Display configuration IMMEDIATELY (no imports yet)
    console.print("\n[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
    console.print("[bold]MMRAG V2 Document Processor[/bold]")
    console.print("[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
    console.print("[cyan]ENGINE_USE:[/cyan] Docling v2.66.0")
    console.print(f"[cyan]Input:[/cyan] {input_file}")
    console.print(f"[cyan]Output:[/cyan] {output_dir}")
    console.print(f"[cyan]Vision Provider:[/cyan] {vision_provider.value}")
    console.print(f"[cyan]Batch Size:[/cyan] {batch_size if batch_size > 0 else 'Disabled'}")
    console.print(f"[cyan]VLM Timeout:[/cyan] {vlm_timeout}s")
    ocr_str = f"Enabled ({ocr_engine.value})" if enable_ocr else "Disabled"
    console.print(f"[cyan]OCR:[/cyan] {ocr_str}")
    console.print(f"[cyan]Cache:[/cyan] {'Enabled' if enable_cache else 'Disabled'}")
    console.print(f"[cyan]Semantic Overlap:[/cyan] {'Enabled' if semantic_overlap else 'Disabled'}")
    console.print(f"[cyan]VLM Context Depth:[/cyan] {vlm_context_depth}")
    console.print(f"[cyan]Strict QA:[/cyan] {'Enabled' if strict_qa else 'Disabled'}")
    # Display OCR mode info
    if ocr_mode == OCRMode.LAYOUT_AWARE:
        console.print(
            f"[cyan]OCR Mode:[/cyan] [bold yellow]layout-aware[/bold yellow] (3-layer cascade)"
        )
        console.print(f"[cyan]OCR Confidence:[/cyan] {ocr_confidence_threshold}")
        console.print(f"[cyan]Doctr Layer 3:[/cyan] {'Enabled' if enable_doctr else 'Disabled'}")
    else:
        console.print(f"[cyan]OCR Mode:[/cyan] legacy")
    console.print("[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]\n")

    # Flush output immediately so user sees feedback
    import sys

    sys.stdout.flush()

    # Parse pages parameter - supports both comma-separated list and max count
    specific_pages: Optional[List[int]] = None
    max_pages: Optional[int] = None

    if pages:
        if "," in pages:
            # Comma-separated list of specific pages: "6,21,169,241"
            try:
                specific_pages = [int(p.strip()) for p in pages.split(",")]
                console.print(f"[cyan]Processing specific pages:[/cyan] {specific_pages}")
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid page list: '{pages}'. "
                    f"Use comma-separated numbers like '6,21,169,241'",
                    style="bold red",
                )
                raise typer.Exit(code=1)
        else:
            # Single number - treat as max pages
            try:
                max_pages = int(pages)
                console.print(f"[cyan]Processing max pages:[/cyan] {max_pages}")
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid page specification: '{pages}'. "
                    f"Use a number like '10' or comma-separated list like '6,21,169,241'",
                    style="bold red",
                )
                raise typer.Exit(code=1)

    try:
        is_pdf = input_file.suffix.lower() == ".pdf"
        use_batching = batch_size > 0 and is_pdf

        # Smart Vision Orchestration for PDFs
        # NOTE: SmartConfigProvider uses PyMuPDF (fast), not Docling
        # Docling is only loaded when BatchProcessor/V2DocumentProcessor starts
        extraction_strategy = None
        diagnostic_report = None  # GEMINI AUDIT FIX: Diagnostic Layer

        if is_pdf:
            console.print("[dim]🔍 Analyzing document (PyMuPDF)...[/dim]")
            sys.stdout.flush()

            from .orchestration.smart_config import SmartConfigProvider
            from .orchestration.strategy_orchestrator import StrategyOrchestrator
            from .orchestration.document_diagnostic import (
                DocumentDiagnosticEngine,
                create_diagnostic_engine,
            )
            from .orchestration.strategy_profiles import (
                ProfileManager,
                ProfileType,
            )

            # ================================================================
            # GEMINI AUDIT FIX: Run Document Diagnostic Layer FIRST
            # ================================================================
            # Step 0: Pre-flight diagnostic analysis
            console.print("[dim]🔍 Running document diagnostics...[/dim]")
            diagnostic_engine = create_diagnostic_engine(sample_pages=5)
            diagnostic_report = diagnostic_engine.analyze(input_file)

            # Print diagnostic results
            console.print()
            console.print("[bold cyan]━━━━━ DOCUMENT DIAGNOSTICS ━━━━━[/bold cyan]")
            console.print(
                f"[cyan]Modality:[/cyan] {diagnostic_report.physical_check.detected_modality.value}"
            )
            console.print(
                f"[cyan]File Size:[/cyan] {diagnostic_report.physical_check.file_size_mb:.1f} MB"
            )
            console.print(
                f"[cyan]Avg Text/Page:[/cyan] {diagnostic_report.physical_check.avg_text_per_page:.0f} chars"
            )
            console.print(
                f"[cyan]Is Likely Scan:[/cyan] {'Yes' if diagnostic_report.physical_check.is_likely_scan else 'No'}"
            )
            console.print(
                f"[cyan]Confidence:[/cyan] {diagnostic_report.confidence_profile.overall_confidence:.2f}"
            )
            console.print(
                f"[cyan]Era:[/cyan] {diagnostic_report.confidence_profile.detected_era.value}"
            )
            console.print(
                f"[cyan]Domain:[/cyan] {diagnostic_report.confidence_profile.detected_domain.value}"
            )
            console.print(f"[cyan]Strategy:[/cyan] {diagnostic_report.recommended_strategy}")

            if diagnostic_report.confidence_profile.warnings:
                console.print("[yellow]Warnings:[/yellow]")
                for warning in diagnostic_report.confidence_profile.warnings:
                    console.print(f"  [yellow]⚠ {warning}[/yellow]")

            console.print(f"[cyan]Reasoning:[/cyan] {diagnostic_report.physical_check.reasoning}")
            console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            console.print()

            # ================================================================
            # AUTO-PILOT: Select Strategy Profile based on diagnostics
            # ================================================================
            # This uses POLYMORPHISM instead of if/else chains
            # Digital magazines → DigitalMagazineProfile (no scan hints)
            # Scanned documents → ScannedDegradedProfile (with scan hints)
            console.print("[dim]🎯 Selecting strategy profile...[/dim]")
            selected_profile = ProfileManager.select_profile(
                diagnostic_report=diagnostic_report,
                force_profile=None,  # Let diagnostics decide
            )
            profile_params = selected_profile.get_parameters()

            # Print profile selection banner
            console.print()
            console.print("[bold magenta]━━━━━ STRATEGY PROFILE ━━━━━[/bold magenta]")
            console.print(f"[magenta]Profile:[/magenta] {selected_profile.name}")
            console.print(f"[magenta]Type:[/magenta] {selected_profile.profile_type.value}")
            console.print(f"[magenta]VLM Freedom:[/magenta] {profile_params.vlm_freedom.value}")
            console.print(
                f"[magenta]Scan Hints:[/magenta] {'Yes' if profile_params.inject_scan_hints else 'No'}"
            )
            console.print(
                f"[magenta]Min Dimensions:[/magenta] {profile_params.min_image_width}x{profile_params.min_image_height}px"
            )
            console.print(
                f"[magenta]Confidence Threshold:[/magenta] {profile_params.confidence_threshold:.1f}"
            )
            console.print("[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]")
            console.print()

            # Step 1: Analyze document to determine SmartConfig profile (for image stats)
            console.print("[dim]🔍 Analyzing document profile...[/dim]")
            analyzer = SmartConfigProvider()
            smart_profile = analyzer.analyze(input_file)

            # Step 2: Create extraction strategy - OVERRIDE sensitivity from profile
            # The profile parameters take precedence over CLI --sensitivity
            # This ensures digital magazines NEVER get scan settings
            orchestrator = StrategyOrchestrator()

            # Use profile sensitivity if this is a profile-driven flow
            effective_sensitivity = profile_params.sensitivity
            if sensitivity != 0.5:  # User explicitly set a sensitivity
                # User override takes precedence for manual tuning
                effective_sensitivity = sensitivity
                console.print(
                    f"[dim]Using user-specified sensitivity: {effective_sensitivity}[/dim]"
                )
            else:
                console.print(f"[dim]Using profile sensitivity: {effective_sensitivity}[/dim]")

            extraction_strategy = orchestrator.create_strategy(smart_profile, effective_sensitivity)

            # Override strategy parameters from profile for strict separation
            extraction_strategy.min_image_width = profile_params.min_image_width
            extraction_strategy.min_image_height = profile_params.min_image_height
            extraction_strategy.enable_shadow_extraction = profile_params.enable_shadow_extraction

            # GEMINI AUDIT FIX: Adjust strategy based on diagnostic results
            if diagnostic_report.should_force_scan_mode():
                console.print(
                    "[yellow]⚠ Diagnostic: Forcing scan mode due to physical checks[/yellow]"
                )
                # Enable OCR if document is likely a scan
                if not enable_ocr:
                    console.print(
                        "[yellow]  → Consider using --enable-ocr for better text extraction[/yellow]"
                    )

            # ================================================================
            # SMART OCR MODE AUTO-DETECTION (Phase 1B)
            # ================================================================
            # When ocr_mode == "auto", use diagnostics to select the best mode:
            # - Scanned/degraded documents → layout-aware (3-layer OCR cascade)
            # - Digital documents → legacy (shadow extraction)
            # ================================================================
            effective_ocr_mode = ocr_mode
            if ocr_mode == OCRMode.AUTO:
                is_scanned = diagnostic_report.physical_check.is_likely_scan
                detected_modality = diagnostic_report.physical_check.detected_modality.value

                if is_scanned or detected_modality in ("scanned", "scanned_degraded"):
                    effective_ocr_mode = OCRMode.LAYOUT_AWARE
                    console.print()
                    console.print("[bold green]━━━━━ SMART OCR DETECTION ━━━━━[/bold green]")
                    console.print(
                        f"[green]Auto-detected:[/green] Scanned document ({detected_modality})"
                    )
                    console.print(
                        "[green]OCR Mode:[/green] [bold]layout-aware[/bold] (3-layer OCR cascade)"
                    )
                    console.print("[dim]Override with --ocr-mode legacy if needed[/dim]")
                    console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]")
                    console.print()
                else:
                    effective_ocr_mode = OCRMode.LEGACY
                    console.print(
                        f"[dim]🔍 Auto-detected digital document → using legacy OCR mode[/dim]"
                    )

            # Step 3: Print strategy banner (provides immediate feedback)
            orchestrator.print_strategy_banner(extraction_strategy)

        # Default effective_ocr_mode for non-PDF files
        if not is_pdf:
            effective_ocr_mode = ocr_mode

        if use_batching:
            # Use BatchProcessor for large PDFs (lazy import)
            BatchProcessor = _lazy_import_batch_processor()
            processor = BatchProcessor(
                output_dir=str(output_dir),
                batch_size=batch_size,
                vision_provider=vision_provider.value,
                vision_model=vision_model,
                vision_api_key=resolved_key,
                vlm_timeout=vlm_timeout,
                enable_ocr=enable_ocr,
                ocr_engine=ocr_engine.value,
                extraction_strategy=extraction_strategy,
                max_pages=max_pages,
                specific_pages=specific_pages,
                allow_fullpage_shadow=allow_fullpage_shadow,
                strict_qa=strict_qa,
                semantic_overlap=semantic_overlap,
                vlm_context_depth=vlm_context_depth,
                # Layout-aware OCR parameters (Phase 1B) - USE effective_ocr_mode!
                ocr_mode=effective_ocr_mode.value,
                ocr_confidence_threshold=ocr_confidence_threshold,
                enable_doctr=enable_doctr,
            )

            # REQ-OCR-01: Pass profile parameters for OCR hints (ScannedDegradedProfile)
            # This enables the hybrid OCR+VLM layer for scanned documents
            if profile_params is not None:
                processor.set_profile_params(profile_params)
                if profile_params.enable_ocr_hints:
                    console.print()
                    console.print("[bold yellow]━━━━━ OCR-HYBRID LAYER ━━━━━[/bold yellow]")
                    console.print(f"[yellow]Status:[/yellow] ENABLED")
                    console.print(f"[yellow]Render DPI:[/yellow] {profile_params.render_dpi}")
                    console.print(
                        f"[yellow]OCR Min Confidence:[/yellow] {profile_params.ocr_min_confidence}"
                    )
                    console.print("[yellow]Mode:[/yellow] OCR hints → VLM Judge")
                    console.print("[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]")
                    console.print()

            result = processor.process_pdf(input_file)

            # Display results
            if result.success:
                console.print("\n[bold green]✓ Batch Processing Complete![/bold green]")
            else:
                console.print(
                    "\n[bold yellow]⚠ Batch Processing Completed with Errors[/bold yellow]"
                )

            console.print(f"[green]Output:[/green] {result.output_jsonl}")
            console.print(f"[green]Assets:[/green] {result.assets_dir}")
            console.print(f"[green]Batches:[/green] {result.batches_processed}")
            console.print(f"[green]Total Chunks:[/green] {result.total_chunks}")
            console.print(f"[green]Time:[/green] {result.processing_time_seconds:.1f}s")

            if result.vision_stats:
                console.print(f"\n[dim]Vision Stats:[/dim]")
                console.print(
                    f"  [dim]Cache size: {result.vision_stats.get('cache_size', 0)}[/dim]"
                )
                console.print(
                    f"  [dim]Processed: {result.vision_stats.get('processed_count', 0)}[/dim]"
                )

            if result.errors:
                console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
                for err in result.errors[:5]:
                    console.print(f"  [red]• {err}[/red]")

        else:
            # Use regular V2DocumentProcessor (lazy import)
            V2DocumentProcessor = _lazy_import_processor()
            proc = V2DocumentProcessor(
                output_dir=str(output_dir),
                enable_ocr=enable_ocr,
                ocr_engine=ocr_engine.value,
                max_pages=max_pages,
                vision_provider=vision_provider.value,
                vision_api_key=resolved_key,
                vision_cache_dir=cache_dir,
                extraction_strategy=extraction_strategy,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Processing document...", total=None)
                output_path = proc.process_to_jsonl(str(input_file))
                progress.update(task, completed=True)

            stats = proc.get_vision_stats()

            console.print("\n[bold green]✓ Processing Complete![/bold green]")
            console.print(f"[green]Output:[/green] {output_path}")
            console.print(f"[green]Assets:[/green] {output_dir / 'assets'}")

            if stats:
                console.print(f"\n[dim]Vision Stats:[/dim]")
                cache_size = stats.get("cache_size", 0)
                proc_count = stats.get("processed_count", 0)
                console.print(f"  [dim]Cache: {cache_size} images[/dim]")
                console.print(f"  [dim]Processed: {proc_count} images[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] File not found: {e}", style="bold red")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command("batch")
def batch_process(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing documents to process",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir",
        "-o",
        help="Directory for output files",
    ),
    pattern: str = typer.Option(
        "*.pdf",
        "--pattern",
        "-p",
        help="Glob pattern for files to process",
    ),
    vision_provider: VisionProviderType = typer.Option(
        VisionProviderType.OLLAMA,
        "--vision-provider",
        "-v",
        help="Vision provider for image enrichment",
        case_sensitive=False,
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key for cloud providers",
    ),
    enable_ocr: bool = typer.Option(
        False,
        "--enable-ocr/--no-ocr",
        help="Enable OCR for scanned documents",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Batch process multiple documents from a directory.

    Examples:

        # Process all PDFs in a directory
        mmrag-v2 batch ./documents --vision-provider ollama

        # Process with pattern matching
        mmrag-v2 batch ./documents -p "*.epub" -v none
    """
    setup_logging(verbose)

    files: List[Path] = list(input_dir.glob(pattern))

    if not files:
        console.print(f"[yellow]No files matching '{pattern}' found in {input_dir}[/yellow]")
        raise typer.Exit(code=0)

    resolved_key = resolve_api_key(api_key, vision_provider)

    cloud_providers = (
        VisionProviderType.OPENAI,
        VisionProviderType.ANTHROPIC,
        VisionProviderType.HAIKU,
    )
    if vision_provider in cloud_providers and not resolved_key:
        console.print(
            f"[red]Error:[/red] {vision_provider.value} requires an API key.",
            style="bold red",
        )
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Processing {len(files)} files...[/bold]\n")

    success_count = 0
    error_count = 0
    errors: List[str] = []

    # Lazy import once for all files
    V2DocumentProcessor = _lazy_import_processor()

    for file_path in files:
        try:
            console.print(f"[cyan]Processing:[/cyan] {file_path.name}")

            doc_output_dir = output_dir / file_path.stem
            doc_output_dir.mkdir(parents=True, exist_ok=True)

            processor = V2DocumentProcessor(
                output_dir=str(doc_output_dir),
                enable_ocr=enable_ocr,
                vision_provider=vision_provider.value,
                vision_api_key=resolved_key,
                vision_cache_dir=doc_output_dir,
            )

            processor.process_to_jsonl(str(file_path))

            console.print("  [green]✓ Complete[/green]")
            success_count += 1

        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            errors.append(f"{file_path.name}: {e}")
            error_count += 1

    console.print(f"\n[bold]Batch Processing Complete[/bold]")
    console.print(f"  [green]Success: {success_count}[/green]")
    if error_count > 0:
        console.print(f"  [red]Errors: {error_count}[/red]")

        error_log = output_dir / "ingestion_errors.log"
        with open(error_log, "w") as f:
            for error in errors:
                f.write(error + "\n")
        console.print(f"  [dim]Error log: {error_log}[/dim]")


@app.command("version")
def show_version() -> None:
    """Display version information."""
    console.print("\n[bold]MMRAG V2 Document Processor[/bold]")
    console.print("Version: 2.0.0")
    console.print("ENGINE_USE: Docling v2.66.0")
    console.print("\n[dim]Vision Providers:[/dim]")
    console.print("  • ollama    - Local llava (localhost:11434) [DEFAULT]")
    console.print("  • openai    - OpenAI GPT-4o-mini")
    console.print("  • anthropic - Claude 3.5 Haiku")
    console.print("  • none      - Fallback (breadcrumb + anchor text)")
    console.print("\n[dim]Batch Processing:[/dim]")
    console.print("  Use --batch-size N to split large PDFs into N-page batches")
    console.print("  Default: 10 pages per batch")


@app.command("check")
def check_providers() -> None:
    """Check availability of vision providers."""
    import requests

    console.print("\n[bold]Checking Vision Providers...[/bold]\n")

    # Check Ollama
    console.print("[cyan]Ollama[/cyan] (localhost:11434)")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            llava = [n for n in model_names if "llava" in n.lower()]
            if llava:
                console.print(f"  [green]✓ Available[/green] - llava: {llava}")
            else:
                console.print("  [yellow]⚠ Running but llava not found[/yellow]")
                console.print("    [dim]Run: ollama pull llava:latest[/dim]")
        else:
            console.print("  [red]✗ Not responding[/red]")
    except requests.ConnectionError:
        console.print("  [red]✗ Not running[/red]")
        console.print("    [dim]Start with: ollama serve[/dim]")
    except Exception as e:
        console.print(f"  [red]✗ Error: {e}[/red]")

    # Check OpenAI
    console.print("\n[cyan]OpenAI[/cyan]")
    if os.environ.get("OPENAI_API_KEY"):
        console.print("  [green]✓ OPENAI_API_KEY is set[/green]")
    else:
        console.print("  [yellow]⚠ OPENAI_API_KEY not set[/yellow]")

    # Check Anthropic
    console.print("\n[cyan]Anthropic[/cyan]")
    if os.environ.get("ANTHROPIC_API_KEY"):
        console.print("  [green]✓ ANTHROPIC_API_KEY is set[/green]")
    else:
        console.print("  [yellow]⚠ ANTHROPIC_API_KEY not set[/yellow]")

    console.print()


# ============================================================================
# ENTRYPOINT
# ============================================================================


def main() -> None:
    """Main entrypoint for CLI."""
    app()


if __name__ == "__main__":
    main()
