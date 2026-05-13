"""Phase 3 — `B4B_FULL_DOC_PICTURE_DEDUP` regression tests.

See `docs/PLAN_V2.10.md` §"Phase 3". The pHash dedup at
`BatchProcessor` finalize-time was rejecting same-style figures on
consecutive Earthship pages and stylized chapter-intro illustrations
on Python Distilled, orphaning whole image-only pages and reporting
them as `MISSING_PAGES` at strict-gate time. The Phase 3 fix
preserves AT MOST one image per image-only page in pHash-dense docs.

The contract pinned here is universal page-shape, not doc-specific:

  - A duplicate IMAGE chunk on a page that has at least one
    `TEXT`/`TABLE` chunk surviving to export → continues to be
    rejected (page is already covered; storage dedup is the right
    call).
  - A duplicate IMAGE chunk on a page with NO surviving
    `TEXT`/`TABLE` chunks → preserved when it is the FIRST image
    on that page; subsequent duplicates on the same page still
    drop (one image is enough to cover the page).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    AssetReference,
    ChunkMetadata,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    SpatialMetadata,
    create_text_chunk,
)


def _identical_illustration_png(path: Path, seed: int = 0) -> None:
    """Write a deterministic non-blank illustration to disk.

    The pHash collapses to the same signature for the same seed; a
    different seed produces a different signature.
    """
    rng = np.random.default_rng(seed)
    # Generate a low-frequency pattern; pHash is dominated by the
    # low-frequency DCT coefficients so a low-frequency pattern
    # produces a stable, distinctive hash.
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    arr = np.repeat(np.repeat(arr, 4, axis=0), 4, axis=1)
    Image.fromarray(arr).save(path, "PNG")


def _image_chunk(
    *,
    page: int,
    asset_path: str,
    doc_id: str = "doc_phase3",
) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=f"{doc_id}_{page:03d}_image_{Path(asset_path).stem[-8:]}",
        doc_id=doc_id,
        modality=Modality.IMAGE,
        content=f"[Figure on page {page}]",
        asset_ref=AssetReference(file_path=asset_path),
        metadata=ChunkMetadata(
            page_number=page,
            chunk_type=None,
            source_file="phase3.pdf",
            file_type=FileType.PDF,
            position=page * 10,
            spatial=SpatialMetadata(bbox=[10, 10, 990, 990]),
            hierarchy=HierarchyMetadata(
                parent_heading=None,
                breadcrumb_path=["phase3", f"Page {page}"],
                level=2,
            ),
        ),
    )


def _build_bp(tmp_path: Path) -> BatchProcessor:
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    bp = BatchProcessor(output_dir=str(output))
    return bp


def _export_and_collect(
    bp: BatchProcessor, export_chunks: List[IngestionChunk]
) -> tuple[List[dict], dict]:
    """Run the BatchProcessor's export path and return (jsonl_rows, stats).

    Test wrapper for the finalize-time pHash dedup decision. The page
    bookkeeping mirrors the production loop in `batch_processor.py`
    (`_phash_image_only_pages` + `_phash_pages_with_exported_image`),
    and the carve-out decision delegates to the production helper
    `BatchProcessor._phash_carve_out_should_preserve_duplicate` so any
    future drift in that contract surfaces here too.
    """
    # Wire pHash registry on with the production threshold.
    from mmrag_v2.utils.image_hash_registry import create_image_hash_registry
    bp._image_hash_registry = create_image_hash_registry(threshold=10)
    bp._current_pdf_path = bp.output_dir.parent / "phase3.pdf"

    surviving: List[IngestionChunk] = []
    pages_with_non_image = {
        c.metadata.page_number
        for c in export_chunks
        if c.modality in (Modality.TEXT, Modality.TABLE)
        and c.metadata
        and c.metadata.page_number
    }
    pages_with_image_chunk = {
        c.metadata.page_number
        for c in export_chunks
        if c.modality == Modality.IMAGE
        and c.metadata
        and c.metadata.page_number
    }
    image_only_pages = pages_with_image_chunk - pages_with_non_image
    # Mirrors the production set: pages that have written at least one
    # IMAGE chunk during this run (unique OR preserved-duplicate).
    pages_with_exported_image: set[int] = set()
    duplicate_dropped = 0
    duplicate_preserved = 0
    for chunk in export_chunks:
        if chunk.modality != Modality.IMAGE:
            surviving.append(chunk)
            continue
        asset_ref = chunk.asset_ref
        if not (asset_ref and asset_ref.file_path):
            surviving.append(chunk)
            continue
        full_path = bp.output_dir / asset_ref.file_path
        if not full_path.exists():
            surviving.append(chunk)
            continue
        with Image.open(full_path) as img:
            dup = bp._image_hash_registry.check_and_register(
                image=img,
                page_number=chunk.metadata.page_number,
                asset_path=asset_ref.file_path,
            )
        page = chunk.metadata.page_number
        if dup.is_duplicate:
            if BatchProcessor._phash_carve_out_should_preserve_duplicate(
                page_number=page,
                image_only_pages=image_only_pages,
                pages_with_exported_image=pages_with_exported_image,
            ):
                pages_with_exported_image.add(page)
                duplicate_preserved += 1
                surviving.append(chunk)
            else:
                duplicate_dropped += 1
        else:
            if page is not None:
                pages_with_exported_image.add(page)
            surviving.append(chunk)
    stats = {
        "duplicate_dropped": duplicate_dropped,
        "duplicate_preserved": duplicate_preserved,
        "image_only_pages": sorted(image_only_pages),
        "pages_with_exported_image": sorted(pages_with_exported_image),
    }
    return [c.model_dump(mode="json") for c in surviving], stats


def test_phash_dedup_preserves_image_only_page_first_image(tmp_path) -> None:
    """The targeted Phase 3 contract: a page whose only export chunk is an
    IMAGE that pHash-collides with an earlier page must NOT have its
    image rejected. Otherwise the page becomes a strict-gate
    MISSING_PAGES failure.
    """
    bp = _build_bp(tmp_path)
    assets = bp.output_dir / "assets"
    a1 = "assets/doc_phase3_010_figure_00.png"
    a2 = "assets/doc_phase3_011_figure_00.png"
    # Same content (seed=42) → pHash-identical → distance=0 → duplicate.
    _identical_illustration_png(assets / "doc_phase3_010_figure_00.png", seed=42)
    _identical_illustration_png(assets / "doc_phase3_011_figure_00.png", seed=42)

    text_p10 = create_text_chunk(
        doc_id="doc_phase3",
        content=(
            "Page 10 has a substantial paragraph of body text so that the "
            "page is not an image-only page and the second figure on page "
            "11 should be tested in the image-only contract."
        ),
        source_file="phase3.pdf",
        file_type=FileType.PDF,
        page_number=10,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["phase3", "Page 10"],
            level=2,
        ),
    )
    img_p10 = _image_chunk(page=10, asset_path=a1)
    img_p11 = _image_chunk(page=11, asset_path=a2)

    surviving, stats = _export_and_collect(bp, [text_p10, img_p10, img_p11])

    # Page 10 keeps its image (first occurrence). Page 11's image is a
    # near-duplicate, but page 11 has no text/table chunk → must be
    # preserved. The export must therefore include BOTH images.
    pages = {row["metadata"]["page_number"] for row in surviving}
    assert 11 in pages, (
        f"Page 11 image-only page must retain its image despite pHash "
        f"collision; survivors: {sorted(pages)} stats={stats}"
    )
    assert stats["duplicate_preserved"] == 1
    assert stats["duplicate_dropped"] == 0


def test_phash_dedup_drops_duplicate_when_page_already_has_text(tmp_path) -> None:
    """Negative regression: when the page hosting the duplicate image
    already has surviving TEXT content, the existing dedup behavior
    (drop the duplicate) must hold. The Phase 3 carve-out is scoped
    to image-only pages and must NOT widen storage cost on text-bearing
    pages."""
    bp = _build_bp(tmp_path)
    assets = bp.output_dir / "assets"
    a1 = "assets/doc_phase3_020_figure_00.png"
    a2 = "assets/doc_phase3_021_figure_00.png"
    _identical_illustration_png(assets / "doc_phase3_020_figure_00.png", seed=7)
    _identical_illustration_png(assets / "doc_phase3_021_figure_00.png", seed=7)

    text_p20 = create_text_chunk(
        doc_id="doc_phase3",
        content="Page 20 body prose.",
        source_file="phase3.pdf",
        file_type=FileType.PDF,
        page_number=20,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["phase3", "Page 20"],
            level=2,
        ),
    )
    text_p21 = create_text_chunk(
        doc_id="doc_phase3",
        content="Page 21 has its own body paragraph here.",
        source_file="phase3.pdf",
        file_type=FileType.PDF,
        page_number=21,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["phase3", "Page 21"],
            level=2,
        ),
    )
    img_p20 = _image_chunk(page=20, asset_path=a1)
    img_p21 = _image_chunk(page=21, asset_path=a2)

    surviving, stats = _export_and_collect(
        bp, [text_p20, text_p21, img_p20, img_p21]
    )

    # Page 21 has a text chunk; the duplicate image must still drop.
    page_to_modalities = {}
    for row in surviving:
        p = row["metadata"]["page_number"]
        page_to_modalities.setdefault(p, set()).add(row["modality"])
    assert "image" not in page_to_modalities.get(21, set()), (
        "Duplicate image on a page that already has text content must "
        "continue to be rejected by pHash dedup (negative regression)."
    )
    assert "text" in page_to_modalities.get(21, set())
    assert stats["duplicate_dropped"] == 1
    assert stats["duplicate_preserved"] == 0


def test_phash_dedup_drops_duplicate_after_unique_image_on_same_image_only_page(
    tmp_path,
) -> None:
    """Regression: previously the carve-out only tracked
    `_phash_kept_first_on_page` (pages where a duplicate had been
    preserved). If page N first emitted a UNIQUE image and then a
    near-duplicate of an earlier page's image, the duplicate would
    still slip through because the bookkeeping had never recorded
    the unique emission. Contract: ANY image already exported for
    the page covers it, so the subsequent near-duplicate must drop.
    """
    bp = _build_bp(tmp_path)
    assets = bp.output_dir / "assets"

    # Page 50 has a UNIQUE image (seed=200 — distinct from the
    # decoy below). Page 51 is image-only with TWO images: the
    # first is unique to page 51 (seed=201, distinct), the second
    # is a near-duplicate of an OFF-PAGE asset already in the
    # registry (seed=300). The carve-out must preserve nothing on
    # page 51's second image because page 51 is already covered
    # by its first unique image.
    _identical_illustration_png(assets / "doc_phase3_050_figure_00.png", seed=200)
    _identical_illustration_png(assets / "doc_phase3_050_figure_extra.png", seed=300)
    _identical_illustration_png(assets / "doc_phase3_051_figure_00.png", seed=201)
    _identical_illustration_png(assets / "doc_phase3_051_figure_01.png", seed=300)

    text_p50 = create_text_chunk(
        doc_id="doc_phase3",
        content=(
            "Page 50 has a substantive paragraph so that page 50 is "
            "text-bearing — meaning the duplicate of seed=300 first "
            "registered here is NOT subject to the page-coverage "
            "carve-out on page 50 itself."
        ),
        source_file="phase3.pdf",
        file_type=FileType.PDF,
        page_number=50,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["phase3", "Page 50"],
            level=2,
        ),
    )
    img_p50 = _image_chunk(page=50, asset_path="assets/doc_phase3_050_figure_00.png")
    img_p50_extra = _image_chunk(
        page=50, asset_path="assets/doc_phase3_050_figure_extra.png"
    )
    img_p51_unique = _image_chunk(
        page=51, asset_path="assets/doc_phase3_051_figure_00.png"
    )
    img_p51_dup = _image_chunk(
        page=51, asset_path="assets/doc_phase3_051_figure_01.png"
    )

    surviving, stats = _export_and_collect(
        bp,
        [text_p50, img_p50, img_p50_extra, img_p51_unique, img_p51_dup],
    )

    # Page 50 is text-bearing — both its images may survive without
    # the carve-out. Page 51 is image-only — exactly one image must
    # survive (the unique one). The seed=300 duplicate must drop.
    p51_images = [
        row for row in surviving
        if row["metadata"]["page_number"] == 51 and row["modality"] == "image"
    ]
    assert len(p51_images) == 1, (
        f"Image-only page 51 must keep exactly one image (the unique "
        f"emission). The seed=300 near-duplicate must NOT bypass dedup "
        f"once a unique image is already exported for the page. "
        f"Got {len(p51_images)} survivor(s); stats={stats}"
    )
    # The survivor must be the unique image (figure_00), not the
    # near-duplicate (figure_01).
    survived_asset = p51_images[0]["asset_ref"]["file_path"]
    assert "doc_phase3_051_figure_00" in survived_asset, (
        f"Unique image on page 51 must be the survivor (not the duplicate). "
        f"Got: {survived_asset}"
    )
    assert stats["duplicate_dropped"] == 1
    assert stats["duplicate_preserved"] == 0


def test_phash_carve_out_decision_helper_pins_page_already_covered_branch() -> None:
    """Pin the pure decision contract directly (no PIL / file I/O).

    The helper returns False (drop) when the page already has any
    exported image, regardless of whether that earlier emission was
    unique or a preserved duplicate. This is the bug-fix contract:
    bookkeeping must reflect ALL exported images, not just preserved
    duplicates.
    """
    image_only = {7}
    # Empty exported set → preserve.
    assert BatchProcessor._phash_carve_out_should_preserve_duplicate(
        page_number=7,
        image_only_pages=image_only,
        pages_with_exported_image=set(),
    )
    # Page already has an exported image (the production loop adds
    # the page on BOTH unique and preserved-duplicate paths) → drop.
    assert not BatchProcessor._phash_carve_out_should_preserve_duplicate(
        page_number=7,
        image_only_pages=image_only,
        pages_with_exported_image={7},
    )
    # Page is text-bearing → drop regardless of bookkeeping.
    assert not BatchProcessor._phash_carve_out_should_preserve_duplicate(
        page_number=7,
        image_only_pages=set(),
        pages_with_exported_image=set(),
    )
    # Missing page_number → drop (safe default).
    assert not BatchProcessor._phash_carve_out_should_preserve_duplicate(
        page_number=None,
        image_only_pages=image_only,
        pages_with_exported_image=set(),
    )


def test_phash_dedup_drops_second_duplicate_on_image_only_page(tmp_path) -> None:
    """The carve-out preserves AT MOST one image per image-only page.
    A second pHash-colliding image on the same image-only page still
    drops, because the page is already covered after the first
    preservation.
    """
    bp = _build_bp(tmp_path)
    assets = bp.output_dir / "assets"
    # 3 same-seed images: first stamps the registry, second + third
    # both collide; on an image-only target page only one should be
    # preserved.
    a1 = "assets/doc_phase3_030_figure_00.png"
    a2 = "assets/doc_phase3_031_figure_00.png"
    a3 = "assets/doc_phase3_031_figure_01.png"
    _identical_illustration_png(assets / "doc_phase3_030_figure_00.png", seed=99)
    _identical_illustration_png(assets / "doc_phase3_031_figure_00.png", seed=99)
    _identical_illustration_png(assets / "doc_phase3_031_figure_01.png", seed=99)

    img_p30 = _image_chunk(page=30, asset_path=a1)
    img_p31_a = _image_chunk(page=31, asset_path=a2)
    img_p31_b = _image_chunk(page=31, asset_path=a3)
    # No text on either page → both are image-only

    surviving, stats = _export_and_collect(bp, [img_p30, img_p31_a, img_p31_b])

    page_image_count = {}
    for row in surviving:
        if row["modality"] == "image":
            p = row["metadata"]["page_number"]
            page_image_count[p] = page_image_count.get(p, 0) + 1
    # Page 30 keeps its first image (registry stamp).
    # Page 31 keeps exactly one (the first duplicate carve-out).
    # The second duplicate on page 31 drops because the page is already
    # covered.
    assert page_image_count.get(30) == 1
    assert page_image_count.get(31) == 1
    assert stats["duplicate_preserved"] == 1
    assert stats["duplicate_dropped"] == 1


def test_shadow_extraction_threshold_relaxes_when_page_has_no_prior_chunks() -> None:
    """Phase 3 (SHADOW-EXTRACTION lane): a mid-size image (453x258 — width
    above the 300 floor but height below it) on a page where Docling
    emitted nothing must pass the relaxed threshold. This is what
    closes Python_Distilled pp 686/688/913 (chapter-intro diagrams
    with 24-27% area, height ~258-290 — just below the standard
    300x300 gate, but the rendered page is non-blank so the strict
    gate flags MISSING_PAGES otherwise). Pinned via the pure
    `V2DocumentProcessor._shadow_image_meets_threshold` helper so the
    contract is exercised directly without needing a full processor
    instance.
    """
    from mmrag_v2.processor import V2DocumentProcessor

    # 453x258 image at 24.1 % area (Python_Distilled p686 shape).
    width, height, area = 453, 258, 0.241

    # Page already has chunks → standard threshold → NOT emitted.
    assert not V2DocumentProcessor._shadow_image_meets_threshold(
        img_width=width,
        img_height=height,
        area_ratio=area,
        page_needs_coverage=False,
    ), (
        "Mid-size 453x258 / 24 % image must fail the standard 300x300 + 40 % "
        "shadow threshold when the page already has chunks (negative regression)."
    )

    # Page has NO prior chunks → relaxed threshold → emitted.
    assert V2DocumentProcessor._shadow_image_meets_threshold(
        img_width=width,
        img_height=height,
        area_ratio=area,
        page_needs_coverage=True,
    ), (
        "Mid-size 453x258 / 24 % image must pass the relaxed 200x200 threshold "
        "when the page has no prior chunks (PLAN_V2.10 Phase 3 closes "
        "Python_Distilled pp 686/688/913 this way)."
    )


def test_shadow_extraction_relaxed_threshold_still_filters_tiny_icons() -> None:
    """Negative regression: even on a page with no prior chunks, the
    relaxed 200x200 floor must NOT pull in tiny decorative icons
    (100x100 et al). Those pages legitimately register as
    MISSING_PAGES_BLANK at the strict gate; we must not invent
    low-value image chunks for them."""
    from mmrag_v2.processor import V2DocumentProcessor

    # 100x100 icon at 1.6 % area on a 612x792 page.
    assert not V2DocumentProcessor._shadow_image_meets_threshold(
        img_width=100,
        img_height=100,
        area_ratio=0.016,
        page_needs_coverage=True,
    ), (
        "100x100 icon on an otherwise-empty page must still be filtered: "
        "the Phase 3 relaxation goes down to 200x200, not below."
    )

    # 150x150 still under the relaxed floor.
    assert not V2DocumentProcessor._shadow_image_meets_threshold(
        img_width=150,
        img_height=150,
        area_ratio=0.036,
        page_needs_coverage=True,
    )

    # Standard threshold also rejects a 200x200 image when the page is healthy.
    assert not V2DocumentProcessor._shadow_image_meets_threshold(
        img_width=200,
        img_height=200,
        area_ratio=0.065,
        page_needs_coverage=False,
    )


def test_shadow_extraction_threshold_keeps_full_page_image_in_both_lanes() -> None:
    """Earthship p109 shape: small page where the figure fills the
    entire page (area_ratio = 100 %). Must pass in BOTH lanes
    regardless of `page_needs_coverage` — neither the standard nor
    the relaxed gate may regress on full-page editorial imagery."""
    from mmrag_v2.processor import V2DocumentProcessor

    # 292x220 image on a 292x220 page = 100 % area.
    for coverage in (True, False):
        assert V2DocumentProcessor._shadow_image_meets_threshold(
            img_width=292,
            img_height=220,
            area_ratio=1.00,
            page_needs_coverage=coverage,
        ), (
            f"Full-page editorial image must pass the area gate in both "
            f"lanes (page_needs_coverage={coverage})."
        )


def test_phash_dedup_preserves_image_only_page_when_image_only_pages_set_is_built_correctly(tmp_path) -> None:
    """Guard against a regression where `image_only_pages` is computed
    over the wrong chunk list — e.g. if the set were computed over
    survivors instead of `export_chunks`, the page would be missing
    from the set after its first chunk dropped and the carve-out
    would never fire."""
    bp = _build_bp(tmp_path)
    assets = bp.output_dir / "assets"
    _identical_illustration_png(assets / "doc_phase3_040_figure_00.png", seed=11)
    _identical_illustration_png(assets / "doc_phase3_041_figure_00.png", seed=11)
    _identical_illustration_png(assets / "doc_phase3_042_figure_00.png", seed=11)

    chunks = [
        _image_chunk(page=40, asset_path="assets/doc_phase3_040_figure_00.png"),
        _image_chunk(page=41, asset_path="assets/doc_phase3_041_figure_00.png"),
        _image_chunk(page=42, asset_path="assets/doc_phase3_042_figure_00.png"),
    ]
    surviving, stats = _export_and_collect(bp, chunks)

    survivor_pages = {row["metadata"]["page_number"] for row in surviving}
    # All three image-only pages must keep an image chunk.
    assert survivor_pages == {40, 41, 42}, (
        f"All three image-only pages must retain at least one image. "
        f"Got: {sorted(survivor_pages)}; stats={stats}"
    )
    assert stats["duplicate_preserved"] == 2
    assert stats["duplicate_dropped"] == 0
