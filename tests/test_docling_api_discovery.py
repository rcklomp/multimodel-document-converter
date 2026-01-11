"""
Docling API Discovery Script

Purpose: Find the correct way to access layout elements from Docling v2.66.0
"""

from docling.document_converter import DocumentConverter
from pathlib import Path
import json

# Initialize converter
converter = DocumentConverter()

# Convert first page only (faster)
result = converter.convert("Firearms.pdf")

print("=" * 80)
print("DOCLING API DISCOVERY")
print("=" * 80)

# 1. Inspect ConversionResult structure
print("\n1. ConversionResult attributes:")
print(f"   Type: {type(result)}")
print(f"   Attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

# 2. Check if result has 'document' attribute
if hasattr(result, "document"):
    print("\n2. Document object found:")
    doc = result.document
    print(f"   Type: {type(doc)}")
    print(f"   Attributes: {[attr for attr in dir(doc) if not attr.startswith('_')]}")

    # 3. Check for pages
    if hasattr(doc, "pages"):
        print(f"\n3. Pages found: {len(doc.pages)} pages")

        page1 = doc.pages[0]
        print(f"   Page 1 type: {type(page1)}")
        print(f"   Page 1 attributes: {[attr for attr in dir(page1) if not attr.startswith('_')]}")

        # 4. Try common attributes
        print("\n4. Attempting to access page content:")

        # Try: body
        if hasattr(page1, "body"):
            print(f"   ✓ page.body exists: {type(page1.body)}")
            body = page1.body
            print(
                f"     Body attributes: {[attr for attr in dir(body) if not attr.startswith('_')]}"
            )

        # Try: cells
        if hasattr(page1, "cells"):
            print(f"   ✓ page.cells exists: {len(page1.cells)} cells")
            if len(page1.cells) > 0:
                cell1 = page1.cells[0]
                print(f"     Cell type: {type(cell1)}")
                print(
                    f"     Cell attributes: {[attr for attr in dir(cell1) if not attr.startswith('_')]}"
                )

        # Try: annotations
        if hasattr(page1, "annotations"):
            print(f"   ✓ page.annotations exists")

        # Try: get_text() or export methods
        if hasattr(page1, "export_to_markdown"):
            markdown = page1.export_to_markdown()
            print(f"   ✓ page.export_to_markdown() works: {len(markdown)} chars")
            print(f"     First 200 chars: {markdown[:200]}...")

        # Try: images
        if hasattr(page1, "images"):
            print(
                f"   ✓ page.images exists: {len(page1.images) if hasattr(page1.images, '__len__') else 'N/A'}"
            )

# 5. Try alternative: Export to dict/JSON
print("\n5. Export formats:")

if hasattr(result, "document"):
    doc = result.document

    # Try export_to_dict
    if hasattr(doc, "export_to_dict"):
        doc_dict = doc.export_to_dict()
        print(f"   ✓ document.export_to_dict() works")
        print(f"     Keys: {list(doc_dict.keys())}")

        # Save full structure for inspection
        with open("docling_structure.json", "w") as f:
            json.dump(doc_dict, f, indent=2, default=str)
        print("   ✓ Saved full structure to docling_structure.json")

    # Try export_to_markdown
    if hasattr(doc, "export_to_markdown"):
        markdown = doc.export_to_markdown()
        print(f"   ✓ document.export_to_markdown() works: {len(markdown)} chars")

        # Save markdown
        with open("docling_markdown.md", "w") as f:
            f.write(markdown)
        print("   ✓ Saved markdown to docling_markdown.md")

# 6. Check for layout information in the dict
print("\n6. Searching for layout/element information:")

if hasattr(result, "document"):
    doc = result.document
    if hasattr(doc, "export_to_dict"):
        doc_dict = doc.export_to_dict()

        # Look for 'body', 'elements', 'blocks', 'layout', etc.
        for key in ["body", "elements", "blocks", "layout", "content", "items"]:
            if key in doc_dict:
                print(f"   ✓ Found '{key}' in doc_dict")
                value = doc_dict[key]
                print(f"     Type: {type(value)}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"     Length: {len(value)}")
                    print(f"     First item type: {type(value[0])}")
                    print(
                        f"     First item keys: {list(value[0].keys()) if isinstance(value[0], dict) else 'N/A'}"
                    )

print("\n" + "=" * 80)
print("DISCOVERY COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Review 'docling_structure.json' to find layout elements")
print("2. Review 'docling_markdown.md' to verify content extraction")
print("3. Look for bounding boxes, element types, confidence scores")
