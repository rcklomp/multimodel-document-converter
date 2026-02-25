"""
Docling API Discovery Script.

Note: This file is script-oriented and intentionally does not execute at import
time, so pytest collection remains hermetic.
"""

from pathlib import Path
import json


def main() -> None:
    from docling.document_converter import DocumentConverter

    pdf_path = Path("Firearms.pdf")
    if not pdf_path.exists():
        print(f"Missing input PDF: {pdf_path}")
        return

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    print("=" * 80)
    print("DOCLING API DISCOVERY")
    print("=" * 80)

    print("\n1. ConversionResult attributes:")
    print(f"Type: {type(result)}")
    print(f"Attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

    if hasattr(result, "document"):
        doc = result.document
        print("\n2. Document object found:")
        print(f"Type: {type(doc)}")
        print(f"Attributes: {[attr for attr in dir(doc) if not attr.startswith('_')]}")

        if hasattr(doc, "export_to_dict"):
            doc_dict = doc.export_to_dict()
            print(f"export_to_dict keys: {list(doc_dict.keys())}")
            with open("docling_structure.json", "w", encoding="utf-8") as f:
                json.dump(doc_dict, f, indent=2, default=str)
            print("Saved docling_structure.json")

        if hasattr(doc, "export_to_markdown"):
            markdown = doc.export_to_markdown()
            with open("docling_markdown.md", "w", encoding="utf-8") as f:
                f.write(markdown)
            print("Saved docling_markdown.md")


if __name__ == "__main__":
    main()
