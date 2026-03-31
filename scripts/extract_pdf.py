"""
Extract text from PDF and save to file for inspection.
Simple script to verify PDF extraction works correctly.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from pypdf import PdfReader

pdf_path = "1-s2.0-S0140673625025036-main.pdf"
output_path = "article_text.txt"

print("=" * 70)
print("EXTRACTING TEXT FROM PDF")
print("=" * 70)
print(f"\nInput PDF: {pdf_path}")

reader = PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")

text_parts = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        text_parts.append(text)
        print(f"  Extracted page {i+1}/{len(reader.pages)} ({len(text)} chars)")

full_text = "\n\n".join(text_parts)

print(f"\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)
print(f"\nTotal characters: {len(full_text)}")
print(f"Total words: ~{len(full_text.split())}")
print(f"Total lines: {len(full_text.splitlines())}")

# Save to file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\nSaved to: {output_path}")
print("\nFirst 500 characters of extracted text:")
print("-" * 70)
print(full_text[:500])
print("-" * 70)
print("\nPlease review the file to ensure extraction looks correct.")
print("Then run: python run_pipeline_on_text.py")
