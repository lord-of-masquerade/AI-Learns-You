"""
src/pdf_analyzer.py
Utility for extracting text from PDFs and sending to Claude for analysis.
Used by app.py — can also be run standalone.
"""

import io
import os


def extract_text(file_obj) -> str:
    """
    Extract plain text from an uploaded file object (PDF or TXT).
    Returns empty string on failure.
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_obj.read())) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except ImportError:
        # Fallback: try reading as plain text (works for .txt uploads)
        try:
            file_obj.seek(0)
            return file_obj.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    except Exception as e:
        print(f"[pdf_analyzer] Extraction error: {e}")
        return ""


def analyse_document(text: str) -> str:
    """
    Send document text to Claude and return a structured study analysis.
    Requires ANTHROPIC_API_KEY env variable.
    """
    try:
        import anthropic
    except ImportError:
        return "❌ anthropic package not installed. Run: pip install anthropic"

    if not text.strip():
        return "❌ No text extracted from document."

    snippet = text[:3500]
    prompt = (
        "You are a study assistant. Analyse this document and provide:\n\n"
        "1. **Summary** (3–4 sentences)\n"
        "2. **Key Concepts** (bullet list of 5–8 main topics)\n"
        "3. **Weak Areas to Focus On** (what students typically miss)\n"
        "4. **Suggested Study Order** (how to tackle this material)\n\n"
        f"Document:\n{snippet}"
    )

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="You are a concise, helpful study assistant. Use clear markdown formatting.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/pdf_analyzer.py <path_to_pdf_or_txt>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        text = extract_text(f)

    print(f"\nExtracted {len(text):,} characters.\n")
    print("=" * 50)
    result = analyse_document(text)
    print(result)
