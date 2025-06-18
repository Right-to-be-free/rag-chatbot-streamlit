import os

def load_file(file_path: str) -> str:
    """Load text content from a file (supports .txt, .pdf, .csv, .docx)."""
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    text_content = ""
    if ext == ".txt":
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
    elif ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("Please install 'pymupdf' to enable PDF support.")
        doc = fitz.open(file_path)
        text_pages = [page.get_text() for page in doc]
        text_content = "\n".join(text_pages)
        doc.close()
    elif ext == ".csv":
        # Try using pandas for CSV, otherwise read as plain text
        try:
            import pandas as pd
        except ImportError:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        else:
            df = pd.read_csv(file_path, dtype=str, header=None)
            text_content = df.to_csv(index=False, header=False)
    elif ext == ".docx":
        try:
            import docx2txt
        except ImportError:
            try:
                from docx import Document
            except ImportError:
                raise ImportError("Please install 'python-docx' or 'docx2txt' to enable DOCX support.")
            doc = Document(file_path)
            full_text = [para.text for para in doc.paragraphs]
            text_content = "\n".join(full_text)
        else:
            text_content = docx2txt.process(file_path)
    else:
        # Fallback: try reading as text for other extensions
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
    return text_content or ""
