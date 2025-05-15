from constants import OCR_METRICS # Removed unused import of OCR_PERFORMANCE_METRICS

def evaluate_ocr_quality(text: str, provider: str, metadata=None) -> dict:
    """
    Calculates basic observable metrics about the extracted OCR text.
    Does NOT calculate a composite 'quality score' as that requires ground truth.
    """
    if not text:
        return {
            "word_count": 0,
            "line_count": 0,
            "char_count": 0,
            "avg_line_length": 0.0,
        }

    metrics = {
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
        "char_count": len(text),
        "avg_line_length": len(text) / max(len(text.splitlines()), 1),
    }
    return metrics