OCR_METRICS = {
    "text_quality": {
        "good": 0.8,
        "medium": 0.5,
        "poor": 0.3
    },
    # ... rest of metrics configuration ...
}

OCR_PERFORMANCE_METRICS = {
    "providers": {
        "Mistral": {
            "ideal_for": ["Complex documents", "Tables", "Mixed layouts"],
            # ... provider specific metrics ...
        },
        # ... other providers ...
    }
}
