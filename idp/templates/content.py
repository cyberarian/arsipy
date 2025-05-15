__all__ = ['get_content_template']

def get_content_template():
    return """
        <div class="app-content">
            <h1 class="content-header">{title}</h1>
            
            <div class="content-quote">
                <p>{quote}</p>
            </div>

            <section class="feature-section limitations">
                <h3>💡 Keterbatasan OCR Tradisional</h3>
                <ul class="feature-list">
                    {traditional_limits}
                </ul>
            </section>

            <section class="feature-section advantages">
                <h3>✨ Keunggulan OCR dengan VLMs</h3>
                <ul class="feature-list">
                    {modern_features}
                </ul>
            </section>

            <div class="disclaimer">
                <h3>⚠️ Disclaimer</h3>
                <p>{disclaimer}</p>
            </div>
        </div>
    """
