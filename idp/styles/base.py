def get_base_styles():
    return """
        .app-content {
            background-color: #0E1117;
            color: #FAFAFA;
            padding: 2rem;
            border-radius: 12px;
        }
        .content-header {
            font-size: 2rem;
            margin-bottom: 2rem;
            color: #FAFAFA;
        }
        .content-quote {
            background-color: rgba(26, 115, 232, 0.1);
            border-left: 4px solid #1a73e8;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 0 8px 8px 0;
        }
        .feature-section {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .feature-list {
            list-style: none;
            padding: 0;
            margin: 1rem 0;
        }
        .feature-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0;
            color: #FAFAFA;
        }
        .disclaimer {
            background-color: rgba(255, 59, 48, 0.1);
            border-left: 4px solid #ff3b30;
            padding: 1.5rem;
            margin-top: 2rem;
            border-radius: 0 8px 8px 0;
        }
    """
