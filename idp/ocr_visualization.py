import streamlit as st
import pandas as pd
import io
from PIL import Image, ImageDraw
import fitz
from constants import OCR_PERFORMANCE_METRICS

def visualize_ocr_comparison(results):
    """Generate comparison visualization based on available metrics"""
    st.write("### OCR Performance Comparison")

    # Summary metrics in columns
    cols = st.columns(len(results))
    for idx, (provider, data) in enumerate(results.items()):
        with cols[idx]:
            # Ensure 'metrics' key exists and is a dict before accessing sub-keys
            metrics = data.get("metrics", {})
            word_count = metrics.get("word_count", 0) # Default to 0 if not found

            # Display metrics
            st.metric(
                label=f"{provider}",
                value=f"{word_count} words" # Display word count as primary value
            )
            
            # Show provider strengths
            provider_info = OCR_PERFORMANCE_METRICS["providers"][provider]
            st.write("**Best for:**")
            for strength in provider_info["strengths"][:2]:  # Show top 2 strengths
                st.write(f"- {strength}")
    
    # Detailed comparison table (removed score-based columns)
    st.write("#### Detailed Analysis")
    df = pd.DataFrame([{
        "Provider": provider,
        "Words": data.get("metrics", {}).get("word_count", "N/A"),
        "Lines": data.get("metrics", {}).get("line_count", "N/A"),
        "Chars": data.get("metrics", {}).get("char_count", "N/A"),
        "Avg Line Len": f"{data.get('metrics', {}).get('avg_line_length', 0):.1f}",
    } for provider, data in results.items()])
    
    st.dataframe(df.set_index("Provider"), use_container_width=True)
    

def visualize_ocr_results(page_image, parsed_elements):
    """Visualize detected elements on page"""
    try:
        img = Image.open(io.BytesIO(page_image))
        draw = ImageDraw.Draw(img)
        
        colors = {
            'text': '#00FF00',
            'table': '#0000FF',
            'image': '#FF0000',
            'heading': '#FFA500'
        }
        
        for element in parsed_elements:
            bbox = element.get('bbox')
            element_type = element.get('type', 'text')
            if bbox:
                draw.rectangle(bbox, outline=colors.get(element_type, '#FFFFFF'), width=2)
        
        return img
    except Exception as e:
        st.error(f"Error visualizing OCR results: {str(e)}")
        return None

def visualize_provider_parsing(provider, file_bytes, file_name, page_num=1):
    """Generate provider-specific parsing visualization"""
    try:
        pdf_doc = fitz.open(stream=file_bytes)
        page = pdf_doc[page_num-1]
        elements = []
        
        # Get text blocks with provider-specific handling
        page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_WHITESPACE) # Example flags, adjust if needed
        blocks = page_dict.get("blocks", [])
        
        for block in blocks:
            if block.get("type") == 0:  # Text
                # Simplify: Treat all text blocks detected by PyMuPDF as generic 'Text'
                elements.append({
                    "type": "text",
                    "bbox": block["bbox"],
                    "color": "#00FF00", # Green for text
                    "label": "Text"
                })
            
            elif block.get("type") == 1:  # Image
                elements.append({
                    "type": "image",
                    "bbox": block["bbox"],
                    "color": "#FF0000",
                    "label": "Image"
                })
                
        pdf_doc.close()
        return elements
        
    except Exception as e:
        st.error(f"Error in parsing visualization: {str(e)}")
        return []

def draw_parsing_visualization(file_bytes, page_num, elements):
    """Draw parsing visualization on page image"""
    try:
        with fitz.open(stream=file_bytes) as pdf_doc:
            page = pdf_doc[page_num - 1]
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            draw = ImageDraw.Draw(img)
            
            for elem in elements:
                bbox = elem["bbox"]
                draw.rectangle(
                    bbox,
                    outline=elem["color"],
                    width=2
                )
                
                # Add label if there's space
                if bbox[1] > 15:
                    draw.text(
                        (bbox[0], bbox[1]-15),
                        elem["label"],
                        fill=elem["color"]
                    )
            
            return img
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None