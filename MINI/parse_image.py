import cv2
import pytesseract
import pandas as pd
import re

def preprocess_image(image_path):
    """Preprocess the image for better OCR accuracy."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extract_text_with_boxes(image):
    """Extract text with bounding boxes for layout detection."""
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    return data

def extract_text_from_image(image):
    """Extract plain text using OCR."""
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def parse_text_to_sections_with_boxes(data):
    """Parse text using bounding boxes for better tabular structure."""
    sections = {}
    current_section = None
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 50:  # Only consider high-confidence OCR results
            text = data["text"][i].strip()
            if not text:
                continue
            
            # Detect section headers
            if re.match(r"(Test|Differential Leucocyte Count|Platelets|.*Haematology Report.*)", text, re.IGNORECASE):
                current_section = text
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(text)
    
    # Organize into structured format
    structured_data = {}
    for section, content in sections.items():
        rows = []
        for line in content:
            # Split lines into columns based on spaces
            columns = re.split(r'\s{2,}', line)
            rows.append(columns)
        df = pd.DataFrame(rows)
        structured_data[section] = df
    return structured_data

def parse_image(image_path):
    """Complete parsing workflow for medical reports."""
    preprocessed_image = preprocess_image(image_path)
    ocr_data = extract_text_with_boxes(preprocessed_image)
    parsed_sections = parse_text_to_sections_with_boxes(ocr_data)
    return parsed_sections
