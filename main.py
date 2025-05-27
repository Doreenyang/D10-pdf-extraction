import os
import pytesseract
from PIL import Image
import pdf2image
import pandas as pd
import re
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import numpy as np
from collections import defaultdict

class ArchitecturalPDFExtractor:
    def __init__(self, pdf_path, output_excel="output.xlsx", temp_dir="temp_images"):
        self.pdf_path = pdf_path
        self.output_excel = output_excel
        self.temp_dir = temp_dir
        self.text_data = []
        self.image_data = []
        self.drawing_regions = []
        self.spec_sections = defaultdict(list)
        self.keywords = []
        self.doc_info = {}
        
        # Create temp directory if needed
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def set_keywords(self, keywords):
        """Set construction-specific keywords to search for"""
        self.keywords = keywords + [
            # Common architectural terms
            'scale', 'elevation', 'section', 'detail',
            # Specification sections
            'general', 'materials', 'execution',
            # Drawing elements
            'title block', 'revision', 'sheet number'
        ]
        
    def preprocess_image(self, image):
        """Enhance image quality for better OCR results"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Optional: denoising
            # processed = cv2.fastNlMeansDenoising(processed, h=10)
            
            return Image.fromarray(processed)
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return image
            
    def extract_with_ocr(self, dpi=300, parallel=True):
        """Optimized OCR extraction for large architectural drawings"""
        start_time = time.time()
        
        try:
            # Convert PDF to images with thread pooling
            images = pdf2image.convert_from_path(
                self.pdf_path,
                dpi=dpi,
                thread_count=4,
                fmt='jpeg',
                output_folder=self.temp_dir
            )
            
            print(f"Converted {len(images)} pages to images in {time.time() - start_time:.1f}s")
            
            def process_page(i, image):
                # Preprocess image for better OCR
                processed_img = self.preprocess_image(image)
                
                # Custom OCR config for architectural drawings
                custom_config = r'--oem 3 --psd 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.,:;/\()[]%°\'"'
                text = pytesseract.image_to_string(processed_img, config=custom_config)
                
                # Detect if this is likely a drawing sheet
                is_drawing = self._detect_drawing_sheet(text)
                
                return {
                    'page': i+1,
                    'text': text,
                    'source': 'ocr',
                    'is_drawing': is_drawing,
                    'processing_time': time.time() - start_time
                }
            
            # Process pages in parallel
            if parallel:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(
                        lambda x: process_page(x[0], x[1]),
                        enumerate(images)
                    ))
                self.text_data.extend(results)
            else:
                for i, image in enumerate(images):
                    self.text_data.append(process_page(i, image))
                    
        except Exception as e:
            print(f"OCR extraction failed: {str(e)}")
            
    def _detect_drawing_sheet(self, text):
        """Heuristics to identify drawing sheets vs specification pages"""
        text_lower = text.lower()
        
        # Common drawing sheet indicators
        drawing_indicators = [
            'scale', 'elevation', 'section', 'detail',
            'sheet no', 'drawing no', 'rev', 'revision',
            'title block', 'north arrow'
        ]
        
        return any(indicator in text_lower for indicator in drawing_indicators)
        
    def extract_native_text(self):
        """Enhanced native text extraction with structure detection"""
        try:
            doc = fitz.open(self.pdf_path)
            self.doc_info['page_count'] = len(doc)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_blocks = page.get_text("dict")["blocks"]
                
                page_text = ""
                drawing_elements = []
                
                for block in text_blocks:
                    if "lines" in block:  # Text block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                                
                        # Detect specification section headers
                        section = self._detect_spec_section(block_text)
                        if section:
                            self.spec_sections[section].append({
                                'page': page_num + 1,
                                'text': block_text.strip(),
                                'bbox': block["bbox"]
                            })
                            
                        page_text += block_text + "\n"
                        
                        # Store text block with position
                        self.text_data.append({
                            'page': page_num + 1,
                            'text': block_text.strip(),
                            'bbox': block["bbox"],
                            'type': 'text_block',
                            'source': 'native'
                        })
                    
                    elif "image" in block:  # Image block
                        drawing_elements.append(block["bbox"])
                        
                # Classify page type based on content
                is_drawing = self._detect_drawing_sheet(page_text)
                
                if is_drawing and drawing_elements:
                    self.drawing_regions.append({
                        'page': page_num + 1,
                        'regions': drawing_elements
                    })
                    
        except Exception as e:
            print(f"Native text extraction failed: {str(e)}")
            
    def _detect_spec_section(self, text):
        """Identify specification section headers"""
        text = text.strip().lower()
        
        section_keywords = {
            'general': ['general', 'scope', 'summary'],
            'materials': ['material', 'product', 'equipment'],
            'execution': ['execution', 'installation', 'erection']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in text for keyword in keywords):
                return section
        return None
        
    def extract_drawing_data(self):
        """Specialized extraction for drawing sheets"""
        drawing_data = []
        
        for entry in self.text_data:
            if entry.get('is_drawing', False):
                # Extract drawing metadata
                metadata = {
                    'page': entry['page'],
                    'type': 'drawing',
                    'title': self._extract_drawing_title(entry['text']),
                    'scale': self._extract_drawing_scale(entry['text']),
                    'number': self._extract_drawing_number(entry['text']),
                    'revisions': self._extract_revisions(entry['text'])
                }
                drawing_data.append(metadata)
                
        return drawing_data
        
    def _extract_drawing_title(self, text):
        """Extract drawing title from text"""
        # Look for common title patterns
        matches = re.findall(r'(?:drawing|sheet)\s*title:\s*(.+?)\n', text, re.IGNORECASE)
        return matches[0] if matches else ""
        
    def _extract_drawing_scale(self, text):
        """Extract drawing scale from text"""
        # Common scale patterns
        scale_patterns = [
            r'scale:\s*([0-9/]+"\s*=\s*[0-9\'-]+)',
            r'scale\s*([0-9/]+"\s*=\s*[0-9\'-]+)',
            r'([0-9/]+"\s*=\s*[0-9\'-]+)'
        ]
        
        for pattern in scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""
        
    def generate_excel(self):
        """Generate comprehensive Excel report for architectural documents"""
        # Create structured data
        structured_data = []
        
        # Add document info
        structured_data.append({
            'type': 'document_info',
            'content': f"Pages: {self.doc_info.get('page_count', 'N/A')}",
            'page': 'N/A'
        })
        
        # Add drawing data
        drawing_data = self.extract_drawing_data()
        for drawing in drawing_data:
            structured_data.append({
                'type': 'drawing_metadata',
                'page': drawing['page'],
                'title': drawing['title'],
                'scale': drawing['scale'],
                'drawing_number': drawing['number'],
                'revisions': ', '.join(drawing['revisions'])
            })
        
        # Add specification sections
        for section, items in self.spec_sections.items():
            for item in items:
                structured_data.append({
                    'type': 'spec_section',
                    'section': section,
                    'page': item['page'],
                    'content': item['text']
                })
        
        # Create Excel file
        writer = pd.ExcelWriter(self.output_excel, engine='xlsxwriter')
        
        # Main data sheet
        pd.DataFrame(structured_data).to_excel(
            writer, sheet_name='Document Structure', index=False)
            
        # Drawing register sheet
        if drawing_data:
            pd.DataFrame(drawing_data).to_excel(
                writer, sheet_name='Drawing Register', index=False)
                
        # Specification index sheet
        if self.spec_sections:
            spec_data = []
            for section, items in self.spec_sections.items():
                spec_data.extend([{
                    'section': section,
                    'page': item['page'],
                    'content': item['text']
                } for item in items])
                
            pd.DataFrame(spec_data).to_excel(
                writer, sheet_name='Specification Index', index=False)
                
        writer.close()
        
    def process_architectural_pdf(self):
        """Optimized processing pipeline for architectural PDFs"""
        print(f"Processing architectural PDF: {self.pdf_path}")
        
        # First try native extraction
        self.extract_native_text()
        
        # If little text found, try OCR (common for scanned drawings)
        if len(self.text_data) < self.doc_info.get('page_count', 0) * 0.5:
            print("Low text yield - attempting OCR extraction")
            self.extract_with_ocr(dpi=250)  # Lower DPI for faster processing
            
        # Extract drawing metadata
        drawing_data = self.extract_drawing_data()
        
        # Generate comprehensive report
        self.generate_excel()
        
        return {
            'page_count': self.doc_info.get('page_count', 0),
            'drawings': len(drawing_data),
            'spec_sections': len(self.spec_sections),
            'text_blocks': len(self.text_data)
        }
    

#usage

# Initialize with architectural PDF
extractor = ArchitecturalPDFExtractor(
    "static/construction_drawings.pdf",
    "static/construction_data.xlsx"
)

# Set construction-specific keywords
extractor.set_keywords([
    "HVAC", "signage", "structural", "MEP",
    "foundation", "finish", "scheduling"
])

# Process the document
results = extractor.process_architectural_pdf()

print(f"Processed {results['page_count']} pages")
print(f"Identified {results['drawings']} drawing sheets")
print(f"Found {results['spec_sections']} specification sections")