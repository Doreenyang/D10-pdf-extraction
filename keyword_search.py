import os
import sys
import subprocess
from PIL import Image
import pandas as pd
import re
from datetime import datetime

# Check and install required packages
def install_packages():
    required = {
        'pymupdf': 'fitz',
        'pytesseract': 'pytesseract',
        'pdf2image': 'pdf2image',
        'pandas': 'pandas',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'PySimpleGUI': 'PySimpleGUI'
    }
    
    missing = []
    for pkg, imp in required.items():
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

install_packages()

import fitz  # PyMuPDF
import pytesseract
import pdf2image
import PySimpleGUI as sg
import numpy as np
import cv2

class PDFExtractor:
    def __init__(self):
        self.text_data = []
        self.drawing_pages = set()
        self.keyword_hits = []
        
    def extract_text(self, pdf_path):
        """Extract text using PyMuPDF with OCR fallback"""
        self.text_data = []
        self.drawing_pages = set()
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # If little text found, try OCR
                if len(text.strip()) < 100:
                    try:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img = self.preprocess_image(img)
                        text = pytesseract.image_to_string(img)
                        self.drawing_pages.add(page_num + 1)
                    except Exception as e:
                        text = "[IMAGE/DRAWING PAGE]"
                        self.drawing_pages.add(page_num + 1)
                
                self.text_data.append({
                    'page': page_num + 1,
                    'text': text,
                    'is_drawing': (page_num + 1) in self.drawing_pages
                })
            
            return len(doc)
        except Exception as e:
            sg.popup_error(f"Failed to process PDF:\n{str(e)}")
            return 0
    
    def preprocess_image(self, img):
        """Enhance image quality for better OCR results"""
        try:
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return Image.fromarray(processed)
        except:
            return img
    
    def search_keywords(self, keywords):
        """Search for keywords across all pages"""
        self.keyword_hits = []
        
        for entry in self.text_data:
            page = entry['page']
            text = entry['text']
            
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    start = max(0, text.lower().index(keyword.lower()) - 20)
                    end = min(len(text), start + len(keyword) + 40)
                    context = text[start:end].replace('\n', ' ')
                    
                    self.keyword_hits.append({
                        'page': page,
                        'keyword': keyword,
                        'context': context,
                        'is_drawing': entry['is_drawing']
                    })
        
        return self.keyword_hits
    
    def generate_excel(self, output_path):
        """Generate Excel report with all data"""
        try:
            excel_data = []
            
            # Document summary
            excel_data.append({
                'Type': 'Document Summary',
                'Page': 'All',
                'Content': f"Total pages: {len(self.text_data)}",
                'Drawing Pages': len(self.drawing_pages),
                'Is Drawing': ''
            })
            
            # Keyword matches
            for hit in self.keyword_hits:
                excel_data.append({
                    'Type': 'Keyword Match',
                    'Page': hit['page'],
                    'Keyword': hit['keyword'],
                    'Content': hit['context'],
                    'Is Drawing': 'Yes' if hit['is_drawing'] else 'No'
                })
            
            # Page content
            for entry in self.text_data:
                content_preview = entry['text'][:200] + ("..." if len(entry['text']) > 200 else "")
                excel_data.append({
                    'Type': 'Page Content',
                    'Page': entry['page'],
                    'Content': content_preview,
                    'Full Content Length': len(entry['text']),
                    'Is Drawing': 'Yes' if entry['is_drawing'] else 'No'
                })
            
            pd.DataFrame(excel_data).to_excel(output_path, index=False)
            return True
        except Exception as e:
            sg.popup_error(f"Failed to generate Excel:\n{str(e)}")
            return False

def main():
    # Set simple theme without requiring latest PySimpleGUI
    sg.theme('SystemDefault')
    
    layout = [
        [sg.Text('PDF Extractor with Keyword Search', font=('Helvetica', 16))],
        [sg.Text('PDF File:'), sg.Input(key='-FILE-'), sg.FileBrowse(file_types=(("PDF Files", "*.pdf"),))],
        [sg.Text('Search Keywords (comma separated):'), sg.Input(key='-KEYWORDS-')],
        [sg.Button('Process PDF'), sg.Button('Search'), sg.Button('Export Excel'), sg.Button('Exit')],
        [sg.Multiline(size=(100, 25), key='-OUTPUT-', autoscroll=True, font=('Courier New', 10))],
        [sg.StatusBar('Ready', key='-STATUS-', size=(100, 1))]
    ]
    
    window = sg.Window('PDF Extraction Tool', layout)
    extractor = PDFExtractor()
    
    while True:
        event, values = window.read()
        
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
            
        elif event == 'Process PDF':
            if values['-FILE-'] and os.path.exists(values['-FILE-']):
                window['-STATUS-'].update('Processing PDF...')
                window['-OUTPUT-'].update('')
                window.refresh()
                
                total_pages = extractor.extract_text(values['-FILE-'])
                
                if total_pages > 0:
                    window['-OUTPUT-'].update(f"Processed {total_pages} pages\n")
                    window['-OUTPUT-'].print(f"Found {len(extractor.drawing_pages)} drawing pages")
                    window['-STATUS-'].update('Ready')
                else:
                    window['-OUTPUT-'].update("Failed to process PDF")
                    window['-STATUS-'].update('Error')
            else:
                sg.popup_error("Please select a valid PDF file")
                
        elif event == 'Search':
            if hasattr(extractor, 'text_data') and extractor.text_data:
                if values['-KEYWORDS-'].strip():
                    keywords = [k.strip() for k in values['-KEYWORDS-'].split(',') if k.strip()]
                    hits = extractor.search_keywords(keywords)
                    
                    if hits:
                        result = f"Found {len(hits)} matches:\n\n"
                        for hit in hits:
                            result += f"Page {hit['page']}: {hit['keyword']}\n"
                            result += f"Context: {hit['context']}\n\n"
                        window['-OUTPUT-'].update(result)
                        window['-STATUS-'].update(f"Found {len(hits)} matches")
                    else:
                        window['-OUTPUT-'].update("No matches found")
                        window['-STATUS-'].update('No matches')
                else:
                    sg.popup_error("Please enter keywords to search")
            else:
                sg.popup_error("Please process a PDF first")
                
        elif event == 'Export Excel':
            if hasattr(extractor, 'text_data') and extractor.text_data:
                default_name = "extracted_data.xlsx"
                if values['-FILE-']:
                    default_name = os.path.splitext(os.path.basename(values['-FILE-']))[0] + "_extracted.xlsx"
                
                output_file = sg.popup_get_file('Save Excel File', save_as=True, 
                                              default_extension='.xlsx',
                                              file_types=(("Excel Files", "*.xlsx"),),
                                              initial_folder=os.getcwd(),
                                              default_path=default_name)
                
                if output_file:
                    window['-STATUS-'].update('Exporting...')
                    window.refresh()
                    
                    if extractor.generate_excel(output_file):
                        sg.popup(f"Successfully exported to:\n{output_file}")
                        window['-STATUS-'].update('Export complete')
                    else:
                        window['-STATUS-'].update('Export failed')
            else:
                sg.popup_error("Please process a PDF first")
    
    window.close()

if __name__ == "__main__":
    main()