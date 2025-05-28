import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from PIL import Image
import PySimpleGUI as sg
import shutil
from datetime import datetime

# Set Tesseract path (change this to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Doreen\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class ArchitecturalDrawingAnalyzer:
    def __init__(self):
        self.output_dir = "output_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def pdf_to_images(self, pdf_path, dpi=300):
        """Convert PDF pages to images with improved line preservation"""
        doc = fitz.open(pdf_path)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        image_paths = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)  # Remove alpha for cleaner B/W
            
            # Enhance lines for better detection
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = self.enhance_lines(np.array(img))
            
            output_path = os.path.join(self.output_dir, f"{base_name}_page{page_num+1:03d}.png")
            Image.fromarray(img).save(output_path, dpi=(dpi, dpi))
            image_paths.append(output_path)
            
        return image_paths

    def enhance_lines(self, img):
        """Improve line visibility in architectural drawings"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Binarization with adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to 3-channel for consistency
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

    def detect_objects(self, image_path):
        """Simplified object detection for demo purposes"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find contours (simulating object detection)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 10000:  # Filter by size
                x,y,w,h = cv2.boundingRect(cnt)
                
                # Check if this might be signage (contains text)
                roi = img[y:y+h, x:x+w]
                text = pytesseract.image_to_string(roi, config='--psm 6')
                
                if text.strip():
                    detections.append({
                        'type': 'signage',
                        'bbox': [x,y,x+w,y+h],
                        'confidence': min(0.9, area/5000),  # Fake confidence for demo
                        'text': text.strip()
                    })
        
        return detections

    def analyze_pdf(self, pdf_path):
        """Full analysis pipeline"""
        image_paths = self.pdf_to_images(pdf_path)
        results = []
        
        for img_path in image_paths:
            page_num = int(os.path.basename(img_path).split('_page')[-1].split('.')[0])
            detections = self.detect_objects(img_path)
            
            # Draw visualization
            img = cv2.imread(img_path)
            for det in detections:
                x1,y1,x2,y2 = det['bbox']
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, f"{det['type']}: {det['text']}", 
                           (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            # Save visualization
            vis_path = os.path.join(self.output_dir, f"vis_{os.path.basename(img_path)}")
            cv2.imwrite(vis_path, img)
            
            results.append({
                'page': page_num,
                'image': vis_path,
                'detections': detections
            })
        
        return results

def create_gui():
    sg.theme('LightGrey1')
    
    layout = [
        [sg.Text('Architectural Drawing Analyzer', font=('Helvetica', 16))],
        [sg.Text('Select PDF:'), sg.Input(key='-PDF-'), sg.FileBrowse(file_types=(("PDF Files", "*.pdf"),))],
        [sg.Button('Analyze'), sg.Button('Open Results'), sg.Button('Exit')],
        [sg.Multiline(size=(80, 15), key='-OUTPUT-', autoscroll=True)],
        [sg.Image(key='-IMAGE-', size=(600, 400))],
        [sg.StatusBar('Ready', key='-STATUS-')]
    ]
    
    window = sg.Window('Drawing Analyzer Demo', layout, resizable=True)
    analyzer = ArchitecturalDrawingAnalyzer()
    
    while True:
        event, values = window.read()
        
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
            
        elif event == 'Analyze':
            pdf_path = values['-PDF-']
            if pdf_path:
                window['-STATUS-'].update('Analyzing...')
                window['-OUTPUT-'].update('')
                
                try:
                    results = analyzer.analyze_pdf(pdf_path)
                    window['-OUTPUT-'].print(f"Analysis complete for {os.path.basename(pdf_path)}")
                    
                    # Show first page results
                    if results:
                        first_page = results[0]
                        window['-OUTPUT-'].print(f"Page 1 Detections:")
                        for det in first_page['detections']:
                            window['-OUTPUT-'].print(f"- {det['type']}: {det['text']} (confidence: {det['confidence']:.2f})")
                        
                        # Display visualization
                        window['-IMAGE-'].update(filename=first_page['image'])
                    
                except Exception as e:
                    window['-OUTPUT-'].print(f"Error: {str(e)}")
                
                window['-STATUS-'].update('Ready')
                
        elif event == 'Open Results':
            if os.path.exists(analyzer.output_dir):
                os.startfile(analyzer.output_dir)
    
    window.close()

if __name__ == "__main__":
    # Check for required packages
    try:
        import fitz
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf", "opencv-python", "pytesseract", "pillow"])
    
    create_gui()