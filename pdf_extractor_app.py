import os
import sys
import fitz  # PyMuPDF
import PySimpleGUI as sg
from datetime import datetime
from PIL import Image
import shutil

class ArchitecturalPDFConverter:
    def __init__(self):
        self.output_dir = "training_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def convert_pdf_to_images(self, pdf_path, dpi=600):
        """Convert PDF pages to high-res images"""
        try:
            doc = fitz.open(pdf_path)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render at high DPI
                zoom = dpi / 72  # 72 is default PDF DPI
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                output_path = os.path.join(
                    self.output_dir,
                    f"{base_name}_page{page_num+1:03d}.png"
                )
                pix.save(output_path)
                
            return True, len(doc)
        except Exception as e:
            return False, str(e)

    def batch_convert(self, folder_path, dpi=600):
        """Process all PDFs in a folder"""
        results = []
        for file in os.listdir(folder_path):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder_path, file)
                success, pages_or_error = self.convert_pdf_to_images(pdf_path, dpi)
                if success:
                    results.append(f"Converted {file} ({pages_or_error} pages)")
                else:
                    results.append(f"Failed {file}: {pages_or_error}")
        return results

def create_gui():
    sg.theme('LightBlue2')
    
    layout = [
        [sg.Text('Architectural Drawing Converter', font=('Helvetica', 16))],
        [sg.Text('DPI Setting:'), sg.Input('600', key='-DPI-', size=(5,1))],
        [
            sg.Button('Convert Single PDF'),
            sg.Button('Convert Folder'),
            sg.Button('Open Output'),
            sg.Button('Exit')
        ],
        [sg.Multiline(size=(80, 20), key='-OUTPUT-', autoscroll=True)],
        [sg.StatusBar('Ready', key='-STATUS-')]
    ]
    
    window = sg.Window('PDF to Training Data Converter', layout)
    converter = ArchitecturalPDFConverter()
    
    while True:
        event, values = window.read()
        
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
            
        elif event == 'Convert Single PDF':
            file = sg.popup_get_file('Select PDF', file_types=(("PDF Files", "*.pdf"),))
            if file:
                window['-STATUS-'].update('Converting...')
                window['-OUTPUT-'].update('')
                success, result = converter.convert_pdf_to_images(file, int(values['-DPI-']))
                if success:
                    window['-OUTPUT-'].print(f"Success! Created {result} image files")
                else:
                    window['-OUTPUT-'].print(f"Error: {result}")
                window['-STATUS-'].update('Ready')
                
        elif event == 'Convert Folder':
            folder = sg.popup_get_folder('Select folder with PDFs')
            if folder:
                window['-STATUS-'].update('Batch converting...')
                window['-OUTPUT-'].update('')
                results = converter.batch_convert(folder, int(values['-DPI-']))
                window['-OUTPUT-'].update('\n'.join(results))
                window['-STATUS-'].update('Batch complete')
                
        elif event == 'Open Output':
            if os.path.exists(converter.output_dir):
                os.startfile(converter.output_dir)
    
    window.close()

if __name__ == "__main__":
    # Check and install PyMuPDF if needed
    try:
        import fitz
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    
    create_gui()