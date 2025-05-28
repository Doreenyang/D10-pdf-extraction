import os
import sys
import fitz  # PyMuPDF
import PySimpleGUI as sg
from datetime import datetime
from PIL import Image
import shutil
import subprocess

class ArchitecturalFileConverter:
    def __init__(self):
        self.output_dir = "training_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def convert_pdf_to_images(self, pdf_path, dpi=600):
        """Convert PDF pages to high-res images with proper DPI metadata"""
        try:
            doc = fitz.open(pdf_path)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render at high DPI
                zoom = dpi / 72  # 72 is default PDF DPI
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG with proper DPI metadata
                output_path = os.path.join(
                    self.output_dir,
                    f"{base_name}_page{page_num+1:03d}.png"
                )
                
                # Convert pixmap to PIL Image to set DPI properly
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.save(output_path, dpi=(dpi, dpi), quality=100)
                
            return True, len(doc)
        except Exception as e:
            return False, str(e)

    def process_image_file(self, image_path, target_dpi=600):
        """Process single image file and ensure target DPI in metadata"""
        try:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            with Image.open(image_path) as img:
                # Convert to RGB if needed (for PNG with alpha channel)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate scaling factor if image has DPI info
                current_dpi = img.info.get('dpi', (72, 72))[0]
                scaling = target_dpi / current_dpi if current_dpi > 0 else 1.0
                
                # Calculate new dimensions if scaling needed
                if scaling != 1.0:
                    new_width = int(img.width * scaling)
                    new_height = int(img.height * scaling)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Create output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    self.output_dir,
                    f"{base_name}_{timestamp}.png"
                )
                
                # Save with target DPI
                img.save(output_path, 'PNG', dpi=(target_dpi, target_dpi), quality=100)
                
            return True, output_path
        except Exception as e:
            return False, str(e)

    def batch_convert(self, folder_path, dpi=600):
        """Process all supported files in a folder with target DPI"""
        results = []
        supported_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        
        for file in os.listdir(folder_path):
            file_lower = file.lower()
            file_path = os.path.join(folder_path, file)
            
            try:
                if file_lower.endswith('.pdf'):
                    success, pages_or_error = self.convert_pdf_to_images(file_path, dpi)
                    if success:
                        results.append(f"Converted PDF {file} ({pages_or_error} pages @ {dpi} DPI)")
                    else:
                        results.append(f"Failed PDF {file}: {pages_or_error}")
                elif any(file_lower.endswith(ext) for ext in supported_extensions[1:]):
                    success, output_or_error = self.process_image_file(file_path, dpi)
                    if success:
                        # Verify output DPI
                        with Image.open(output_or_error) as img:
                            output_dpi = img.info.get('dpi', (0, 0))[0]
                        results.append(f"Processed image {file} -> {os.path.basename(output_or_error)} @ {output_dpi} DPI")
                    else:
                        results.append(f"Failed image {file}: {output_or_error}")
            except Exception as e:
                results.append(f"Error processing {file}: {str(e)}")
                    
        return results

def create_gui():
    sg.theme('LightBlue2')
    
    layout = [
        [sg.Text('Architectural Drawing Converter', font=('Helvetica', 16))],
        [sg.Text('Target DPI:'), sg.Input('600', key='-DPI-', size=(5,1)), 
         sg.Text('(Recommended: 300-600 for high quality)')],
        [
            sg.Button('Convert Single File'),
            sg.Button('Convert Folder'),
            sg.Button('Open Output'),
            sg.Button('Exit')
        ],
        [sg.Multiline(size=(80, 20), key='-OUTPUT-', autoscroll=True, reroute_stdout=True, reroute_stderr=True)],
        [sg.StatusBar('Ready', key='-STATUS-')]
    ]
    
    window = sg.Window('File to Training Data Converter', layout, finalize=True)
    converter = ArchitecturalFileConverter()
    
    while True:
        event, values = window.read()
        
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
            
        elif event == 'Convert Single File':
            file = sg.popup_get_file('Select File', file_types=(
                ("Supported Files", "*.pdf;*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp"),
                ("PDF Files", "*.pdf"),
                ("Image Files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp"),
                ("All Files", "*.*")
            ))
            if file:
                window['-STATUS-'].update('Converting...')
                window['-OUTPUT-'].update('')
                
                try:
                    dpi = int(values['-DPI-'])
                    if dpi <= 0:
                        raise ValueError
                except:
                    window['-OUTPUT-'].print("Invalid DPI value. Using default 600 DPI")
                    dpi = 600
                
                if file.lower().endswith('.pdf'):
                    success, result = converter.convert_pdf_to_images(file, dpi)
                    if success:
                        window['-OUTPUT-'].print(f"Success! Created {result} image files @ {dpi} DPI")
                    else:
                        window['-OUTPUT-'].print(f"Error: {result}")
                else:
                    success, result = converter.process_image_file(file, dpi)
                    if success:
                        # Verify output DPI
                        with Image.open(result) as img:
                            output_dpi = img.info.get('dpi', (0, 0))[0]
                        window['-OUTPUT-'].print(f"Success! Processed image @ {output_dpi} DPI: {result}")
                    else:
                        window['-OUTPUT-'].print(f"Error: {result}")
                        
                window['-STATUS-'].update('Ready')
                
        elif event == 'Convert Folder':
            folder = sg.popup_get_folder('Select folder with files')
            if folder:
                window['-STATUS-'].update('Batch converting...')
                window['-OUTPUT-'].update('')
                
                try:
                    dpi = int(values['-DPI-'])
                    if dpi <= 0:
                        raise ValueError
                except:
                    window['-OUTPUT-'].print("Invalid DPI value. Using default 600 DPI")
                    dpi = 600
                
                results = converter.batch_convert(folder, dpi)
                window['-OUTPUT-'].update('\n'.join(results))
                window['-STATUS-'].update('Batch complete')
                
        elif event == 'Open Output':
            if os.path.exists(converter.output_dir):
                os.startfile(converter.output_dir)
    
    window.close()

if __name__ == "__main__":
    # Check and install required packages if needed
    try:
        import fitz
        from PIL import Image
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf", "pillow"])
    
    create_gui()