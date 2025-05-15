import os
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import argparse
import shutil

# --- Configuration & Constants ---
SIGNAGE_CSI_CODES = [
    "101400", "101416", "101419", "101423", "101426", "101433", "101436"
]
SIGNAGE_KEYWORDS = [
    "signage", "sign", "signs", "wayfinding", "room identification", "room id",
    "door sign", "directory", "directories", "monument sign", "pylon sign",
    "channel letter", "dimensional letter", "plaque", "plaques", "ada sign", "braille sign"
]
DRAWING_SIGNAGE_INDICATORS = [
    "sign schedule", "signage plan", "signage details", "sign type",
    "sign location plan", "keynote", "callout"
]

# --- Helper Functions for Text Analysis (Adapted from previous script) ---
def find_signage_in_text(page_text, page_number):
    """Searches for signage CSI codes and keywords in the text of a single page."""
    findings = []
    # Check for "101400s signage"
    for match in re.finditer(r"101400s?\s+signage", page_text, re.IGNORECASE):
        context = get_context(page_text, match.start(), match.end())
        findings.append({"page": page_number, "type": "Spec CSI", "term": "101400s signage", "context": context})

    # Check for other signage CSI codes
    for code in SIGNAGE_CSI_CODES:
        pattern_code = r"(^|\s|\n)" + re.escape(code) + r"([-\s\w]*)?(\s+[A-Za-z\s&]+Signage|[A-Za-z\s&]+Letters|[A-Za-z\s&]+Plaques|[A-Za-z\s&]+Signs)?"
        for match in re.finditer(pattern_code, page_text, re.IGNORECASE | re.MULTILINE):
            context = get_context(page_text, match.start(), match.end())
            findings.append({"page": page_number, "type": "Spec CSI", "term": code, "match": match.group(0).strip(), "context": context})

    # Check for general keywords
    combined_keywords = SIGNAGE_KEYWORDS + DRAWING_SIGNAGE_INDICATORS
    for keyword in combined_keywords:
        pattern_keyword = r"\b" + re.escape(keyword) + r"\b"
        for match in re.finditer(pattern_keyword, page_text, re.IGNORECASE):
            context = get_context(page_text, match.start(), match.end())
            findings.append({"page": page_number, "type": "Keyword", "term": keyword, "context": context})
    
    return findings

def get_context(text, start, end, window=150):
    """Extracts context around a match."""
    snippet_start = max(0, start - window)
    snippet_end = min(len(text), end + window)
    snippet = text[snippet_start:snippet_end]
    # Try to get full lines for the snippet
    first_newline_before = snippet.rfind("\n", 0, start - snippet_start)
    if first_newline_before != -1:
        snippet = snippet[first_newline_before+1:]
    last_newline_after = snippet.find("\n", end - snippet_start)
    if last_newline_after != -1:
        snippet = snippet[:last_newline_after]
    return snippet.strip().replace("\n", " ")

# --- Core PDF Processing Function ---
def process_pdf(pdf_path, output_base_dir):
    """Processes a single PDF: converts to images, OCRs, analyzes for signage."""
    pdf_filename = os.path.basename(pdf_path)
    project_name = os.path.splitext(pdf_filename)[0]
    project_output_dir = os.path.join(output_base_dir, project_name)
    
    # Create output directories
    ocr_text_dir = os.path.join(project_output_dir, "ocr_text_per_page")
    # images_dir = os.path.join(project_output_dir, "page_images") # Optional: if user wants to save images
    if os.path.exists(project_output_dir):
        shutil.rmtree(project_output_dir) # Clean up previous run for this PDF
    os.makedirs(ocr_text_dir, exist_ok=True)
    # os.makedirs(images_dir, exist_ok=True)

    print(f"Processing {pdf_filename}...")
    all_page_findings = []
    try:
        images = convert_from_path(pdf_path, dpi=300) # Use 300 DPI for better OCR
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return None

    for i, page_image in enumerate(images):
        page_num = i + 1
        print(f"  - OCRing page {page_num}/{len(images)}...")
        try:
            # page_image.save(os.path.join(images_dir, f"page_{page_num}.png"), "PNG") # Optional
            page_text = pytesseract.image_to_string(page_image, lang=\'eng\')
            with open(os.path.join(ocr_text_dir, f"page_{page_num}_text.txt"), "w", encoding="utf-8") as f_text:
                f_text.write(page_text)
            
            page_findings = find_signage_in_text(page_text, page_num)
            if page_findings:
                all_page_findings.extend(page_findings)
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract is not installed or not in your PATH.")
            return None
        except Exception as e:
            print(f"Error during OCR or analysis for page {page_num}: {e}")
            continue # Skip to next page if one page fails
    
    print(f"Finished OCR and initial analysis for {pdf_filename}.")
    return project_name, project_output_dir, all_page_findings

# --- Reporting Functions ---
def generate_markdown_report(project_name, project_output_dir, all_findings):
    md_content = f"# Signage Analysis Report: {project_name}\n\n"
    if not all_findings:
        md_content += "No signage-related keywords or CSI codes found in the document.\n"
    else:
        md_content += "## Summary of Findings:\n"
        md_content += f"- Total pages with potential signage references: {len(set(f[\'page\'] for f in all_findings))}\n\n"
        md_content += "## Detailed Findings by Page:\n"
        # Group findings by page
        findings_by_page = {}
        for item in all_findings:
            page = item["page"]
            if page not in findings_by_page:
                findings_by_page[page] = []
            findings_by_page[page].append(item)
        
        for page_num in sorted(findings_by_page.keys()):
            md_content += f"### Page {page_num}:\n"
            for finding in findings_by_page[page_num]:
                md_content += f"- **Type:** {finding[\'type\']}, **Term:** `{finding[\'term\]}`"
                if "match" in finding:
                    md_content += f", **Matched Text:** `{finding[\'match\]}`"
                md_content += f"\n  - **Context:** _{finding[\'context\]}_\n"
            md_content += "\n"

    report_path = os.path.join(project_output_dir, "PROJECT_SIGNAGE_OVERVIEW.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report generated: {report_path}")
    return report_path

def generate_excel_report(project_name, project_output_dir, all_findings):
    if not all_findings:
        print("No findings to generate Excel report.")
        return None

    df_data = []
    for finding in all_findings:
        df_data.append({
            "Project Name": project_name,
            "Page Number": finding["page"],
            "Finding Type": finding["type"],
            "Term/Code": finding["term"],
            "Matched Text (if any)": finding.get("match", "N/A"),
            "Context": finding["context"]
        })
    
    df = pd.DataFrame(df_data)
    report_path = os.path.join(project_output_dir, f"{project_name}_Signage_Details.xlsx")
    try:
        df.to_excel(report_path, index=False, sheet_name="Signage_Details")
        print(f"Excel report generated: {report_path}")
        return report_path
    except Exception as e:
        print(f"Error generating Excel report: {e}. Make sure 'openpyxl' is installed (pip install openpyxl)")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PDF architectural drawings for signage information using OCR.")
    parser.add_argument("pdf_file", help="Path to the PDF file to analyze.")
    parser.add_argument("--output_dir", default="./pdf_analysis_output", help="Directory to save analysis results.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file not found at {args.pdf_file}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    analysis_results = process_pdf(args.pdf_file, args.output_dir)

    if analysis_results:
        proj_name, proj_output_dir, findings = analysis_results
        generate_markdown_report(proj_name, proj_output_dir, findings)
        generate_excel_report(proj_name, proj_output_dir, findings)
        print(f"\nAnalysis complete. Results saved in: {proj_output_dir}")
    else:
        print("PDF analysis failed.")

