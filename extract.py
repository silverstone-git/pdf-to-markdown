import fitz
import pdfplumber
import re
import yaml
import pytesseract
import numpy as np
from typing import Literal, final
from PIL import Image
import os
import logging
import traceback
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import argparse
import io
from google import genai
from google.genai import types
import mimetypes
from gradio_client import Client, handle_file
import gradio as gr
import tempfile



warnings.filterwarnings("ignore")

config= {}
try:
    with open(Path("config/config.yaml").resolve(), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config= {
        'OUTPUT_DIR': '.',
        'PAGE_DELIMITER': '____NEXT PAGE____'
    }
except Exception as e:
    print("unhandled while opening default config in pdf2markdown: ", e)


class PDFExtractor(ABC):
    """Abstract base class for PDF extraction."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{Path(__file__).stem}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def extract(self) ->  tuple[object, list[object]] | tuple[Literal[''], list[object]] | None:
        """Abstract method for extracting content from PDF."""
        pass


class MarkdownPDFExtractor(PDFExtractor):
    """Class for extracting markdown-formatted content from PDF."""

    BULLET_POINTS = "•◦▪▫●○"

    def __init__(self, pdf_path, output_path= config.get("OUTPUT_DIR", '.'), page_delimiter= config.get("PAGE_DELIMITER", ''), model_name: str | None= None):
        super().__init__(pdf_path)

        if model_name is None:
            # self.MODEL_NAME= "gemini-2.5-flash"
            self.MODEL_NAME= "Nanonets-OCR-s"
        else:
            self.MODEL_NAME= model_name

        if  "gemini" in self.MODEL_NAME:
            self.gclient = genai.Client(api_key= os.getenv("GEMINI_API_KEY", ''))
        elif "anonet" in self.MODEL_NAME:
            # self.nclient= Client("prithivMLmods/Multimodal-OCR2")

            # zerogpu public
            self.nclient= Client("deepak-mehta/ocr-simplify", hf_token= os.getenv('HF_TOKEN', ''))


        self.markdown_content= ""
        self.pdf_filename = Path(pdf_path).stem
        self.output_path= output_path

        output_filepath= f"{Path(self.output_path)}/{self.pdf_filename}.md"
        self.output_filepath= output_filepath

        self.page_delimiter= page_delimiter
        Path(output_path).mkdir(parents=True, exist_ok=True)



    def extract(self):
        try:
            markdown_content, markdown_pages = self.extract_markdown()
            self.save_markdown(markdown_content)
            self.markdown_content= markdown_content
            self.logger.info(
                f"Markdown content has been saved to {Path(self.output_path)}/{self.pdf_filename}.md"
            )
            return markdown_content, markdown_pages

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            self.logger.exception(traceback.format_exc())

            error_message= str(e).lower()
            if "GPU" in error_message and "quota" in error_message:
                return "GPU quota error", []
            return "", []


    def image_ocr(self, pil_image, img_bytes, max_new_tokens: int | None = None, prompt: str | None= None):
        if prompt is None:
            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
        if max_new_tokens is None:
            max_new_tokens= 4096

        w, h= pil_image.size
        if w < 200 or h < 50:
            return "<img> A small image </img>"

        model_name= self.MODEL_NAME.lower()
        if 'gemini' in model_name:

            image_format = pil_image.format
            dummy_filename = f"dummy.{image_format.lower()}"
            mime_type, _ = mimetypes.guess_type(dummy_filename)
            response=  self.gclient.models.generate_content(
                model= self.MODEL_NAME,
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes.getvalue(),
                        mime_type= mime_type
                    ),
                    prompt
                ]
            )
            # print("response :", response)
            return response.text
        elif 'nanonet' in model_name:

            result= ""
            try:
                with tempfile.NamedTemporaryFile(suffix=f'.{pil_image.format.lower()}', mode= 'w') as temp_file:
                    pil_image.save(temp_file.name)
                    print("file name: ", temp_file.name)
                    gr_image= handle_file(temp_file.name)
                    print("gr image : ", gr_image)
                    result = self.nclient.predict(
                        # model_name="Nanonets-OCR-s",
                        # text= prompt,
                        gr_image,
                        # max_new_tokens=max_new_tokens,
                        # temperature=0.6,
                        # top_p=0.9,
                        # top_k=50,
                        # repetition_penalty=1.2,

                        # prithiv model
                        # api_name="/generate_image"

                        max_new_tokens,

                        # spaces zerogpu
                        api_name="/predict"
                    )
                    print("ocr'd: ", result[:100] + "...")
            except Exception as e:
                print("Error during nanonet inference", e)
                error_message = str(e)
                if "You have exceeded your Pro GPU quota" in error_message:
                    # print("\n\n\nFALLING BACK TO TESS\n\n\n")
                    # return pytesseract.image_to_string(pil_image)
                    raise e

            
            return result
        else:
            return pytesseract.image_to_string(pil_image)



    def extract_markdown(self):
            """
            Extracts all possible content from a PDF, prioritizing searchable text,
            then OCR for embedded images, and finally full-page OCR for scanned pages.
            Avoids redundant OCR where possible.

            Returns:
                tuple: A tuple containing:
                    - str: The concatenated markdown content of all pages.
                    - list: A list of strings, where each string is the comprehensive markdown
                            for a corresponding page.
            """
            all_pages_markdown = []
            full_document_markdown = [] # Changed to list of lines/blocks to handle insertions better

            try:
                doc = fitz.open(self.pdf_path)
                self.logger.info(f"Opened PDF: {self.pdf_path}")

                tables = self.extract_tables()
                table_index = 0

                # State variables for process_text_block that might need to persist across blocks
                # Re-initialize for each new document, but allow state management within process_text_block for lines
                list_counter = 0
                in_code_block = False
                code_block_content = ""
                code_block_lang = None
                prev_line = ""

                for page_num, page in enumerate(doc):
                    current_page_markdown_blocks = [] # Collect markdown blocks for the current page
                    page_has_searchable_text = False
                    # page_has_embedded_images = False

                    self.logger.info(f"\nProcessing page {page_num + 1}...")

                    blocks = page.get_text('dict')['blocks']
                    page_height = page.rect.height
                    links = self.extract_links(page)

                    # Phase 1: Process text blocks and embedded image blocks
                    for block_num, block in enumerate(blocks):
                        if block['type'] == 0:  # Text block
                            page_has_searchable_text = True
                            processed_text = self.process_text_block(
                                block,
                                page_height,
                                links,
                                list_counter,
                                in_code_block,
                                code_block_content,
                                code_block_lang,
                                prev_line,
                            )
                            if processed_text.strip():
                                current_page_markdown_blocks.append(processed_text)

                        elif block['type'] == 1:  # Image block
                            page_has_embedded_images = True
                            self.logger.info(f"  Found embedded image block (Page {page_num+1}, Block {block_num+1})")
                            img_data = block['image']

                            try:
                                image_bytes= io.BytesIO(img_data)
                                pil_image = Image.open(image_bytes)
                                ocr_text_from_block_image = self.image_ocr(
                                    pil_image, image_bytes, max_new_tokens=15000
                                )

                                if ocr_text_from_block_image.strip():
                                    self.logger.info("    OCR found text in embedded image block.")
                                    current_page_markdown_blocks.append(f"\n\n\n{ocr_text_from_block_image.strip()}\n\n")
                                else:
                                    self.logger.info(f"    No OCR text from embedded image block. Adding generic placeholder.")
                                    current_page_markdown_blocks.append("\n\n![Image Placeholder](image_on_page_{page_num+1}_block_{block_num+1}.png)\n\n") # Consider saving images
                            except Exception as e:
                                self.logger.error(f"    Error processing embedded image block for OCR: {e}")
                                current_page_markdown_blocks.append("\n\n![Image Processing Error](error_on_page_{page_num+1}_block_{block_num+1}.png)\n\n")
                                error_message= str(e).lower()
                                if "GPU" in error_message and "quota" in error_message:
                                    raise e


                    # Insert tables at their approximate positions (after blocks are processed for the page)
                    # You might need more sophisticated logic here if table positions are granular
                    while (
                        table_index < len(tables)
                        and tables[table_index]["page"] == page.number
                    ):
                        current_page_markdown_blocks.append(
                            self.table_to_markdown(tables[table_index]["content"])
                        )
                        table_index += 1

                    # Phase 2: Full-page OCR if the page seems to be a scanned image or lacks sufficient searchable text
                    # We prioritize actual searchable text and embedded image OCR.
                    # Only if very little or no text was found, we resort to full-page OCR.
                    combined_current_page_text_length = len("".join(current_page_markdown_blocks).strip())

                    # A heuristic: if almost no searchable text and no significant OCR from embedded images
                    if not page_has_searchable_text and combined_current_page_text_length < 100: # Threshold for considering "minimal text"
                        self.logger.info(f"  Page {page_num + 1} appears to be a scanned image or has minimal text. Attempting full-page OCR.")
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                            img_bytes = pix.tobytes("png")
                            image_bytestream= io.BytesIO(img_bytes)
                            pil_image = Image.open(image_bytestream)

                            ocr_text_from_page = self.image_ocr(
                                pil_image, image_bytestream, max_new_tokens=15000
                            )

                            if ocr_text_from_page.strip():
                                self.logger.info(f"  Successfully extracted text via full-page OCR for page {page_num + 1}.")
                                # If full-page OCR yields significant content and other methods didn't,
                                # replace or augment. Here, we'll replace to avoid double-counting if it's primarily scanned.
                                # You might choose to append if you want to combine (e.g., if there's header text + scanned body)
                                if combined_current_page_text_length < 50: # If almost nothing was found before, replace
                                    current_page_markdown_blocks = [ocr_text_from_page.strip()]
                                else: # Otherwise, augment (append)
                                    current_page_markdown_blocks.append(f"\n\n\n{ocr_text_from_page.strip()}\n\n")
                            else:
                                self.logger.info(f"  Full-page OCR yielded no text for page {page_num+1}.")
                        except Exception as e:
                            self.logger.error(f"  Error during full-page OCR on page {page_num+1}: {e}")
                            error_message= str(e).lower()
                            if "GPU" in error_message and "quota" in error_message:
                                raise e
                    else:
                        self.logger.info(f"  Page {page_num + 1} has sufficient searchable text or embedded image OCR; skipping full-page OCR.")

                    # Join collected markdown blocks for the current page
                    final_page_markdown = "\n".join(filter(None, current_page_markdown_blocks)).strip()
                    all_pages_markdown.append(self.post_process_markdown(final_page_markdown))
                    full_document_markdown.append(self.post_process_markdown(final_page_markdown))
                    full_document_markdown.append(self.page_delimiter)


                    self.logger.info(f"  Comprehensive text for page {page_num + 1} (first 200 chars):\n{final_page_markdown[:200]}...")
                    print(f"\n--- Page {page_num+1} Done ---\n")
                    print(final_page_markdown[:500]) # Print first 500 chars of page markdown

                doc.close()
                return "".join(full_document_markdown), all_pages_markdown

            except fitz.FileNotFoundError:
                self.logger.error(f"PDF file not found: {self.pdf_path}")
                return "", []
            except Exception as e:
                self.logger.critical(f"An unexpected error occurred during markdown extraction: {e}")
                self.logger.exception(traceback.format_exc())

                error_message= str(e).lower()
                if "GPU" in error_message and "quota" in error_message:
                    return "GPU quota error", []
                else:
                    return "", []

    def extract_tables(self):
        """Extract tables from PDF using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    if len(page_tables) > 128:
                        continue
                    for table in page_tables:
                        tables.append({"page": page_number, "content": table})
            self.logger.info(f"Extracted {len(tables)} tables from the PDF.")
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
            self.logger.exception(traceback.format_exc())
        return tables

    def table_to_markdown(self, table):
        """Convert a table to markdown format."""
        if not table:
            return ""

        try:
            table = [
                ["" if cell is None else str(cell).strip() for cell in row]
                for row in table
            ]
            col_widths = [max(len(cell) for cell in col) for col in zip(*table)]

            markdown = ""
            for i, row in enumerate(table):
                formatted_row = [
                    cell.ljust(col_widths[j]) for j, cell in enumerate(row)
                ]
                markdown += "| " + " | ".join(formatted_row) + " |\n"

                if i == 0:
                    markdown += (
                        "|"
                        + "|".join(["-" * (width + 2) for width in col_widths])
                        + "|\n"
                    )

            return markdown
        except Exception as e:
            self.logger.error(f"Error converting table to markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def perform_ocr(self, image, image_bytes):
        """Perform OCR on the given image."""
        try:
            # ocr_result = pytesseract.image_to_string(
            #     image
            # )
            ocr_result= self.image_ocr(image, image_bytes, max_new_tokens=15000)


            return ocr_result.strip()
        except Exception as e:
            self.logger.error(f"Error performing OCR: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def caption_image(self, image, image_bytes):
        """Generate a caption for the given image."""
        try:
            ocr_text = self.perform_ocr(image, image_bytes)
            if ocr_text:
                return ocr_text

            # Convert image to RGB if it's not already
            if image.mode != "RGB":
                image = image.convert("RGB")

            caption= self.image_ocr(image, image_bytes, max_new_tokens=15000, prompt= "Write a caption for this image")
            return caption

        except Exception as e:
            self.logger.error(f"Error captioning image: {e}")
            self.logger.exception(traceback.format_exc())
            error_message= str(e)
            if "GPU" in error_message and "quota" in error_message:
                raise e
            return ""

    def clean_text(self, text):
        """Clean the given text by removing extra spaces."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def apply_formatting(self, text, flags):
        """Apply markdown formatting to the given text based on flags."""
        text = text.strip()
        if not text:
            return text

        is_bold = flags & 2**4
        is_italic = flags & 2**1
        is_monospace = flags & 2**3
        is_superscript = flags & 2**0
        is_subscript = flags & 2**5

        if is_monospace:
            text = f"`{text}`"
        elif is_superscript and not bool(re.search(r"\s+", text)):
            text = f"^{text}^"
        elif is_subscript and not bool(re.search(r"\s+", text)):
            text = f"~{text}~"

        if is_bold and is_italic:
            text = f"***{text}***"
        elif is_bold:
            text = f"**{text}**"
        elif is_italic:
            text = f"*{text}*"

        return f" {text} "

    def is_bullet_point(self, text):
        """Check if the given text is a bullet point."""
        return text.strip().startswith(tuple(self.BULLET_POINTS))

    def convert_bullet_to_markdown(self, text):
        """Convert a bullet point to markdown format."""
        text = re.sub(r"^\s*", "", text)
        return re.sub(f"^[{re.escape(self.BULLET_POINTS)}]\s*", "- ", text)

    def is_numbered_list_item(self, text):
        """Check if the given text is a numbered list item."""
        return bool(re.match(r"^\d+\s{0,3}[.)]", text.strip()))

    def convert_numbered_list_to_markdown(self, text, list_counter):
        """Convert a numbered list item to markdown format."""
        text = re.sub(r"^\s*", "", text)
        return re.sub(r"^\d+\s{0,3}[.)]", f"{list_counter}. ", text)

    def is_horizontal_line(self, text):
        """Check if the given text represents a horizontal line."""
        return bool(re.match(r"^[_-]+$", text.strip()))

    def extract_links(self, page):
        """Extract links from the given page."""
        links = []
        try:
            for link in page.get_links():
                if link["kind"] == 2:  # URI link
                    links.append({"rect": link["from"], "uri": link["uri"]})
            self.logger.info(f"Extracted {len(links)} links from the page.")
        except Exception as e:
            self.logger.error(f"Error extracting links: {e}")
            self.logger.exception(traceback.format_exc())
        return links

    def detect_code_block(self, prev_line, current_line):
        """Detect if the current line starts a code block."""
        patterns = {
            "python": [
                (
                    r"^(?:from|import)\s+\w+",
                    r"^(?:from|import|def|class|if|for|while|try|except|with)\s",
                ),
                (r"^(?:def|class)\s+\w+", r"^\s{4}"),
                (r"^\s{4}", r"^\s{4,}"),
            ],
            "javascript": [
                (
                    r"^(?:function|const|let|var)\s+\w+",
                    r"^(?:function|const|let|var|if|for|while|try|catch|class)\s",
                ),
                (r"^(?:if|for|while)\s*\(", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "html": [
                (
                    r"^<(!DOCTYPE|html|head|body|div|p|a|script|style)",
                    r"^<(!DOCTYPE|html|head|body|div|p|a|script|style)",
                ),
                (r"^<\w+.*>$", r"^\s{2,}<"),
                (r"^\s{2,}<", r"^\s{2,}<"),
            ],
            "shell": [
                (r"^(?:\$|\#)\s", r"^(?:\$|\#)\s"),
                (r"^[a-z_]+\s*=", r"^[a-z_]+\s*="),
            ],
            "bash": [
                (
                    r"^(?:#!/bin/bash|alias|export|source)\s",
                    r"^(?:#!/bin/bash|alias|export|source|echo|read|if|for|while|case|function)\s",
                ),
                (r"^(?:if|for|while|case|function)\s", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "cpp": [
                (
                    r"^#include\s*<",
                    r"^(?:#include|using|namespace|class|struct|enum|template|typedef)\s",
                ),
                (r"^(?:class|struct|enum)\s+\w+", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "java": [
                (
                    r"^(?:import|package)\s+\w+",
                    r"^(?:import|package|public|private|protected|class|interface|enum)\s",
                ),
                (r"^(?:public|private|protected)\s+class\s+\w+", r"^\s{4,}"),
                (r"^\s{4,}", r"^\s{4,}"),
            ],
            "json": [
                (r"^\s*{", r'^\s*["{[]'),
                (r'^\s*"', r'^\s*["}],?$'),
                (r"^\s*\[", r"^\s*[}\]],?$"),
            ],
        }

        for lang, pattern_pairs in patterns.items():
            for prev_pattern, curr_pattern in pattern_pairs:
                if re.match(prev_pattern, prev_line.strip()) and re.match(
                    curr_pattern, current_line.strip()
                ):
                    return lang

        return None

    def process_text_block(
        self,
        block,
        page_height,
        links,
        list_counter,
        in_code_block,
        code_block_content,
        code_block_lang,
        prev_line,
    ):
        """Process a text block and convert it to markdown."""
        try:
            block_rect = block["bbox"]
            if block_rect[1] < 50 or block_rect[3] > page_height - 50:
                return ""  # Skip headers and footers

            block_text = ""
            last_y1 = None
            last_font_size = None

            for line in block["lines"]:
                line_text = ""
                curr_font_size = [span["size"] for span in line["spans"]]

                for span in line["spans"]:
                    text = span["text"]
                    font_size = span["size"]
                    flags = span["flags"]
                    span_rect = span["bbox"]

                    if self.is_horizontal_line(text):
                        line_text += "\n---\n"
                        continue

                    text = self.clean_text(text)

                    if text.strip():
                        header_level = self.get_header_level(font_size)
                        if header_level > 0:
                            text = f"\n{'#' * header_level} {text}\n\n"

                        else:
                            is_list_item = self.is_bullet_point(
                                text
                            ) or self.is_numbered_list_item(text)

                            if is_list_item:
                                marker, content = re.split(
                                    r"(?<=^[•◦▪▫●○\d.)])\s*", text, 1
                                )
                                formatted_content = self.apply_formatting(
                                    content, flags
                                )
                                text = f"{marker} {formatted_content}"
                            else:
                                text = self.apply_formatting(text, flags)

                    for link in links:
                        if fitz.Rect(span_rect).intersects(link["rect"]):
                            text = f"[{text.strip()}]({link['uri']})"
                            break

                    line_text += text

                if last_y1 is not None:
                    avg_last_font_size = (
                        sum(last_font_size) / len(last_font_size)
                        if last_font_size
                        else 0
                    )
                    avg_current_font_size = sum(curr_font_size) / len(curr_font_size)
                    font_size_changed = (
                        abs(avg_current_font_size - avg_last_font_size) > 1
                    )

                    if abs(line["bbox"][3] - last_y1) > 2 or font_size_changed:
                        block_text += "\n"

                block_text += self.clean_text(line_text) + " "
                last_font_size = curr_font_size
                last_y1 = line["bbox"][3]

            markdown_content = ""
            lines = block_text.split("\n")
            for i, line in enumerate(lines):
                clean_line = self.clean_text(line)

                if not in_code_block:
                    code_lang = self.detect_code_block(prev_line, clean_line)
                    if code_lang:
                        in_code_block = True
                        code_block_lang = code_lang
                        code_block_content = prev_line + "\n" + clean_line + "\n"
                        prev_line = clean_line
                        continue

                if in_code_block:
                    code_block_content += clean_line + "\n"
                    if (
                        i == len(lines) - 1
                        or self.detect_code_block(clean_line, lines[i + 1])
                        != code_block_lang
                    ):
                        markdown_content += (
                            f"```{code_block_lang}\n{code_block_content}```\n\n"
                        )
                        in_code_block = False
                        code_block_content = ""
                        code_block_lang = None
                else:
                    if self.is_bullet_point(clean_line):
                        markdown_content += "\n" + self.convert_bullet_to_markdown(
                            clean_line
                        )
                        list_counter = 0
                    elif self.is_numbered_list_item(clean_line):
                        list_counter += 1
                        markdown_content += (
                            "\n"
                            + self.convert_numbered_list_to_markdown(
                                clean_line, list_counter
                            )
                        )
                    else:
                        markdown_content += f"{clean_line}\n"
                        list_counter = 0

                prev_line = clean_line

            return markdown_content + "\n"
        except Exception as e:
            self.logger.error(f"Error processing text block: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def process_image_block(self, page, block):
        """Process an image block and convert it to markdown."""
        try:
            image_rect = block["bbox"]
            zoom_x = 2.0  # horizontal zoom
            zoom_y = 2.0  # vertical zoom
            mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
            pix = page.get_pixmap(clip=image_rect, matrix=mat, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if image.width < 20 or image.height < 20:
                return ""

            image_filename = (
                f"{self.pdf_filename}_image_{int(page.number)+1}_{block['number']}.png"
            )
            image_path = (
                Path(self.output_path) / image_filename
            )  # Convert to Path object
            image.save(image_path, "PNG", optimize=True, quality=95)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr)
            caption = self.caption_image(image, img_byte_arr)

            if not caption:
                caption = (
                    f"{self.pdf_filename}_image_{int(page.number)+1}_{block['number']}"
                )

            return f"![{caption}]({image_path})\n\n"  # image_path is now a Path object
        except Exception as e:
            self.logger.error(f"Error processing image block: {e}")
            self.logger.exception(traceback.format_exc())
            return ""


    def get_header_level(self, font_size):
        """Determine header level based on font size."""
        if font_size > 24:
            return 1
        elif font_size > 20:
            return 2
        elif font_size > 18:
            return 3
        elif font_size > 16:
            return 4
        elif font_size > 14:
            return 5
        elif font_size > 12:
            return 6
        else:
            return 0

    def post_process_markdown(self, markdown_content):
        """Post-process the markdown content."""
        try:
            markdown_content = re.sub(
                r"\n{3,}", "\n\n", markdown_content
            )  # Remove excessive newlines
            markdown_content = re.sub(
                r"(\d+)\s*\n", "", markdown_content
            )  # Remove page numbers
            markdown_content = re.sub(
                r" +", " ", markdown_content
            )  # Remove multiple spaces
            markdown_content = re.sub(
                r"\s*(---\n)+", "\n\n---\n", markdown_content
            )  # Remove duplicate horizontal lines

            def remove_middle_headers(match):
                line = match.group(0)
                # Keep the initial header and remove all subsequent '#' characters
                return re.sub(
                    r"(^#{1,6}\s).*?(?=\n)",
                    lambda m: m.group(1)
                    + re.sub(r"#", "", m.group(0)[len(m.group(1)) :]),
                    line,
                )

            markdown_content = re.sub(
                r"^#{1,6}\s.*\n",
                remove_middle_headers,
                markdown_content,
                flags=re.MULTILINE,
            )  # Remove headers in the middle of lines
            return markdown_content
        except Exception as e:
            self.logger.error(f"Error post-processing markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return markdown_content

    def save_markdown(self, markdown_content):
        """Save the markdown content to a file."""
        try:
            os.makedirs(Path(self.output_path), exist_ok=True)
            with open(
                self.output_filepath,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(markdown_content)
            self.logger.info("Markdown content saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving markdown content: {e}")
            self.logger.exception(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(
        description="Extract markdown-formatted content from a PDF file."
    )
    parser.add_argument("--pdf_path", help="Path to the input PDF file", required=True)
    args = parser.parse_args()

    extractor = MarkdownPDFExtractor(args.pdf_path)
    markdown_pages = extractor.extract()
    return markdown_pages


if __name__ == "__main__":
    main()
