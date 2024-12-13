from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
import os
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import pytesseract
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSEREACT_EXE')
from bs4 import BeautifulSoup
from embeddings import *

class BaseExtractor():

    def __init__(self):
        pass

class PDFExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.embeddings = HFEmbedding('keepitreal/vietnamese-sbert')

    def load(self, pdf_path):
        result = {
            "org_path": pdf_path,
            "content": "",
        }

        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            pdf_text = ""

            for page in doc:
                pdf_text += page.get_text()

            if not pdf_text.strip():
                # If the PDF has no text content, use OCR
                images = convert_from_path(pdf_path)
                ocr_text = ""
                for image in images:
                    ocr_text += pytesseract.image_to_string(image, lang="vie")
                pdf_text = ocr_text

            if not pdf_text.strip():
                # If still no content, return an error
                raise ValueError("No content found in the PDF.")

            # Update result dictionary
            result["content"] = pdf_text

            # Create a Document instance
            document = Document(
                page_content=pdf_text,
                metadata={"source": pdf_path}
            )
            return document

        except Exception as e:
            error_message = str(e)
            print(f"Error processing {os.path.basename(pdf_path)}: {error_message}")
            return Document(
                page_content="",
                metadata={"source": pdf_path, "error": error_message}
            )

    def loads(self, pdfs_dir):
        documents = []
        for pdf_file in os.listdir(pdfs_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdfs_dir, pdf_file)
                documents.append(self.load(pdf_path))
        return documents

class HTMLExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.embeddings = HFEmbedding('keepitreal/vietnamese-sbert')

    def load(self, html_path):
        try:
            with open(html_path, "r", encoding="utf-8") as file:
                html_content = file.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract the title
            title_div = soup.find("div", class_="col-md-12")
            title = title_div.get_text(strip=True) if title_div else "No title found"

            # Extract the content
            content_div = soup.find("div", class_="col-md-10 col-md-offset-1")
            content = content_div.get_text(strip=True) if content_div else "No content found"

            # Replace hyperlinks with URLs in content
            if content_div:
                for a_tag in content_div.find_all("a", href=True):
                    a_tag.replace_with(a_tag['href'])

            text = title + '\n' + content

            # Create a Document instance
            document = Document(
                page_content=text,
                metadata={"source": html_path}
            )
            return document

        except Exception as e:
            error_message = str(e)
            print(f"Error processing {os.path.basename(html_path)}: {error_message}")
            return Document(
                page_content="",
                metadata={"source": html_path, "error": error_message}
            )

    def loads(self, htmls_dir):
        documents = []
        for html_file in os.listdir(htmls_dir):
            if html_file.endswith(".html"):
                html_path = os.path.join(htmls_dir, html_file)
                documents.append(self.load(html_path))
        return documents


class TXTExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.embeddings = HFEmbedding('keepitreal/vietnamese-sbert')
        # self.text_splitter = SemanticChunker(self.embeddings.core, breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90)

    def loads(self, txts_dir):
        loader = DirectoryLoader(txts_dir, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        return documents

    def load(self, txt_path):
        loader = TextLoader(txt_path, glob="*.txt")
        documents = loader.load()
        
        return documents

class XLSXExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.embeddings = HFEmbedding('keepitreal/vietnamese-sbert')

    def load(self, xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
            content = df.to_string(index=False)

            if not content.strip():
                raise ValueError("No content found in the XLSX file.")

            # Create a Document instance
            document = Document(
                page_content=content,
                metadata={"source": xlsx_path}
            )
            return document

        except Exception as e:
            error_message = str(e)
            print(f"Error processing {os.path.basename(xlsx_path)}: {error_message}")
            return Document(
                page_content="",
                metadata={"source": xlsx_path, "error": error_message}
            )

    def loads(self, xlsx_dir):
        documents = []
        for xlsx_file in os.listdir(xlsx_dir):
            if xlsx_file.endswith(".xlsx"):
                xlsx_path = os.path.join(xlsx_dir, xlsx_file)
                documents.append(self.load(xlsx_path))
        return documents

class CSVExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.embeddings = HFEmbedding('keepitreal/vietnamese-sbert')

    def load(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            content = df.to_string(index=False)

            if not content.strip():
                raise ValueError("No content found in the CSV file.")

            # Create a Document instance
            document = Document(
                page_content=content,
                metadata={"source": csv_path}
            )
            return document

        except Exception as e:
            error_message = str(e)
            print(f"Error processing {os.path.basename(csv_path)}: {error_message}")
            return Document(
                page_content="",
                metadata={"source": csv_path, "error": error_message}
            )

    def loads(self, csv_dir):
        documents = []
        for csv_file in os.listdir(csv_dir):
            if csv_file.endswith(".csv"):
                csv_path = os.path.join(csv_dir, csv_file)
                documents.append(self.load(csv_path))
        return documents

