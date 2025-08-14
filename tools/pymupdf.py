import logging
import os
import io
from collections.abc import Generator
from typing import Any, Optional

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)


class PymupdfTool(Tool):
    """
    A tool for extracting text from PDF files using PyMuPDF.

    This fork is INPUT-TOLERANT:
      - Accepts an Array[Object] (generic dicts) OR Dify File objects.
      - Each item may provide:
          - `blob` (bytes)  OR
          - `url` / `remote_url` (absolute URL to a PDF, including signed links)
        Optional fields used for nicer output:
          - `filename`, `mime_type`, `size`
    """

    # ------------- helpers -------------

    @staticmethod
    def _import_fitz():
        """
        Support both import names used by PyMuPDF across environments.
        """
        try:
            import pymupdf as fitz  # modern name
            return fitz
        except Exception:
            import fitz  # fallback
            return fitz

    @staticmethod
    def _first(*vals: Optional[str]) -> Optional[str]:
        for v in vals:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    @staticmethod
    def _infer_filename(item: dict, url: Optional[str]) -> str:
        # Prefer explicit filename if provided
        name = item.get("filename")
        if isinstance(name, str) and name.strip():
            return name.strip()

        # Derive from URL path if available
        if url:
            base = url.split("?", 1)[0]  # strip query
            leaf = os.path.basename(base) or "document.pdf"
            return leaf

        # Fallback
        return "document.pdf"

    @staticmethod
    def _ensure_pdf_bytes(item: Any, timeout: int = 30) -> bytes:
        """
        Return PDF bytes from:
          - Dify File (has .blob) or dict with 'blob'
          - dict with 'url' or 'remote_url'
        """
        # Case 1: looks like Dify File with .blob
        blob = getattr(item, "blob", None)
        if blob:
            return blob

        # Case 2: dict with 'blob'
        if isinstance(item, dict) and isinstance(item.get("blob"), (bytes, bytearray)):
            return bytes(item["blob"])

        # Case 3: fetch via URL/remote_url
        url = None
        if isinstance(item, dict):
            url = PymupdfTool._first(item.get("url"), item.get("remote_url"))
        if not url or not isinstance(url, str):
            raise ValueError("No usable PDF source found (need 'blob' or absolute 'url'/'remote_url').")

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()

        # Optional: quick content-type sanity check (do not hard-fail on misreported servers)
        ctype = resp.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower():
            logger.debug(f"Content-Type not indicating PDF: {ctype} (continuing anyway)")

        return resp.content

    @staticmethod
    def _extract_text_from_pdf_bytes(pdf_bytes: bytes, fitz_module) -> list[dict]:
        """
        Return a list of page dicts: { 'text': str, 'metadata': { 'page': int, 'file_name': str } }
        """
        pages = []
        # Use a BytesIO stream for PyMuPDF
        stream = io.BytesIO(pdf_bytes)
        doc = None
        try:
            doc = fitz_module.open(stream=stream, filetype="pdf")
            for i in range(doc.page_count):
                page = doc.load_page(i)
                text = page.get_text("text")
                pages.append({"text": text})
        finally:
            if doc is not None:
                doc.close()
            stream.close()
        return pages

    # ------------- main entry -------------

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        files = tool_parameters.get("files")
        if not files or not isinstance(files, list):
            yield self.create_text_message("No files provided. Please supply an array of PDF objects or Dify Files.")
            return

        # Import PyMuPDF
        try:
            fitz_module = self._import_fitz()
        except Exception as e:
            msg = f"Error: PyMuPDF library not installed or failed to import. {e}"
            logger.error(msg)
            yield self.create_text_message(msg)
            return

        # Process each input
        for item in files:
            try:
                # Resolve URL for logging/filename purposes if present
                url = None
                if isinstance(item, dict):
                    url = self._first(item.get("url"), item.get("remote_url"))
                elif hasattr(item, "url"):
                    url = getattr(item, "url")

                filename = self._infer_filename(item if isinstance(item, dict) else {}, url)
                logger.info(f"Processing PDF: {filename}")

                # Get bytes
                pdf_bytes = self._ensure_pdf_bytes(item)

                # Extract text (per page)
                page_dicts = self._extract_text_from_pdf_bytes(pdf_bytes, fitz_module)

                # Attach metadata and construct outputs
                for idx, pd in enumerate(page_dicts, start=1):
                    pd["metadata"] = {"page": idx, "file_name": filename}

                # Join for human-readable text stream
                joined_text = "\n\n---PAGE BREAK---\n\n".join(p["text"] for p in page_dicts)

                # 1) Human-readable text
                yield self.create_text_message(joined_text)

                # 2) Structured JSON
                yield self.create_json_message({filename: page_dicts})

                # 3) Raw text blob
                yield self.create_blob_message(
                    joined_text.encode("utf-8", errors="ignore"),
                    meta={"mime_type": "text/plain"},
                )

            except Exception as e:
                err = f"Error processing file: {e}"
                logger.exception(err)
                # Emit helpful error info for both text and JSON channels
                yield self.create_text_message(err)
                fname = "unknown.pdf"
                if isinstance(item, dict):
                    fname = self._infer_filename(item, self._first(item.get("url"), item.get("remote_url")))
                yield self.create_json_message({fname: {"error": str(e)}})
