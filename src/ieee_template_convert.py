"""Convert a Strict-OOXML .docx (purl.oclc.org namespaces, unit-suffixed
measurements) to Transitional OOXML so python-docx can open it.

IEEE distributes conference-template-letter.docx saved as ISO 29500 Strict;
python-docx only speaks Transitional. Two mechanical differences matter here:

1. namespace URIs: http://purl.oclc.org/ooxml/... -> the corresponding
   http://schemas.openxmlformats.org/... URI;
2. measurements: Strict allows unit-suffixed values ("9pt", "612pt"); the
   Transitional schema wants bare integers in context-dependent units —
   half-points for run sizes, eighths of a point for border widths, and
   twentieths of a point (twips) everywhere else.

Booleans, content types, and part layout are shared between the two variants.
"""
from __future__ import annotations

import io
import re
import zipfile

NS_MAP = [
    # longest / most specific first
    ("http://purl.oclc.org/ooxml/officeDocument/relationships/extendedProperties",
     "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties"),
    ("http://purl.oclc.org/ooxml/officeDocument/relationships/",
     "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"),
    ("http://purl.oclc.org/ooxml/officeDocument/relationships",
     "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    ("http://purl.oclc.org/ooxml/officeDocument/math",
     "http://schemas.openxmlformats.org/officeDocument/2006/math"),
    ("http://purl.oclc.org/ooxml/officeDocument/extendedProperties",
     "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"),
    ("http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes",
     "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"),
    ("http://purl.oclc.org/ooxml/officeDocument/sharedTypes",
     "http://schemas.openxmlformats.org/officeDocument/2006/sharedTypes"),
    ("http://purl.oclc.org/ooxml/officeDocument/customXml",
     "http://schemas.openxmlformats.org/officeDocument/2006/customXml"),
    ("http://purl.oclc.org/ooxml/wordprocessingml/main",
     "http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
    ("http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing",
     "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
    ("http://purl.oclc.org/ooxml/drawingml/main",
     "http://schemas.openxmlformats.org/drawingml/2006/main"),
    ("http://purl.oclc.org/ooxml/drawingml/chart",
     "http://schemas.openxmlformats.org/drawingml/2006/chart"),
    ("http://purl.oclc.org/ooxml/drawingml/picture",
     "http://schemas.openxmlformats.org/drawingml/2006/picture"),
]

# attributes measured in half-points (run/font sizes)
HALF_POINT = re.compile(
    r'(<w:(?:sz|szCs|kern|position)\b[^>]*?w:val=")(-?[0-9.]+)pt(")')
# border widths: eighths of a point
BORDER = re.compile(
    r'(<w:(?:top|bottom|left|right|start|end|insideH|insideV|bar|between|bdr)\b'
    r'[^>]*?w:sz=")(-?[0-9.]+)pt(")')
# any remaining pt-suffixed attribute value: twips
TWIPS = re.compile(r'(="?)(-?[0-9.]+)pt(")')


def _scale(match, factor):
    lead, val, tail = match.group(1), float(match.group(2)), match.group(3)
    return f"{lead}{round(val * factor)}{tail}"


def convert_xml(xml: str) -> str:
    for old, new in NS_MAP:
        xml = xml.replace(old, new)
    xml = HALF_POINT.sub(lambda m: _scale(m, 2), xml)
    xml = BORDER.sub(lambda m: _scale(m, 8), xml)
    xml = TWIPS.sub(lambda m: _scale(m, 20), xml)
    return xml


def strict_to_transitional(src_path: str) -> io.BytesIO:
    """Return an in-memory Transitional copy of a Strict .docx."""
    out = io.BytesIO()
    with zipfile.ZipFile(src_path) as zin, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith((".xml", ".rels")):
                data = convert_xml(data.decode("utf-8")).encode("utf-8")
            zout.writestr(info, data)
    leftovers = []
    with zipfile.ZipFile(out) as z:
        for name in z.namelist():
            if name.endswith((".xml", ".rels")):
                txt = z.read(name).decode("utf-8")
                if "purl.oclc.org" in txt or re.search(r'="[0-9.]+pt"', txt):
                    leftovers.append(name)
    if leftovers:
        raise RuntimeError(f"strict remnants in: {leftovers}")
    out.seek(0)
    return out
