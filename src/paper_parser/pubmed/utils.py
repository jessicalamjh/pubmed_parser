"""Utility functions for PubMed/PMC XML parsing."""

import os
import re
from collections.abc import Mapping

from lxml import etree

from paper_parser.shared.schemas import DEFAULT_REF_TYPE


# Element tags whose entire subtree is "noise" we don't want in the reading
# text: footnote bodies, author notes, table footers. These are usually in
# <back> or inside <table-wrap>, but can also appear inline inside a <p>.
_NOISE_TAGS: frozenset[str] = frozenset({
    "fn",
    "fn-group",
    "author-notes",
    "table-wrap-foot",
})

# xref ref-type values that mark footnote-like citations (not real refs to
# figures/tables/bib entries). Stripping these removes the inline marker
# (e.g. a superscript "1") so it doesn't confuse ref-localization.
_NOISE_XREF_REF_TYPES: frozenset[str] = frozenset({
    "fn",
    "table-fn",
    "author-notes",
})

# Tail between two sibling bibliography xrefs when the author wrote a range (e.g. 1–5).
_BIBR_RANGE_TAIL_RE = re.compile(r"^[\s\-–—‐]+$")

# Separator inserted between every citation mark when a range is expanded (canonical ", ").
_BIBR_EXPAND_SEP = ", "


def _local_tag(el: etree._Element) -> str:
    """Return an element's tag without its XML namespace prefix.

    lxml reports tags as ``{namespace}name`` when a namespace is present.
    For JATS PMC XML we only care about the local name, so this helper
    strips the namespace part if any.
    """
    tag = el.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag or ""


def _find_child(parent: etree._Element, local_name: str) -> etree._Element | None:
    """Return the first direct child of ``parent`` with the given local tag.

    This is a namespace-agnostic equivalent of ``parent.find(local_name)``:
    useful for JATS where elements may or may not carry a namespace prefix
    depending on the source document.
    """
    for child in parent:
        if _local_tag(child) == local_name:
            return child
    return None


def build_xml_tree(x: str | etree._ElementTree) -> etree._ElementTree:
    """Parse a PMC JATS XML source into an ``ElementTree``.

    Accepts a filepath, an XML string, or an already-parsed
    ``ElementTree`` (returned as-is). Ensures the ``xlink`` namespace is
    declared on the root so downstream XPath/attribute lookups work
    regardless of the source document.
    """
    if isinstance(x, etree._ElementTree):
        return x

    if not isinstance(x, str) or not x.strip():
        raise ValueError(
            "Input must be a non-empty string containing a filepath or XML content"
        )

    if os.path.isfile(x):
        with open(x, "r") as f:
            x_string = f.read()
    else:
        x_string = x

    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(x_string.encode("utf-8"), parser)
    if "xlink" not in root.nsmap:
        root.set(
            "{http://www.w3.org/2000/xmlns/}xlink",
            "http://www.w3.org/1999/xlink",
        )
    return etree.ElementTree(root)


def strip_noise(root: etree._Element) -> None:
    """Remove footnote-like subtrees from ``root`` in place, preserving tails.

    - ``<fn>``, ``<fn-group>``, ``<author-notes>``, ``<table-wrap-foot>`` are
      dropped entirely (body + any marker text).
    - ``<xref>`` whose ``ref-type`` is footnote-like (``fn``, ``table-fn``,
      ``author-notes``) is dropped; the numeric marker inside (e.g.
      ``<sup>1</sup>``) goes with it.

    The removed element's tail text is merged back into the preceding
    sibling's tail, or into the parent's text if there is no preceding
    sibling, so surrounding prose stays intact.
    """
    to_remove: list[etree._Element] = []
    for el in root.iter():
        tag = _local_tag(el)
        if tag in _NOISE_TAGS:
            to_remove.append(el)
        elif tag == "xref":
            rt = (el.attrib.get("ref-type") or "").strip().lower()
            if rt in _NOISE_XREF_REF_TYPES:
                to_remove.append(el)

    for el in to_remove:
        parent = el.getparent()
        if parent is None:
            # Already detached because an ancestor was removed earlier.
            continue
        tail = el.tail
        if tail:
            idx = parent.index(el)
            if idx > 0:
                sib = parent[idx - 1]
                sib.tail = (sib.tail or "") + tail
            else:
                parent.text = (parent.text or "") + tail
        parent.remove(el)


def _is_bibr_xref(el: etree._Element) -> bool:
    if _local_tag(el) != "xref":
        return False
    rt = (el.attrib.get("ref-type") or "").strip().lower()
    return rt == "bibr"


def _citation_mark_int(el: etree._Element) -> int | None:
    raw = stringify(el, delimiter="", recurse=True).strip()
    if re.fullmatch(r"\d+", raw):
        return int(raw)
    return None


def _bibr_xref_like(template: etree._Element, rid: str, text: str) -> etree._Element:
    el = etree.Element(template.tag)
    for k, v in template.attrib.items():
        el.set(k, v)
    el.set("rid", rid)
    el.text = text
    el.tail = None
    return el


def expand_bibr_citation_ranges(
    root: etree._Element, bibliography: Mapping[str, object]
) -> None:
    """Insert bibliography ``<xref>`` nodes for implied refs inside a numeric range.

    Only handles **two sibling** ``bibr`` xrefs (first and last marker), with a dash-like
    tail between them. Inferred ``rid`` values for inserted nodes must appear as keys in
    ``bibliography`` (from ``back/ref-list/ref/@id``); otherwise the range is left unchanged.
    """
    label2ref_id = {}
    for rid, ref in bibliography.items():
        if rid not in label2ref_id and ref.label:
            label2ref_id[ref.label] = rid

    if not label2ref_id:
        return

    for parent in list(root.iter()):
        if len(parent) < 2:
            continue
        i = len(parent) - 2
        while i >= 0:
            a = parent[i]
            b = parent[i + 1]
            if not (_is_bibr_xref(a) and _is_bibr_xref(b)):
                i -= 1
                continue

            tail = a.tail or ""
            if not _BIBR_RANGE_TAIL_RE.fullmatch(tail):
                i -= 1
                continue

            n = _citation_mark_int(a)
            m_hi = _citation_mark_int(b)
            if n is None or m_hi is None or m_hi <= n + 1:
                i -= 1
                continue

            label_a = bibliography[a.attrib.get("rid")].label
            label_b = bibliography[b.attrib.get("rid")].label
            if label_a is None or label_b is None:
                print(a.attrib.get("rid"), b.attrib.get("rid"))
                print(label2ref_id)
            if not label_a.isdigit() or not label_b.isdigit():
                i -= 1
                continue

            int_a = int(label_a)
            int_b = int(label_b)
            if int_b - int_a <= 1:
                i -= 1
                continue

            inferred_labels = [str(x) for x in range(int_a + 1, int_b)]
            if not inferred_labels:
                i -= 1
                continue
            
            sep = _BIBR_EXPAND_SEP
            a.tail = sep
            insert_at = parent.index(b)
            for off, inferred_label in enumerate(inferred_labels):
                inferred_rid = label2ref_id.get(inferred_label)
                if not inferred_rid:
                    print(f"Unable to expand '{label_a}{tail}{label_b}' with {inferred_label=}")
                    continue
                node = _bibr_xref_like(a, inferred_rid, inferred_label)
                node.tail = sep
                parent.insert(insert_at + off, node)
            i -= 1


def get_xml_root(x: str | etree._ElementTree) -> etree._Element:
    """Get the root element of an XML tree, building one if needed."""
    return build_xml_tree(x).getroot()


def stringify(
    node: etree._Element, recurse: bool = True, delimiter: str | None = None
) -> str | list[str]:
    """Convert XML node to string or list of strings.

    Args:
        node: XML element to stringify
        recurse: Whether to recursively process child nodes
        delimiter: If provided, join parts with this delimiter (returns str).
                   If None, returns list of strings.

    Returns:
        String if delimiter is provided, otherwise list of strings
    """
    if node is None:
        return ""

    parts = []
    if node.text:
        parts.append(node.text)
    if recurse:
        for child in node:
            parts.extend(stringify(child))
            if child.tail:
                parts.append(child.tail)
    if isinstance(delimiter, str):
        return delimiter.join(parts)
    return parts


def get_xml_lang(node: etree._Element) -> str | None:
    """Return the language of an XML node (``xml:lang`` or ``lang``)."""
    return node.get("xml:lang") or node.get("lang")


MONTH_NAME_TO_NUMBER_MAP: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def convert_month_name_to_number(month_name: str) -> int:
    """Convert a case-insensitive month name to its 1-12 number."""
    return MONTH_NAME_TO_NUMBER_MAP[month_name.lower()]


PMCOA_XREF_REF_TYPE_MAP: dict[str, str] = {
    "bibr": "bib_entry",
    "fig": "figure",
    "table": "figure",
    "sec": "section",
}


def normalize_pmcoa_ref_type(
    ref_type: str | None, default_ref_type: str = DEFAULT_REF_TYPE
) -> str | None:
    """Normalize a PMC OA xref ref-type attribute to a schema-compatible value."""
    rt = (ref_type or "").strip().lower()
    if not rt:
        return None
    return PMCOA_XREF_REF_TYPE_MAP.get(rt, default_ref_type)
