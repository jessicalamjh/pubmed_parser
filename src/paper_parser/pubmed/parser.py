"""PMC XML parser for extracting paper metadata and content, following the Paper schema.

Design
------
Parsing is split into two phases so that the XML tree walk stays pure and
tokenization happens exactly once per paper:

1. Tree walk (``PaperParser._build_*``): builds the ``Paragraph`` / ``Figure``
   / ``Section`` objects with placeholder sentence contents, and records one
   ``_ParagraphJob`` or ``_CaptionJob`` per text that still needs to be split
   into sentences.

2. Tokenize-and-fill (``PaperParser._fill_jobs``): runs a single
   ``tokenizer.tokenize_batch(...)`` over every text collected in phase 1,
   then mutates each placeholder object's ``contents`` in place.

The module-level frontmatter helpers (ids, title, dates, bibliography, etc.)
do not depend on the tokenizer and remain free functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from paper_parser.shared.schemas import (
    BibEntry,
    Content,
    Date,
    Figure,
    Paper,
    PaperId,
    Paragraph,
    Ref,
    Section,
    Sentence,
)
from paper_parser.shared.sentence_tokenizer import SentenceTokenizer
from paper_parser.pubmed.pmc_id_map import PmcIdMap
from paper_parser.pubmed.utils import (
    _find_child,
    _local_tag,
    build_xml_tree,
    expand_bibr_citation_ranges,
    strip_noise,
    convert_month_name_to_number,
    get_xml_lang,
    get_xml_root,
    normalize_pmcoa_ref_type,
    stringify,
)

logger = logging.getLogger(__name__)


# ========= FRONTMATTER EXTRACTION =========
def extract_paper_ids(x: str | etree._ElementTree) -> list[PaperId]:
    root = get_xml_root(x)
    paper_ids: list[PaperId] = []
    for node in root.xpath("front/article-meta/article-id"):
        try:
            pub_id_type = (node.attrib.get("pub-id-type") or "").strip().lower()
            pub_id = (node.text or "").strip()

            if pub_id_type in ("pmc", "pmcid"):
                digits = "".join(c for c in pub_id if c.isdigit())
                paper_id = PaperId(id_type="pmc", value=f"PMC{digits}")
            elif pub_id_type == "pmid":
                digits = "".join(c for c in pub_id if c.isdigit())
                paper_id = PaperId(id_type="pmid", value=digits)
            elif pub_id_type == "doi":
                parts = "".join(c for c in pub_id if c.isalnum() or c in "./_-;()")
                paper_id = PaperId(id_type="doi", value=parts)
            else:
                continue
            paper_ids.append(paper_id)
        except Exception as e:
            logger.warning(f"Failed to extract paper id from {node=}; error={e}")
    return paper_ids


def extract_pmc_id_from_path(path: str | Path) -> PaperId | None:
    stem = Path(path).stem
    # filepath sometimes has a version number suffix
    if "." in stem:
        stem = stem.split(".")[0]
    digits = "".join(c for c in stem if c.isdigit())
    return PaperId(id_type="pmc", value=f"PMC{digits}") if digits else None


def extract_paper_type(x: str | etree._ElementTree) -> str | None:
    """Extract paper type from XML (e.g. research-article)."""
    root = get_xml_root(x)
    article_type = root.get("article-type", None)
    if isinstance(article_type, str):
        return " ".join(article_type.split())
    return None


def extract_subjects(x: str | etree._ElementTree) -> list[str]:
    """Extract article subjects/categories."""
    root = get_xml_root(x)
    out: list[str] = []
    for node in root.xpath("front/article-meta/article-categories//subj-group/subject"):
        subject = stringify(node, delimiter=" ")
        if subject:
            out.append(subject)
    return out


def extract_title_sentence(x: str | etree._ElementTree) -> Sentence:
    """Extract article title as a Sentence with content_id ['title', 0]."""
    root = get_xml_root(x)
    nodes = root.xpath("front/article-meta//title-group/article-title")
    text = stringify(nodes[0], delimiter=" ") if nodes else ""
    text = " ".join(text.split())
    return Sentence(content_id=["title", 0], text=text or "")


def extract_pub_date(x: str | etree._ElementTree) -> Date | None:
    """Extract publication date as shared Date (year/month/day int | None)."""
    root = get_xml_root(x)

    def score_node(node: etree._Element) -> bool:
        return (
            node.attrib.get("pub-type") == "epub"
            or (
                node.attrib.get("publication-format") == "electronic"
                and node.attrib.get("date-type") == "pub"
            )
        )

    nodes = root.xpath("front/article-meta/pub-date")
    if not nodes:
        return None
    nodes.sort(key=score_node, reverse=True)
    best = nodes[0]

    def _int_or_none(parent: etree._Element, name: str) -> int | None:
        child = _find_child(parent, name)
        if child is None or child.text is None:
            return None
        raw = (child.text or "").strip()
        if not raw:
            return None
        if name == "month" and raw.isalpha():
            try:
                return convert_month_name_to_number(raw)
            except KeyError:
                return None
        try:
            return int(raw)
        except ValueError:
            return None

    year = _int_or_none(best, "year")
    month = _int_or_none(best, "month")
    day = _int_or_none(best, "day")
    return Date(year=year, month=month, day=day)


def extract_bibliography(x: str | etree._ElementTree) -> dict[str, BibEntry]:
    root = get_xml_root(x)
    nodes = root.xpath("back/ref-list/ref")

    bibliography: dict[str, BibEntry] = {}
    for node in nodes:
        rid = node.attrib.get("id")
        if not rid:
            continue
        if rid in bibliography:
            logger.warning(f"Skipping duplicate rid: {rid}")
            continue

        all_paper_ids: list[PaperId] = []
        for pub_id_node in node.xpath("*/pub-id"):
            try:
                pub_id_type = (pub_id_node.attrib.get("pub-id-type") or "").strip().lower()
                pub_id = (pub_id_node.text or "").strip()

                if pub_id_type in ("pmc", "pmcid"):
                    digits = "".join(c for c in pub_id if c.isdigit())
                    paper_id = PaperId(id_type="pmc", value=f"PMC{digits}")
                elif pub_id_type == "pmid":
                    digits = "".join(c for c in pub_id if c.isdigit())
                    paper_id = PaperId(id_type="pmid", value=digits)
                elif pub_id_type == "doi":
                    parts = "".join(c for c in pub_id if c.isalnum() or c in "./_-;()")
                    paper_id = PaperId(id_type="doi", value=parts)
                else:
                    continue
                all_paper_ids.append(paper_id)
            except Exception as e:
                logger.warning(f"Failed to extract paper id from {pub_id_node=}; error={e}")

        label = None
        for child in node.getchildren():
            if child.tag == "label":
                label = child.text
        bibliography[rid] = BibEntry(rid=rid, all_paper_ids=all_paper_ids, label=label)
    return bibliography


# ========= CONTENT HELPERS (pure, no tokenizer needed) =========
def locate_refs_in_paragraph(
    p_el: etree._Element,
    paragraph_text_parts: list[str],
) -> list[Ref]:
    refs_for_paragraph: list[Ref] = []
    prev_part_idx = 0
    paragraph_text = " ".join(paragraph_text_parts)
    for ref_idx, ref in enumerate(p_el.xpath(".//xref")):
        # Find the first occurrence of ref.text in paragraph_text_parts,
        # ignoring parts already claimed by previous refs.
        try:
            if ref_idx == 0:
                this_part_idx = paragraph_text_parts.index(ref.text)
            else:
                this_part_idx = (
                    paragraph_text_parts[prev_part_idx + 1 :].index(ref.text)
                    + prev_part_idx
                    + 1
                )
        except ValueError:
            continue

        ref_start = len(" ".join(paragraph_text_parts[:this_part_idx])) + int(
            this_part_idx > 0
        )
        ref_end = ref_start + len(ref.text)

        assert paragraph_text[ref_start:ref_end] == ref.text

        refs_for_paragraph.append(
            Ref(
                ref_type=normalize_pmcoa_ref_type(ref.attrib.get("ref-type")),
                rid=ref.attrib.get("rid"),
                start=ref_start,
                end=ref_end,
            )
        )
        prev_part_idx = this_part_idx
    return refs_for_paragraph


def locate_sentence_spans_within_paragraph(
    paragraph_text: str,
    sentence_texts: list[str],
) -> list[list[int]]:
    sentence_spans: list[list[int]] = []
    pos = 0
    for sent_text in sentence_texts:
        start = paragraph_text.find(sent_text, pos)
        assert start != -1, f"sentence_text not found in paragraph_text: {sent_text}"
        end = start + len(sent_text)
        sentence_spans.append([start, end])
        pos = end
    return sentence_spans


def allocate_refs_for_paragraph_to_sentences(
    refs_for_paragraph: list[Ref],
    sentence_spans: list[list[int]],
) -> tuple[list[list[Ref]], list[list[int]]]:
    if not refs_for_paragraph or len(sentence_spans) == 0:
        return [[] for _ in sentence_spans], sentence_spans

    refs_per_sentence: list[list[Ref]] = []
    curr_sent_idx = 0
    curr_ref_idx = 0

    while True:
        refs_for_sentence: list[Ref] = []

        while True:
            sentence_start, sentence_end = sentence_spans[curr_sent_idx]

            ref = refs_for_paragraph[curr_ref_idx]

            if ref.start < sentence_end:
                if ref.end > sentence_end:
                    # Ref crosses a sentence boundary; merge sentences until it fits.
                    merge_sent_idx = curr_sent_idx + 1
                    while True:
                        new_sentence_end = sentence_spans[merge_sent_idx][1]
                        if ref.end <= new_sentence_end:
                            sentence_spans[curr_sent_idx][1] = new_sentence_end
                            sentence_spans = (
                                sentence_spans[: curr_sent_idx + 1]
                                + sentence_spans[merge_sent_idx + 1 :]
                            )
                            break

                        merge_sent_idx += 1
                        if merge_sent_idx >= len(sentence_spans):
                            sentence_spans[curr_sent_idx][1] = sentence_spans[-1][1]
                            sentence_spans = sentence_spans[: curr_sent_idx + 1]
                            break

                ref_data = ref.model_dump()
                ref_data.update(
                    {
                        "start": ref.start - sentence_start,
                        "end": ref.end - sentence_start,
                    }
                )
                refs_for_sentence.append(Ref(**ref_data))

                curr_ref_idx += 1
                if curr_ref_idx >= len(refs_for_paragraph):
                    break
            else:
                break
        refs_per_sentence.append(refs_for_sentence)

        curr_sent_idx += 1
        if (
            curr_sent_idx >= len(sentence_spans)
            or curr_ref_idx >= len(refs_for_paragraph)
        ):
            break

    while len(refs_per_sentence) < len(sentence_spans):
        refs_per_sentence.append([])

    return refs_per_sentence, sentence_spans


def _get_best_abstract_node(root: etree._Element) -> etree._Element | None:
    """Return the best abstract element (English, longest, no abstract-type)."""
    nodes = root.xpath("front/article-meta/abstract")
    if not nodes:
        return None

    def score_node(node: etree._Element) -> tuple[int, int, int]:
        lang = get_xml_lang(node)
        is_en = 1 if (lang and lang.lower() == "en") else 0
        text = stringify(node, delimiter=" ") or ""
        no_type = 1 if "abstract-type" not in node.attrib else 0
        return (is_en, len(text), no_type)

    scored = [(score_node(n), n) for n in nodes if "abstract-type" not in n.attrib]
    if not scored:
        return nodes[0]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def _assemble_contents(
    p_el: etree._Element,
    text: str,
    text_parts: list[str],
    sentence_texts: list[str],
    path: list[str | int],
) -> list[Sentence]:
    """Build the final ``Sentence`` list from tokenized splits of ``text``.

    Callers are expected to only invoke this for non-empty texts (empty
    paragraphs / captions are represented by a single empty-text placeholder
    that the stub builders leave in place). Any ``<xref>`` descendants of
    ``p_el`` are localized into the resulting sentences using ``text_parts``.
    """
    if not sentence_texts:
        sentence_texts = [text]

    refs = locate_refs_in_paragraph(p_el, text_parts)
    sentence_spans = locate_sentence_spans_within_paragraph(text, sentence_texts)
    refs_per_sentence, sentence_spans = allocate_refs_for_paragraph_to_sentences(
        refs, sentence_spans
    )

    return [
        Sentence(
            content_id=[*path, i],
            text=text[span[0] : span[1]],
            refs=refs_for_sentence,
        )
        for i, (refs_for_sentence, span) in enumerate(
            zip(refs_per_sentence, sentence_spans)
        )
    ]


# ========= PARSER =========
@dataclass
class _TokenizeJob:
    """One unit of work for the batch tokenizer phase.

    After ``tokenize_batch`` returns, ``container.contents`` is replaced in
    place with the ``Sentence``s produced from ``text`` / ``text_parts`` and
    any ``<xref>``s found inside ``p_el``. The same structure is used for
    paragraphs and figure captions -- a caption is just a paragraph whose
    container happens to be a ``Figure``.
    """

    container: Paragraph | Figure
    p_el: etree._Element
    text: str
    text_parts: list[str]


class PaperParser:
    """Parse PMC JATS XML into ``Paper`` objects.

    Tokenization is performed exactly once per paper, in a single
    ``tokenizer.tokenize_batch(...)`` call after the XML tree walk has built
    the full structure.
    """

    def __init__(
        self,
        tokenizer: SentenceTokenizer,
        pmc_id_map: PmcIdMap | None = None,
    ):
        self._tokenizer = tokenizer
        self._pmc_id_map = pmc_id_map

    def parse(self, x: str | etree._ElementTree | Path) -> Paper:
        src = str(x) if isinstance(x, Path) else x
        tree = build_xml_tree(src)
        root_el = tree.getroot()
        strip_noise(root_el)

        try:
            bibliography = extract_bibliography(tree)
        except Exception as e:
            logger.warning(
                f"Failed to extract bibliography; source={x=}; using default. error={e}"
            )
            bibliography = {}

        if self._pmc_id_map is not None:
            for entry in bibliography.values():
                entry.all_paper_ids = self._pmc_id_map.augment(entry.all_paper_ids)

        expand_bibr_citation_ranges(root_el, bibliography)

        main_id = (
            extract_pmc_id_from_path(x) if isinstance(x, (str, Path)) else None
        )
        if not main_id:
            raise ValueError(
                f"main paper_id is required; could not extract from {x=}"
            )

        paper_ids = extract_paper_ids(tree)
        if all(i.id_type != "pmc" for i in paper_ids):
            paper_ids.append(main_id)

        # Backfill DOI/PMID from the external PMC-ids crosswalk when the XML
        # itself is missing them. Ids discovered in the XML always win.
        if self._pmc_id_map is not None:
            paper_ids = self._pmc_id_map.augment(paper_ids)

        try:
            pub_date = extract_pub_date(tree)
        except Exception as e:
            logger.warning(
                f"Failed to extract pub_date; source={x=}; using default. error={e}"
            )
            pub_date = None

        try:
            paper_type = extract_paper_type(tree)
        except Exception as e:
            logger.warning(
                f"Failed to extract paper_type; source={x=}; using default. error={e}"
            )
            paper_type = None

        try:
            subjects = extract_subjects(tree)
        except Exception as e:
            logger.warning(
                f"Failed to extract subjects; source={x=}; using default. error={e}"
            )
            subjects = []

        try:
            title = extract_title_sentence(tree)
        except Exception as e:
            logger.warning(
                f"Failed to extract title; source={x=}; using default. error={e}"
            )
            title = Sentence(content_id=["title", 0], text="")

        root = get_xml_root(tree)
        jobs: list[_TokenizeJob] = []

        try:
            abstract_node = _get_best_abstract_node(root)
            abstract = (
                self._build_content_list(abstract_node, ["abstract"], jobs)
                if abstract_node is not None
                else []
            )
        except Exception as e:
            logger.warning(
                f"Failed to extract abstract; source={x=}; using default. error={e}"
            )
            abstract = []

        try:
            bodies = root.xpath(".//*[local-name()='body']")
            maintext = (
                self._build_content_list(bodies[0], ["maintext"], jobs)
                if bodies
                else []
            )
        except Exception as e:
            logger.warning(
                f"Failed to extract maintext; source={x=}; using default. error={e}"
            )
            maintext = []

        self._fill_jobs(jobs)

        return Paper(
            paper_id=main_id,
            all_paper_ids=paper_ids,
            paper_type=paper_type,
            pub_date=pub_date,
            subjects=subjects,
            bibliography=bibliography,
            title=title,
            abstract=abstract,
            maintext=maintext,
        )

    # ----- tree walk (no tokenization) -----

    def _build_content_list(
        self,
        parent_el: etree._Element,
        base_path: list[str | int],
        jobs: list[_TokenizeJob],
    ) -> list[Content]:
        """Walk an abstract/body container into a list of ``Content``."""
        out: list[Content] = []
        index = 0
        for el in parent_el:
            tag = _local_tag(el)
            if tag == "p":
                out.append(self._build_paragraph_stub(el, [*base_path, index], jobs))
                index += 1
            elif tag == "sec":
                out.append(self._build_section(el, base_path, index, jobs))
                index += 1
            elif tag == "fig":
                out.append(self._build_figure_stub(el, [*base_path, index], jobs))
                index += 1
        return out

    def _build_section(
        self,
        sec_el: etree._Element,
        base_path: list[str | int],
        index: int,
        jobs: list[_TokenizeJob],
    ) -> Section:
        path = [*base_path, index]
        title_nodes = sec_el.xpath(".//*[local-name()='title']")
        title_text = stringify(title_nodes[0], delimiter=" ") if title_nodes else ""
        title_text = " ".join(title_text.split())
        title = Sentence(content_id=[*path, 0], text=title_text or "")

        children: list[Content] = []
        child_index = 0
        for el in sec_el:
            tag = _local_tag(el)
            if tag == "title":
                continue
            if tag == "p":
                children.append(
                    self._build_paragraph_stub(el, [*path, child_index], jobs)
                )
                child_index += 1
            elif tag == "sec":
                children.append(self._build_section(el, path, child_index, jobs))
                child_index += 1
            elif tag == "fig":
                children.append(
                    self._build_figure_stub(el, [*path, child_index], jobs)
                )
                child_index += 1

        return Section(
            content_id=path,
            content_type="section",
            title=title,
            contents=children,
        )

    def _build_paragraph_stub(
        self,
        p_el: etree._Element,
        path: list[str | int],
        jobs: list[_TokenizeJob],
    ) -> Paragraph:
        """Create a Paragraph whose ``contents`` will be filled in later."""
        placeholder = Sentence(content_id=[*path, 0], text="")
        paragraph = Paragraph(content_id=path, contents=[placeholder])
        self._enqueue_text_job(paragraph, p_el, jobs)
        return paragraph

    def _build_figure_stub(
        self,
        fig_el: etree._Element,
        fig_path: list[str | int],
        jobs: list[_TokenizeJob],
    ) -> Figure:
        """Create a Figure whose caption ``contents`` will be filled in later.

        The ``<caption>`` element is treated like a paragraph source -- inline
        ``<xref>``s inside the caption are localized onto the tokenized
        sentences just like for regular paragraphs.
        """
        placeholder = Sentence(content_id=[*fig_path, 0], text="")
        figure = Figure(
            content_id=fig_path,
            content_type="figure",
            contents=[placeholder],
        )
        cap_nodes = fig_el.xpath(".//*[local-name()='caption']")
        if cap_nodes:
            self._enqueue_text_job(figure, cap_nodes[0], jobs)
        return figure

    def _enqueue_text_job(
        self,
        container: Paragraph | Figure,
        p_el: etree._Element,
        jobs: list[_TokenizeJob],
    ) -> None:
        """Record a tokenization job for ``container`` sourced from ``p_el``.

        Footnote-like content has already been stripped from the whole XML
        tree at the top of ``parse``, so this helper can treat ``p_el`` as
        reading-text only. If the paragraph has no non-whitespace text, the
        container keeps its empty placeholder sentence and no job is
        enqueued.
        """
        text_parts = [" ".join(x.split()).strip() for x in stringify(p_el)]
        text_parts = [x for x in text_parts if x]
        text = " ".join(text_parts)
        if not text:
            return
        jobs.append(
            _TokenizeJob(
                container=container,
                p_el=p_el,
                text=text,
                text_parts=text_parts,
            )
        )

    # ----- tokenize once, fill everything -----

    def _fill_jobs(self, jobs: list[_TokenizeJob]) -> None:
        if not jobs:
            return

        texts = [job.text for job in jobs]
        sents_per_job = self._tokenizer.tokenize_batch(texts)

        for job, sents in zip(jobs, sents_per_job):
            job.container.contents = _assemble_contents(
                p_el=job.p_el,
                text=job.text,
                text_parts=job.text_parts,
                sentence_texts=sents,
                path=list(job.container.content_id),
            )


def extract_paper(
    x: str | etree._ElementTree | Path,
    tokenizer: SentenceTokenizer,
) -> Paper:
    """Convenience wrapper around ``PaperParser(tokenizer).parse(x)``."""
    return PaperParser(tokenizer).parse(x)
