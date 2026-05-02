from __future__ import annotations
from typing import Annotated, Literal, Union, TypeAlias

from datetime import date
from pydantic import BaseModel, BeforeValidator, Field, model_validator


CONTENT_ID_REGIONS = ("title", "abstract", "maintext")


def _validate_content_id(v: list) -> list[str | int]:
    """Ensure content_id is [region, int, int, ...] with region in (title, abstract, maintext)."""
    if not v:
        raise ValueError("content_id must not be empty")
    if not isinstance(v[0], str):
        raise ValueError("content_id must start with a string (region)")
    if v[0] not in CONTENT_ID_REGIONS:
        raise ValueError(f"content_id region must be one of {CONTENT_ID_REGIONS}, got {v[0]!r}")
    for i, x in enumerate(v[1:], 1):
        if not isinstance(x, int) or x < 0:
            raise ValueError(f"content_id[{i}] must be non-negative int, got {type(x).__name__}")
    return v


def stringify_content(
    content: Content,
    delimiter: str | None = None,
    skip_types: set[str] | None = None,
) -> str | list[str]:
    """Join the text of all descendant items of a content node.

    - For Sentence: uses .text
    - For Figure: uses .caption
    - For Paragraph: recursively stringifies all items in .contents
    - For Section: stringifies the title and recursively stringifies all items in .contents
    - `skip_types`: optional set of content_type strings to skip entirely
      (the node and its descendants are ignored)
    """

    def _gather(c: Content) -> list[str]:
        if skip_types and getattr(c, "content_type", None) in skip_types:
            return []

        elif isinstance(c, Sentence):
            return [c.text]

        elif isinstance(c, Figure):
            parts: list[str] = []
            for child in c.contents or []:
                parts.extend(_gather(child))
            return parts

        elif isinstance(c, (Paragraph, Section)):
            child_contents = getattr(c, "contents", None) or []
            parts: list[str] = []
            if isinstance(c, Section):
                parts.append(c.title.text)
            for child in child_contents:
                parts.extend(_gather(child))
            return parts
        
        else:
            raise ValueError(f"Unknown content type: {type(c).__name__}")

    parts = _gather(content)
    if delimiter is None:
        return parts
    return delimiter.join(p for p in parts if p)


def _stringify_contents(
    contents: list[Content],
    delimiter: str | None = None,
    skip_types: set[str] | None = None,
) -> str | list[str]:
    parts: list[str] = []
    for node in contents:
        node_parts = stringify_content(node, delimiter=None, skip_types=skip_types)
        parts.extend(node_parts)
    if delimiter is None:
        return parts
    return delimiter.join(p for p in parts if p)


def _check_content_ids_sequential(
    contents: list[Content], base_path: list[str | int]
) -> None:
    """Raise if content_id at each node is not base_path + [0], [1], ... with no gaps."""
    for i, c in enumerate(contents):
        expected = [*base_path, i]
        if list(c.content_id) != expected:
            raise ValueError(
                f"content_id must be {expected} (position {i} under {base_path}), got {c.content_id}"
            )
        # Recurse into Section/Paragraph contents
        child_contents = getattr(c, "contents", None)
        if child_contents is not None:
            _check_content_ids_sequential(child_contents, expected)


# --- Paper content ---


ContentId = Annotated[list[str | int], BeforeValidator(_validate_content_id)]
ContentType: TypeAlias = Literal["section", "paragraph", "sentence", "figure"]

PaperIdType: TypeAlias = Literal["pmc", "pmid", "doi"]

DEFAULT_REF_TYPE = "other"
RefType: TypeAlias = Literal["bib_entry", DEFAULT_REF_TYPE] | ContentType


class ContentBase(BaseModel):
    content_id: ContentId
    content_type: ContentType


class Ref(BaseModel):
    ref_type: RefType | None = None
    rid: str | None = None
    start: int
    end: int
    
    @model_validator(mode="after")
    def validate_ref(self) -> "Ref":
        if self.start < 0 or self.end < 0:
            raise ValueError("ref start and end must be non-negative")
        if self.start > self.end:
            raise ValueError("ref start must be less than end")
        return self

class Sentence(ContentBase):
    content_type: Literal["sentence"] = "sentence"
    text: str
    refs: list[Ref] = Field(default_factory=list)


class Paragraph(ContentBase):
    content_type: Literal["paragraph"] = "paragraph"
    contents: list["Content"]


class Section(ContentBase):
    content_type: Literal["section"] = "section"
    title: Sentence
    contents: list["Content"]


class Figure(ContentBase):
    content_type: Literal["figure"] = "figure"
    contents: list[Sentence]


Content = Annotated[
    Union[Section, Paragraph, Sentence, Figure],
    Field(discriminator="content_type"),
]
Section.model_rebuild()
Paragraph.model_rebuild()

class Date(BaseModel):
    """Publication date. All fields int | None. 
    
    Hierarchical: year-only OK; month requires year; day requires month and year. 
    
    Invalid parts are set to None."""

    year: int | None = None
    month: int | None = None
    day: int | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_date(cls, data: object) -> dict:
        if not isinstance(data, dict):
            return data
        year = data.get("year")
        month = data.get("month")
        day = data.get("day")

        # Hierarchical: month requires year, day requires month and year
        if year is None:
            month, day = None, None
        elif month is None:
            day = None

        # If full date, validate it exists; otherwise drop day
        if year is not None and month is not None and day is not None:
            try:
                date(year, month, day)
            except ValueError:
                day = None

        return {"year": year, "month": month, "day": day}


class PaperId(BaseModel):
    id_type: PaperIdType
    value: str | None = None

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"PaperId(id_type={self.id_type!r}, value={self.value!r})"

    def __hash__(self) -> int:
        # Allows use as dict keys/sets while still being stable across normalization.
        return hash((self.id_type, self.value))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    @model_validator(mode="after")
    def validate_value(self) -> "PaperId":
        if self.value is None:
            return self

        if self.id_type == "pmc":
            if not self.value.upper().startswith("PMC"):
                raise ValueError(f"PMC ids must start with 'PMC', got {self.value}")
            if not all(c.isdigit() for c in self.value[3:]):
                raise ValueError(f"PMC ids must contain digits after 'PMC', got {self.value}")

        elif self.id_type == "pmid":
            if not all(c.isdigit() for c in self.value):
                raise ValueError(f"PMID must be digits only, got {self.value}")

        elif self.id_type == "doi":
            if " " in self.value:
                raise ValueError(f"DOI must not contain spaces, got {self.value}")
            if "/" not in self.value:
                raise ValueError(f"DOI must contain '/', got {self.value}")

        return self


class BibEntry(BaseModel):
    rid: str
    all_paper_ids: list[PaperId] = Field(default_factory=list)
    label: str | None = None


class Paper(BaseModel):
    paper_id: PaperId
    all_paper_ids: list[PaperId] = Field(default_factory=list)
    paper_type: str | None = None
    pub_date: Date | None = None
    subjects: list[str] = Field(default_factory=list)
    bibliography: dict[str, BibEntry] = Field(default_factory=dict)

    title: Sentence
    abstract: list[Content]
    maintext: list[Content]

    def stringify_abstract(self, delimiter: str | None = None, skip_types: set[str] | None = None) -> str | list[str]:
        return _stringify_contents(self.abstract, delimiter, skip_types)

    def stringify_maintext(self, delimiter: str | None = None, skip_types: set[str] | None = None) -> str | list[str]:
        return _stringify_contents(self.maintext, delimiter, skip_types)

    def get_content(self, content_id: ContentId) -> Content | Sentence | Figure:
        """Return the content object for a given content_id.

        Follows the same assumptions enforced by _check_content_ids_sequential:
        - content_id[0] is one of ('title', 'abstract', 'maintext')
        - Remaining integers index into nested .contents lists.

        Raises ValueError / IndexError if the path is invalid.
        """
        if not content_id:
            raise ValueError("content_id must not be empty")

        region, *indices = content_id

        current_list: list[Content]
        if region == "title":
            current_list = [self.title]
        elif region == "abstract":
            current_list = self.abstract
        elif region == "maintext":
            current_list = self.maintext
        else:
            raise ValueError(f"Unknown content region {region!r}")

        if not indices:
            raise ValueError(f"Missing indices after region {region!r} in content_id {content_id}")

        current: Content | None = None
        for depth, idx in enumerate(indices):
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"Invalid index {idx!r} at depth {depth} in content_id {content_id}")
            if idx >= len(current_list):
                raise IndexError(
                    f"Index {idx} out of range at depth {depth} for region {region!r} "
                    f"(len={len(current_list)}) in content_id {content_id}"
                )

            current = current_list[idx]

            # If there are further indices, we must traverse into .contents
            if depth < len(indices) - 1:
                child_contents = getattr(current, "contents", None)
                if child_contents is None:
                    raise ValueError(
                        f"content_id {content_id} goes deeper than allowed: "
                        f"node at depth {depth} has no 'contents'"
                    )
                current_list = child_contents

        if current is None:
            # Should not happen if indices is non-empty and checks above pass,
            # but keep for completeness.
            raise ValueError(f"Unable to resolve content_id {content_id}")

        return current

    @model_validator(mode="after")
    def validate_content_ids(self) -> "Paper":
        _check_content_ids_sequential([self.title], ["title"])
        _check_content_ids_sequential(self.abstract, ["abstract"])
        _check_content_ids_sequential(self.maintext, ["maintext"])
        return self
