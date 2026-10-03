from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

import lxml.etree as etree  # ty: ignore[unresolved-import]


class PageXMLValidationError(ValueError):
    pass


def secure_xml_parser() -> etree.XMLParser:
    """Create a parser with entity resolution, DTD loading, and network access off."""
    return etree.XMLParser(
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        recover=False,
        huge_tree=False,
    )


def parse_xml_document(path: Path) -> etree._ElementTree:
    try:
        return etree.parse(str(path), parser=secure_xml_parser())
    except (OSError, etree.XMLSyntaxError) as exc:
        raise PageXMLValidationError(
            f"Invalid PAGE-XML document {path}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def _page_schema() -> etree.XMLSchema:
    schema_resource = files("synthscript.loading.page_xml").joinpath(
        "resources/pagecontent.xsd"
    )
    with as_file(schema_resource) as schema_path:
        schema_document = etree.parse(str(schema_path), parser=secure_xml_parser())
    return etree.XMLSchema(schema_document)


def validate_page_xml(path: Path) -> None:
    """Validate a PAGE document against the bundled 2024-07-15 XSD."""
    document = parse_xml_document(Path(path))
    schema = _page_schema()
    if not schema.validate(document):
        error = schema.error_log.last_error
        detail = error.message if error is not None else "unknown schema error"
        raise PageXMLValidationError(f"PAGE-XML validation failed for {path}: {detail}")
