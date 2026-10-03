PAGE_VERSION = "2024-07-15"
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2024-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
PAGE_SCHEMA_URL = f"{PAGE_NS}/pagecontent.xsd"

SYNTHSCRIPT_STROKE_IMAGE = "synthscript:stroke"
SYNTHSCRIPT_BACKGROUND_IMAGE = "synthscript:background"

SYNTHSCRIPT_PAGE_ID = "synthscript.page_id"
SYNTHSCRIPT_SOURCE = "synthscript.source"
SYNTHSCRIPT_SUBINDEX = "synthscript.subindex"
SYNTHSCRIPT_ORIGINAL_ID = "synthscript.original_id"

LABEL_STUDIO_TASK_ID = "label_studio.task_id"
LABEL_STUDIO_ANNOTATION_ID = "label_studio.annotation_id"
LABEL_STUDIO_COMPLETER = "label_studio.completer"
LABEL_STUDIO_UPDATER = "label_studio.updater"
LABEL_STUDIO_PROJECT = "label_studio.project"
LABEL_STUDIO_UNIQUE_ID = "label_studio.unique_id"
LABEL_STUDIO_GROUND_TRUTH = "label_studio.ground_truth"
LABEL_STUDIO_CREATED_AT = "label_studio.created_at"
LABEL_STUDIO_UPDATED_AT = "label_studio.updated_at"
LABEL_STUDIO_LEAD_TIME = "label_studio.lead_time"

PAGE_XML_CREATOR = "page_xml.creator"
PAGE_XML_CREATED = "page_xml.created"
PAGE_XML_LAST_CHANGE = "page_xml.last_change"

NSMAP = {None: PAGE_NS, "xsi": XSI_NS}


def page_tag(local_name: str) -> str:
    return f"{{{PAGE_NS}}}{local_name}"
