"""USPTO patent field name → human-readable label mappings.

Provides bilingual (zh/en) display labels for raw USPTO API field paths so
that exported Excel files show user-friendly column headers instead of
dotted API field names like ``applicationMetaData.inventionTitle``.

The three-tier lookup mirrors the pattern in :mod:`sources.jpo_client`::

    1. Exact path match (e.g. ``applicationMetaData.inventionTitle``)
    2. Leaf-name match (e.g. ``.inventionTitle`` → via ``inventionTitle`` key)
    3. Contains scan (safety net for deeply nested paths)
    4. Identity fallback — returns the original field path unchanged

Usage::

    from sources.export_labels import uspto_field_label

    zh_label = uspto_field_label("applicationMetaData.inventionTitle", "zh")
    # → "发明名称"
    en_label = uspto_field_label("applicationMetaData.inventionTitle", "en")
    # → "Invention Title"
"""

# ── English human-readable labels ──────────────────────────────────────
# Used when lang == "en" (but not the raw API name, which is the fallback).
USPTO_FIELD_LABELS_EN: dict[str, str] = {
    # Top-level fields
    "applicationNumberText": "Application Number",
    "officialDate": "Official Date",
    "documentIdentifier": "Document ID",
    "documentCode": "Document Code",
    "documentCodeDescriptionText": "Document Description",
    "directionCategory": "Direction",
    "downloadOptionBag": "Download Options",
    "eventDataBag": "Event History",
    "assignmentBag": "Assignment Records",
    "correspondenceAddressBag": "Correspondence Address",
    "foreignPriorityBag": "Foreign Priority Claims",
    "parentContinuityBag": "Parent Continuity",
    "childContinuityBag": "Child Continuity",
    "lastIngestionDateTime": "Last Ingestion Date",

    # applicationMetaData
    "applicationMetaData.inventionTitle": "Invention Title",
    "applicationMetaData.firstApplicantName": "First Applicant",
    "applicationMetaData.firstInventorName": "First Inventor",
    "applicationMetaData.filingDate": "Filing Date",
    "applicationMetaData.effectiveFilingDate": "Effective Filing Date",
    "applicationMetaData.grantDate": "Grant Date",
    "applicationMetaData.patentNumber": "Patent Number",
    "applicationMetaData.earliestPublicationNumber": "Earliest Publication Number",
    "applicationMetaData.earliestPublicationDate": "Earliest Publication Date",
    "applicationMetaData.applicationStatusDescriptionText": "Application Status",
    "applicationMetaData.applicationStatusCode": "Status Code",
    "applicationMetaData.applicationStatusDate": "Status Date",
    "applicationMetaData.applicationTypeLabelName": "Application Type",
    "applicationMetaData.applicationTypeCategory": "Type Category",
    "applicationMetaData.applicationTypeCode": "Type Code",
    "applicationMetaData.groupArtUnitNumber": "Art Unit",
    "applicationMetaData.applicationConfirmationNumber": "Confirmation Number",
    "applicationMetaData.docketNumber": "Docket Number",
    "applicationMetaData.firstInventorToFileIndicator": "First Inventor to File",
    "applicationMetaData.nationalStageIndicator": "National Stage",
    "applicationMetaData.smallEntityStatusIndicator": "Small Entity Status",
    "applicationMetaData.businessEntityStatusCategory": "Business Entity Category",
    "applicationMetaData.pctPublicationNumber": "PCT Publication Number",
    "applicationMetaData.pctPublicationDate": "PCT Publication Date",

    # pgpubDocumentMetaData (leaf prefixes)
    "pgpubDocumentMetaData.productIdentifier": "Publ. Product ID",
    "pgpubDocumentMetaData.zipFileName": "Publ. ZIP File",
    "pgpubDocumentMetaData.fileCreateDateTime": "Publ. File Created",
    "pgpubDocumentMetaData.xmlFileName": "Publ. XML File",
    "pgpubDocumentMetaData.fileLocationURI": "Publ. File Location",

    # grantDocumentMetaData
    "grantDocumentMetaData.productIdentifier": "Grant Product ID",
    "grantDocumentMetaData.zipFileName": "Grant ZIP File",
    "grantDocumentMetaData.fileCreateDateTime": "Grant File Created",
    "grantDocumentMetaData.xmlFileName": "Grant XML File",
    "grantDocumentMetaData.fileLocationURI": "Grant File Location",

    # recordAttorney (leaf prefixes)
    "recordAttorney.attorneyBag": "Attorney Info",
    "recordAttorney.powerOfAttorneyBag": "Power of Attorney",
    "recordAttorney.customerNumberCorrespondenceData.powerOfAttorneyAddressBag": "Attorney Address",
    "recordAttorney.customerNumberCorrespondenceData.patronIdentifier": "Customer Number",

    # patentTermAdjustmentData (leaf prefixes)
    "patentTermAdjustmentData.adjustmentTotalQuantity": "PTA Total Days",
    "patentTermAdjustmentData.aDelayQuantity": "PTA A-Delay",
    "patentTermAdjustmentData.bDelayQuantity": "PTA B-Delay",
    "patentTermAdjustmentData.cDelayQuantity": "PTA C-Delay",
    "patentTermAdjustmentData.applicantDayDelayQuantity": "PTA Applicant Delay",
    "patentTermAdjustmentData.overlappingDayQuantity": "PTA Overlap Days",
    "patentTermAdjustmentData.nonOverlappingDayDelayQuantity": "PTA Non-Overlap Days",
    "patentTermAdjustmentData.ipOfficeAdjustmentDelayQuantity": "PTA Office Delay",
    "patentTermAdjustmentData.patentTermAdjustmentHistoryDataBag": "PTA History",

    # Leaf-level entries for partial matching (nested fields that share a
    # common suffix across different parent paths)
    "firstInventorToFileIndicator": "First Inventor to File",
    "applicationStatusCode": "Status Code",
    "applicationTypeCode": "Type Code",
    "filingDate": "Filing Date",
    "applicationStatusDescriptionText": "Application Status",
    "groupArtUnitNumber": "Art Unit",
    "earliestPublicationNumber": "Earliest Publication Number",
    "inventionTitle": "Invention Title",
    "nationalStageIndicator": "National Stage",
    "effectiveFilingDate": "Effective Filing Date",
    "applicationConfirmationNumber": "Confirmation Number",
    "earliestPublicationDate": "Earliest Publication Date",
    "applicationTypeLabelName": "Application Type",
    "applicationStatusDate": "Status Date",
    "docketNumber": "Docket Number",
    "applicationTypeCategory": "Type Category",
    "patentNumber": "Patent Number",
    "grantDate": "Grant Date",
    "firstApplicantName": "First Applicant",
    "firstInventorName": "First Inventor",
    "smallEntityStatusIndicator": "Small Entity Status",
    "businessEntityStatusCategory": "Business Entity Category",
    "pctPublicationNumber": "PCT Publication Number",
    "pctPublicationDate": "PCT Publication Date",
}

# ── Chinese labels ─────────────────────────────────────────────────────
USPTO_FIELD_LABELS_ZH: dict[str, str] = {
    # Top-level fields
    "applicationNumberText": "申请号",
    "officialDate": "官方日期",
    "documentIdentifier": "文档ID",
    "documentCode": "文档代码",
    "documentCodeDescriptionText": "文档描述",
    "directionCategory": "方向",
    "downloadOptionBag": "下载选项",
    "eventDataBag": "事件记录",
    "assignmentBag": "转让记录",
    "correspondenceAddressBag": "通讯地址",
    "foreignPriorityBag": "外国优先权",
    "parentContinuityBag": "母案连续性",
    "childContinuityBag": "子案连续性",
    "lastIngestionDateTime": "最后收录时间",

    # applicationMetaData
    "applicationMetaData.inventionTitle": "发明名称",
    "applicationMetaData.firstApplicantName": "第一申请人",
    "applicationMetaData.firstInventorName": "第一发明人",
    "applicationMetaData.filingDate": "申请日",
    "applicationMetaData.effectiveFilingDate": "有效申请日",
    "applicationMetaData.grantDate": "授权日",
    "applicationMetaData.patentNumber": "专利号",
    "applicationMetaData.earliestPublicationNumber": "最早公开号",
    "applicationMetaData.earliestPublicationDate": "最早公开日",
    "applicationMetaData.applicationStatusDescriptionText": "申请状态",
    "applicationMetaData.applicationStatusCode": "申请状态码",
    "applicationMetaData.applicationStatusDate": "状态日期",
    "applicationMetaData.applicationTypeLabelName": "申请类型",
    "applicationMetaData.applicationTypeCategory": "申请类别",
    "applicationMetaData.applicationTypeCode": "申请类型代码",
    "applicationMetaData.groupArtUnitNumber": "审查部门",
    "applicationMetaData.applicationConfirmationNumber": "确认号",
    "applicationMetaData.docketNumber": "案卷号",
    "applicationMetaData.firstInventorToFileIndicator": "发明人先申请制",
    "applicationMetaData.nationalStageIndicator": "国家阶段",
    "applicationMetaData.smallEntityStatusIndicator": "小实体状态",
    "applicationMetaData.businessEntityStatusCategory": "商业实体类别",
    "applicationMetaData.pctPublicationNumber": "PCT公开号",
    "applicationMetaData.pctPublicationDate": "PCT公开日",

    # pgpubDocumentMetaData
    "pgpubDocumentMetaData.productIdentifier": "公开文档产品ID",
    "pgpubDocumentMetaData.zipFileName": "公开文档ZIP文件",
    "pgpubDocumentMetaData.fileCreateDateTime": "公开文件创建时间",
    "pgpubDocumentMetaData.xmlFileName": "公开文档XML文件",
    "pgpubDocumentMetaData.fileLocationURI": "公开文件位置",

    # grantDocumentMetaData
    "grantDocumentMetaData.productIdentifier": "授权文档产品ID",
    "grantDocumentMetaData.zipFileName": "授权文档ZIP文件",
    "grantDocumentMetaData.fileCreateDateTime": "授权文件创建时间",
    "grantDocumentMetaData.xmlFileName": "授权文档XML文件",
    "grantDocumentMetaData.fileLocationURI": "授权文件位置",

    # recordAttorney
    "recordAttorney.attorneyBag": "代理人信息",
    "recordAttorney.powerOfAttorneyBag": "代理授权书",
    "recordAttorney.customerNumberCorrespondenceData.powerOfAttorneyAddressBag": "代理地址",
    "recordAttorney.customerNumberCorrespondenceData.patronIdentifier": "客户编号",

    # patentTermAdjustmentData
    "patentTermAdjustmentData.adjustmentTotalQuantity": "专利期限调整总天数",
    "patentTermAdjustmentData.aDelayQuantity": "PTA A类延迟",
    "patentTermAdjustmentData.bDelayQuantity": "PTA B类延迟",
    "patentTermAdjustmentData.cDelayQuantity": "PTA C类延迟",
    "patentTermAdjustmentData.applicantDayDelayQuantity": "PTA 申请人延迟天数",
    "patentTermAdjustmentData.overlappingDayQuantity": "PTA 重叠天数",
    "patentTermAdjustmentData.nonOverlappingDayDelayQuantity": "PTA 非重叠延迟天数",
    "patentTermAdjustmentData.ipOfficeAdjustmentDelayQuantity": "PTA 官方延迟天数",
    "patentTermAdjustmentData.patentTermAdjustmentHistoryDataBag": "PTA 历史记录",

    # Leaf-level entries for partial matching
    "firstInventorToFileIndicator": "发明人先申请制",
    "applicationStatusCode": "申请状态码",
    "applicationTypeCode": "申请类型代码",
    "filingDate": "申请日",
    "applicationStatusDescriptionText": "申请状态",
    "groupArtUnitNumber": "审查部门",
    "earliestPublicationNumber": "最早公开号",
    "inventionTitle": "发明名称",
    "nationalStageIndicator": "国家阶段",
    "effectiveFilingDate": "有效申请日",
    "applicationConfirmationNumber": "确认号",
    "earliestPublicationDate": "最早公开日",
    "applicationTypeLabelName": "申请类型",
    "applicationStatusDate": "状态日期",
    "docketNumber": "案卷号",
    "applicationTypeCategory": "申请类别",
    "patentNumber": "专利号",
    "grantDate": "授权日",
    "firstApplicantName": "第一申请人",
    "firstInventorName": "第一发明人",
    "smallEntityStatusIndicator": "小实体状态",
    "businessEntityStatusCategory": "商业实体类别",
    "pctPublicationNumber": "PCT公开号",
    "pctPublicationDate": "PCT公开日",
}


def uspto_field_label(field_path: str, lang: str = "zh") -> str:
    """Return a human-readable label for a USPTO API field path.

    Performs a three-tier lookup and falls back to the original *field_path*
    when no mapping exists — the caller never receives an empty string.

    Args:
        field_path: Dotted USPTO API field path, e.g.
            ``"applicationMetaData.inventionTitle"`` or ``"eventDataBag"``.
        lang: Language code — ``"zh"`` for Chinese, ``"en"`` for English.
            Other values are normalized to ``"zh"``.

    Returns:
        Human-readable label in the requested language, or the original
        *field_path* unchanged if no mapping exists.
    """
    # Normalize lang
    lang = lang.split("-")[0].lower() if lang else "zh"
    if lang not in ("zh", "en"):
        lang = "zh"

    labels = USPTO_FIELD_LABELS_ZH if lang == "zh" else USPTO_FIELD_LABELS_EN

    # Tier 1 — exact path match
    if field_path in labels:
        return labels[field_path]

    # Tier 2 — leaf-name match (e.g. "applicationMetaData.inventionTitle"
    # matched via the leaf key "inventionTitle")
    leaf = field_path.rsplit(".", 1)[-1]
    if leaf in labels:
        return labels[leaf]

    # Tier 3 — contains scan (safety net)
    for api_key, label in labels.items():
        if api_key in field_path:
            return label

    # Tier 4 — identity fallback
    return field_path
