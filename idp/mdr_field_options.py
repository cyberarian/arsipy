# d:\pandasai\idp\mdr_field_options.py
import streamlit as st
"""
This file contains predefined options for various fields
in the Master Document Register (MDR) to be used as dropdowns.
"""

DISCIPLINE_OPTIONS = [
    "A - Architectural", "B - Business & General", "C - Civil & Infrastructure",
    "D - Drilling", "E - Electrical", "F - Risk & Loss Prevention",
    "G - Geotechnical", "H - HVAC", "I - Instrumentation and Metering",
    "J - Marine", "K - Construction, Transportation & Logistics",
    "L - Piping & General Layout", "M - Mechanical and Machinery",
    "N - Structural", "O - Operations Readiness", "P - Process",
    "Q - Quality Management", "R - Regulatory/Environmental/Socioeconomic",
    "S - Safety, Health and Security", "T - Telecommunications",
    "U - Subsea", "V - Contracting & Procurement",
    "W - Systems Completions/Commissioning/Start-Up",
    "X - Materials/Corrosion/Flow Assurance", "Y - Pipelines, Umbilicals, Risers, Flowlines"
]

# Mapping user's "Doc Type" to "Document Type (Detail)" in METADATA_COLUMNS
DOCUMENT_TYPE_DETAIL_OPTIONS = [
    "A - Administration", "B - Design Criteria, Philosophies & Basis",
    "Q - Certifications/Warranties & Quality Records", "C - Calculations",
    "D - Drawings", "L - Lists and Schedules", "P - Procedures & Plans",
    "R - Reports/Technical Studies & Assessments",
    "S - Specifications/Codes/Technical Data Sheets", "H - Models & Databases",
    "M - Manuals", "X - Purchase Orders & Requisitions", "G - Regulatory Submissions"
]

# Mapping user's "Document Type" to "Document Type" in METADATA_COLUMNS
DOCUMENT_TYPE_OPTIONS = [
    "Technical Deliverable", "Non-Technical Deliverable",
    "General Reference", "Supplier Documents"
]

STATUS_OPTIONS = [
    "To Be Produced", "IAB - Issued for As-Built", "IFC - Issued for Construction",
    "IFB - Issued for Bid", "IFD - Issued for Design", "IFF - Issued for FEED",
    "IFH - Issued for HAZOP", "IFI - Issued for Information",
    "IFP - Issued for Purchase", "IFR - Issued for Review",
    "IFT - Issued for Tender", "IFU - Issued for Use",
    "IFV - Issued for Void", "IFRL - Issued for Redlines"
]

PHASE_OPTIONS = [
    "*General Reference (Not developed for Project)", "EPC", "FEED",
    "Operations", "Pre-FEED"
]

# Mapping user's "Area Location" to "Area/Unit/Location" in METADATA_COLUMNS
AREA_UNIT_LOCATION_OPTIONS = [
    "00 - General", "10 - Offshore Platform – Common",
    "11 - Offshore Platform – Train 1", "12 - Offshore Platform – Train 2",
    "13 - Offshore Platform – Train 3", "15 - Offshore Platform - Jacket",
    "20 - Offshore Pipelines", "25 - Onshore Inlet Pipelines",
    "30 - Onshore Gas Treatment Plant – Common",
    "31 - Onshore Gas Treatment Plant – Train 1",
    "32 - Onshore Gas Treatment Plant – Train 2",
    "33 - Onshore Gas Treatment Plant – Train 3",
    "40 - Onshore Pipelines",
    "50 - Onshore Camp, Maintenance, and Operations Facilities",
    "70 - Onshore Gas Pipelines and Metering Stations",
    "80 - Onshore Condensate Pipeline", "90 - Offloading Terminal Ky Ha Port"
]

REVIEW_CATEGORY_OPTIONS = [
    "I - Company AND Verification Agent Review Required",
    "II - Company Review Required (May NOT proceed)",
    "III - Company Review Required (May proceed)",
    "IV - Company Review Not Required"
]

# A dictionary to easily map METADATA_COLUMNS to their options
MDR_SELECTBOX_OPTIONS_MAP = {
    "Discipline": DISCIPLINE_OPTIONS,
    "Document Type": DOCUMENT_TYPE_OPTIONS,
    "Document Type (Detail)": DOCUMENT_TYPE_DETAIL_OPTIONS,
    "Status": STATUS_OPTIONS,
    "Phase": PHASE_OPTIONS,
    "Area/Unit/Location": AREA_UNIT_LOCATION_OPTIONS,
    "Review Category": REVIEW_CATEGORY_OPTIONS,
}

# Helper function to get column configurations
def get_mdr_column_config(all_metadata_columns, options_map):
    config = {}
    for col in all_metadata_columns:
        if col in options_map:
            config[col] = st.column_config.SelectboxColumn(
                col,
                options=options_map[col],
                required=False # Or True if you want to enforce selection
            )
        else:
            # Default to TextColumn for other fields
            config[col] = st.column_config.TextColumn(col)
    return config