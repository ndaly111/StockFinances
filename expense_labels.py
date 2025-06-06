"""
expense_labels.py
Single source of truth for identifying income-statement expense rows
--------------------------------------------------------------------

Edit the lists below whenever Yahoo adds / changes a label.
No other code needs to change.
"""

# ----------   Cost of revenue  ----------
COST_OF_REVENUE = [
    "Cost Of Revenue",
    "Reconciled Cost Of Revenue",
]

# ----------   Research & development  ----------
RESEARCH_AND_DEVELOPMENT = [
    "Research & Development",
    "Research and Development",
    "R&D",
]

# ----------   Selling & marketing  ----------
SELLING_AND_MARKETING = [
    "Selling and Marketing",
    "Sales and Marketing",
]

# ----------   General & administrative  ----------
GENERAL_AND_ADMIN = [
    "General and Administrative",
]

# ----------   SG&A combined variants  ----------
SGA_COMBINED = [
    "Selling General & Administrative",
    "Selling, General & Administrative",
    "Sales, General & Administrative",
    "Selling General And Administration",
    "Selling General and Administration",
    "Selling, General and Administration",
    "Sales, General and Administration",
]

# -----------  Handy container if needed  -----------
ALL_EXPENSE_LABELS = (
    COST_OF_REVENUE
    + RESEARCH_AND_DEVELOPMENT
    + SELLING_AND_MARKETING
    + GENERAL_AND_ADMIN
    + SGA_COMBINED
)
