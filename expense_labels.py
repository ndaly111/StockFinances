# ---------- Cost of Revenue ----------
COST_OF_REVENUE = [
    "Cost Of Revenue",
    "Reconciled Cost Of Revenue",
]

# ---------- Research & Development ----------
RESEARCH_AND_DEVELOPMENT = [
    "Research And Development",
    "Research & Development",
    "R&D",
]

# ---------- Selling & Marketing ----------
SELLING_AND_MARKETING = [
    "Selling And Marketing Expense",
    "Selling and Marketing",
    "Sales and Marketing",
]

# ---------- SG&A Combined ----------
SGA_COMBINED = [
    "Selling General And Administration",
    "Selling General & Administrative",
    "Selling, General & Administrative",
    "Sales, General & Administrative",
    "Selling General and Administration",
    "Selling, General and Administration",
    "Sales, General and Administration",
]

# ---------- General & Administrative ----------
GENERAL_AND_ADMIN = [
    "General And Administrative Expense",
    "Other Gand A",
]

# ---------- Facilities / Depreciation & Amortization ----------
FACILITIES_DA = [
    "Amortization",
    "Amortization Of Intangibles Income Statement",
    "Depreciation Amortization Depletion Income Statement",
    "Depreciation And Amortization In Income Statement",
    "Depreciation Income Statement",
    "Reconciled Depreciation",
    "Rent Expense Supplemental",
    "Occupancy And Equipment",
]

# ---------- Personnel ----------
PERSONNEL_COSTS = [
    "Salaries And Wages",
    "Professional Expense And Contract Services Expense",
]

# ---------- Insurance / Claims ----------
INSURANCE_CLAIMS = [
    "Insurance And Claims",
    "Loss Adjustment Expense",
    "Net Policyholder Benefits And Claims",
    "Policyholder Benefits Ceded",
    "Policyholder Benefits Gross",
]

# ---------- Provisions & Other Operating ----------
OTHER_OPERATING = [
    "Provision For Doubtful Accounts",
    "Excise Taxes",
    "Other Operating Expenses",
]

# Expense categories that should be ignored entirely. These entries typically
# represent totals that already include other line items and would otherwise
# lead to double-counting when stacking expenses.
IGNORE_CATEGORIES = [
    "Operating Expense",
    "Operating Expenses",
    "Total Operating Expense",
    "Total Operating Expenses",
    "Total Expenses",
    "Operating Income",
]
