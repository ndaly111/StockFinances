"""Validation gate: only display a bottom-up number if coverage is high enough and it
lands near published consensus. Benchmarks are coarse anchors, updated occasionally."""
_MIN_COVERAGE = 0.85
_TOL = 0.08   # +/- 8 percentage points on the growth rate
# Published consensus anchors (current-FY YoY EPS growth), refresh periodically.
_BENCH = {"SPY": 0.21, "QQQ": 0.19}


def is_displayable(index_name, growth_this_fy, coverage_weight) -> bool:
    if growth_this_fy is None or coverage_weight is None:
        return False
    if coverage_weight < _MIN_COVERAGE:
        return False
    bench = _BENCH.get(index_name.upper())
    if bench is not None and abs(growth_this_fy - bench) > _TOL:
        return False
    return True
