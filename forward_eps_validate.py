"""Validation gate: only display a bottom-up number if coverage is high enough and it
lands near published consensus. Benchmarks are coarse anchors, updated occasionally."""
_MIN_COVERAGE = 0.85
# Published-consensus anchors (current-FY YoY EPS growth), refresh periodically.
# SPY now uses FactSet directly (this gate only guards the SPY bottom-up fallback),
# so its band stays tight. QQQ has no authoritative free consensus and its real
# growth swings widely (Q1'26 +46% decelerating to ~+18% H1), so its band is wide:
# coverage (>=85%) is the real quality bar; the band only rejects grossly broken
# output (e.g. a scaling bug). Refresh the anchors periodically.
_BENCH = {"SPY": 0.24, "QQQ": 0.22}
_TOL = {"SPY": 0.08, "QQQ": 0.15}   # +/- percentage points on the growth rate
_DEFAULT_TOL = 0.08


def is_displayable(index_name, growth_this_fy, coverage_weight) -> bool:
    if growth_this_fy is None or coverage_weight is None:
        return False
    if coverage_weight < _MIN_COVERAGE:
        return False
    bench = _BENCH.get(index_name.upper())
    if bench is not None:
        tol = _TOL.get(index_name.upper(), _DEFAULT_TOL)
        if abs(growth_this_fy - bench) > tol:
            return False
    return True
