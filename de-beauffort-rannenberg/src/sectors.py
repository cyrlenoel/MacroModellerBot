"""Named domestic sectors for the two-sector DAG.

The GE blocks are still hard-coded T/NT (see EXTENDING.md). These constants
are the labels those blocks use, so a later N-sector pass has one place to
start from without rewriting the solver in this starter.
"""

TRADABLE = "T"
NONTRADABLE = "NT"
DOMESTIC_SECTORS = (TRADABLE, NONTRADABLE)
