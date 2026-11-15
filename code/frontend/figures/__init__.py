"""Pure figure builders: (Analysis, view state) → go.Figure. No Dash."""
from .common import FOC, SEL, THR, home, message_fig, sig
from .dist import dist_fig, lane_height
from .matrix import (SEP, TREE_SEP, classes_fig, confusion_fig, tree_fig,
                     vectors_fig)
from .roc import roc_fig
from .scatter import OWN_SCORE, is_continuous, scatter_fig

__all__ = ["FOC", "SEL", "THR", "SEP", "message_fig", "sig", "dist_fig", "lane_height",
           "classes_fig", "confusion_fig", "vectors_fig", "tree_fig", "TREE_SEP", "roc_fig",
           "scatter_fig", "OWN_SCORE", "is_continuous"]
