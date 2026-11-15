"""Component IDs — single source of truth."""
from types import SimpleNamespace

IDS = SimpleNamespace(
    # stores
    store_fit="store-fit",            # {"spec", "labels", "error", "token"}
    store_thr="store-thr",            # {"token", "values": {label: thr}}  (normalized)
    store_sel="store-sel",            # {"model", "clauses": {src: …}, "mode", "seq"}
    store_focus="store-focus",        # {"model", "rows" | "labels"} | None
    store_theme="store-theme",        # "dark" | "light"
    store_per_class_n="store-per-class-n",
    store_client="store-client",      # random id per browser tab (latest-wins)
    # transactions (callbacks/panels.py): wanted vs rendered state per panel
    store_want="store-want", store_kick="store-kick", tick="tick", sync="sync",
    store_late="store-late", drawn_scatter="drawn-scatter",
    sig_scatter="sig-scatter", sig_roc="sig-roc", sig_dist="sig-dist",
    sig_matrix="sig-matrix",

    # top bar
    summary="summary", btn_theme="btn-theme",

    # sidebar — data / method
    ctrl_model="ctrl-model", ctrl_labels="ctrl-labels", ctrl_skip="ctrl-skip",
    ctrl_methodology="ctrl-methodology", ctrl_aggregator="ctrl-aggregator",
    ctrl_n_centroids="ctrl-n-centroids", per_class_n_panel="per-class-n-panel",
    fit_status="fit-status",

    # sidebar — scores / thresholds
    ctrl_alpha="ctrl-alpha", ctrl_thr_all="ctrl-thr-all",
    btn_thr_eer="btn-thr-eer", btn_thr_reset="btn-thr-reset",
    btn_thr_dcf="btn-thr-dcf",
    thr_pick="thr-pick",                  # {"label", "thr", "seq"} from a plot click
    threshold_panel="threshold-panel",

    # sidebar — selection / matrix colour / export
    badge_selection="badge-selection", btn_sel_clear="btn-sel-clear",
    ctrl_sel_classes="ctrl-sel-classes", sel_summary="sel-summary",
    ctrl_sel_mode="ctrl-sel-mode",        # single | union | intersect
    sel_list="sel-list",
    # own labels (callbacks/labels.py)
    ctrl_label_name="ctrl-label-name", btn_label_apply="btn-label-apply",
    btn_label_clear="btn-label-clear", label_status="label-status",
    store_labels="store-labels",          # {"model", "rev"}
    # one intent store per selection source (callbacks/selection.py)
    int_lasso="int-lasso", int_classes="int-classes", int_click="int-click",
    int_insp="int-insp", int_reset="int-reset",
    store_mods="store-mods",              # {ctrl, shift} at the last mouse-down
    lasso_echo="lasso-echo",              # bumped when a gesture is ignored
    ctrl_color_auto="ctrl-color-auto",   # toggles: auto range / abs
    ctrl_color_power="ctrl-color-power", ctrl_color_range="ctrl-color-range",
    btn_dump_npz="btn-dump-npz", btn_dump_csv="btn-dump-csv",
    btn_dump_json="btn-dump-json", download="download",

    # projection card
    fig_scatter="fig-scatter", sub_scatter="sub-scatter",
    ctrl_reduction="ctrl-reduction", ctrl_perplexity="ctrl-perplexity",
    ctrl_n_neighbors="ctrl-n-neighbors", ctrl_min_dist="ctrl-min-dist",
    wrap_tsne="wrap-tsne", wrap_umap="wrap-umap",
    ctrl_scatter_opts="ctrl-scatter-opts", ctrl_max_points="ctrl-max-points",
    ctrl_color_by="ctrl-color-by",        # "class" or any label column

    # matrix card
    fig_matrix="fig-matrix", sub_matrix="sub-matrix",
    ctrl_matrix_mode="ctrl-matrix-mode", ctrl_matrix_sort="ctrl-matrix-sort",
    wrap_matrix_sort="wrap-matrix-sort",

    # roc card
    fig_roc="fig-roc", sub_roc="sub-roc", ctrl_curve="ctrl-curve",
    ctrl_roc_show="ctrl-roc-show", ctrl_roc_axes="ctrl-roc-axes",

    # distribution card
    fig_dist="fig-dist", sub_dist="sub-dist", ctrl_dist_view="ctrl-dist-view",

    # inspector drawer
    inspector="inspector", inspector_body="inspector-body",
    inspector_audio="inspector-audio",
    inspector_title="inspector-title", btn_inspector_close="btn-inspector-close",
)
