"""Component IDs — single source of truth.

Renaming a control here is enough; every callback follows.
"""
from types import SimpleNamespace

IDS = SimpleNamespace(
    # Stores
    store_centroids="store-centroids",
    store_thresholds="store-thresholds",
    store_view_mode="store-view-mode",
    store_theme="store-theme",
    store_aggregated="store-aggregated",

    # Top-level shell
    shell="shell",
    header="header",
    btn_theme="btn-theme",
    side_col="side-col",
    main_col="main-col",

    # Controls
    ctrl_model="ctrl-model",
    ctrl_methodology="ctrl-methodology",
    ctrl_aggregator="ctrl-aggregator",
    ctrl_n_centroids="ctrl-n-centroids",         # global int — multi-centroid
    ctrl_centroid_approach="ctrl-centroid-approach",  # auto/kmeans/...
    per_class_n_panel="per-class-n-panel",       # injected per-class inputs
    store_per_class_n="store-per-class-n",       # {label: int}
    ctrl_reduction="ctrl-reduction",             # PCA | t-SNE | UMAP
    ctrl_perplexity="ctrl-perplexity",
    ctrl_n_neighbors="ctrl-n-neighbors",
    ctrl_min_dist="ctrl-min-dist",
    ctrl_density_contour="ctrl-density-contour", # toggle on PCA scatter
    ctrl_curve_mode="ctrl-curve-mode",           # ROC | DET
    ctrl_label_allow="ctrl-label-allow",         # checklist of label values
    btn_reset_labels="btn-reset-labels",
    btn_dump_centroids_json="btn-dump-centroids-json",
    download_centroids_json="download-centroids-json",
    ctrl_labels="ctrl-labels",
    ctrl_view_mode="ctrl-view-mode",      # "vectors" button (primary)
    btn_view_centroids="btn-view-centroids",  # "centroids" button
    ctrl_color_abs="ctrl-color-abs",
    ctrl_color_power="ctrl-color-power",
    ctrl_color_range="ctrl-color-range",
    threshold_panel="threshold-panel",    # injected per-class sliders
    btn_reset_aggregation="btn-reset-aggregation",
    badge_selection="badge-selection",

    # Score domain + per-panel toggles
    ctrl_score_domain="ctrl-score-domain",     # raw | normalized
    ctrl_dist_view="ctrl-dist-view",           # box | hist
    ctrl_roc_log_x="ctrl-roc-log-x",
    ctrl_roc_log_y="ctrl-roc-log-y",
    ctrl_roc_mode="ctrl-roc-mode",             # all | per-class | pair
    ctrl_skip_classes="ctrl-skip-classes",     # exclude labels from fit
    ctrl_pca_zoom="ctrl-pca-zoom",             # zoom factor float
    ctrl_show_ellipses="ctrl-show-ellipses",   # covariance ellipses on PCA
    ctrl_ellipse_n_sigma="ctrl-ellipse-n-sigma",
    ctrl_pair_cos_thr="ctrl-pair-cos-thr",
    ctrl_pair_norm_thr="ctrl-pair-norm-thr",
    ctrl_alpha="ctrl-alpha",                   # NORM_ALPHA blend [0,1]

    # Matrix interactions
    btn_view_mode_matrix="btn-view-mode-matrix",   # toggle on matrix header
    ctrl_matrix_sort="ctrl-matrix-sort",           # order keys for cm rows
    store_cm_click_pair="store-cm-click-pair",     # (row_id, col_id) on click

    # Real-data controls
    ctrl_emb_model="ctrl-emb-model",           # embedding model switcher
    ctrl_lang_filter="ctrl-lang-filter",       # language multi-select
    ctrl_dialect_filter="ctrl-dialect-filter",
    ctrl_gender_filter="ctrl-gender-filter",
    ctrl_recording_type_filter="ctrl-recording-type-filter",
    ctrl_noise_level_filter="ctrl-noise-level-filter",
    ctrl_age_bucket_filter="ctrl-age-bucket-filter",
    ctrl_distance_bucket_filter="ctrl-distance-bucket-filter",
    ctrl_vocal_effort_filter="ctrl-vocal-effort-filter",
    ctrl_is_simulated_filter="ctrl-is-simulated-filter",
    ctrl_color_by="ctrl-color-by",             # color scatter by column
    store_filtered_df="store-filtered-df",     # JSON index of kept rows
    stats_panel="stats-panel",                 # summary stats div
    wav_player_panel="wav-player-panel",       # audio player div
    store_clicked_session="store-clicked-session",  # clicked point session_id
    store_emb_model="store-emb-model",         # currently loaded emb model name

    # Dump
    btn_dump_npz="btn-dump-npz",
    btn_dump_csv="btn-dump-csv",
    download_npz="download-npz",
    download_csv="download-csv",

    # Subplots
    fig_pca="fig-pca",
    fig_cm="fig-cm",
    fig_roc="fig-roc",
    fig_dist="fig-dist",
)
