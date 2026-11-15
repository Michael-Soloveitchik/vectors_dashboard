"""Real-data callbacks: language/metadata filters, stats panel, wav player,
cross-lingual speaker-link highlighting, and embedding model switcher.

These callbacks operate on the full df passed in at boot time.
They update store_filtered_df (list of session_ids that pass the active
filters) which downstream callbacks already consume via store_aggregated.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, html, no_update

from ..debug_log import trace
from ..ids import IDS

# Columns that each filter maps to (col in df may be 'language' OR 'file_language')
_LANG_COLS = ["language", "file_language"]


def _apply_filter(df: pd.DataFrame, col: str, values: list[str]) -> pd.Series:
    """Return boolean mask: True = keep. Empty values = keep all."""
    if not values:
        return pd.Series(True, index=df.index)
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].astype(str).isin(values)


def _wav_player_div(session_id: str, wav_dir: Path,
                    max_sec: float = 5.0) -> html.Div:
    """Return an html.Div with an inline audio player + mini waveform."""
    wav_p = wav_dir / f"{session_id}.wav"
    if not wav_p.exists():
        return html.Div(
            f"No wav found: {session_id}.wav",
            style={"fontSize": "11px", "color": "#f0a060"},
        )

    try:
        import soundfile as sf
        sig, sr = sf.read(str(wav_p), dtype="float32", always_2d=False)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        # Trim to max_sec
        n = min(len(sig), int(max_sec * sr))
        sig = sig[:n]
    except Exception as exc:
        return html.Div(f"Error reading wav: {exc}",
                        style={"fontSize": "11px", "color": "#f0a060"})

    # Encode full clip as base64 data URI for inline playback
    try:
        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, sig, sr, format="WAV")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        audio_uri = f"data:audio/wav;base64,{b64}"
    except Exception as exc:
        audio_uri = None

    # Mini waveform via Plotly — import inline to keep module importable
    try:
        import plotly.graph_objects as go
        from dash import dcc
        # Downsample for display
        step = max(1, len(sig) // 400)
        t = np.arange(0, len(sig), step) / sr
        amp = sig[::step]
        fig = go.Figure(go.Scatter(
            x=t, y=amp, mode="lines",
            line=dict(color="#4ecdc4", width=1),
        ))
        fig.update_layout(
            height=80, margin=dict(l=4, r=4, t=4, b=4),
            template="plotly_dark", paper_bgcolor="#0f1115",
            plot_bgcolor="#0f1115",
            xaxis=dict(showgrid=False, zeroline=False,
                       title="sec", tickfont=dict(size=9)),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False),
        )
        waveform = dcc.Graph(figure=fig,
                             config={"displayModeBar": False},
                             style={"height": "80px"})
    except Exception:
        waveform = html.Div()

    children = [
        html.Div(session_id,
                 style={"fontSize": "10px", "color": "#8a8f99",
                        "marginBottom": "4px", "wordBreak": "break-all"}),
        waveform,
    ]
    if audio_uri:
        children.append(
            html.Audio(src=audio_uri, controls=True,
                       style={"width": "100%", "marginTop": "4px"})
        )
    return html.Div(children)


def register(app, df: pd.DataFrame) -> None:
    # Resolve wav dir relative to the real_data module location
    try:
        from data.real_data import WAVS_DIR
        _wav_dir = WAVS_DIR
    except Exception:
        _wav_dir = Path("/home/michael/Research/Projects/RD/Audio_and_Acoustic/Data/Audio/wavs")

    # ── filter → store_filtered_df ────────────────────────────────────────────
    @app.callback(
        Output(IDS.store_filtered_df, "data"),
        Input(IDS.ctrl_lang_filter, "value"),
        Input(IDS.ctrl_dialect_filter, "value"),
        Input(IDS.ctrl_gender_filter, "value"),
        Input(IDS.ctrl_recording_type_filter, "value"),
        Input(IDS.ctrl_noise_level_filter, "value"),
        Input(IDS.ctrl_age_bucket_filter, "value"),
        Input(IDS.ctrl_distance_bucket_filter, "value"),
        Input(IDS.ctrl_vocal_effort_filter, "value"),
        Input(IDS.ctrl_is_simulated_filter, "value"),
        prevent_initial_call=False,
    )
    def _apply_filters(lang, dialect, gender, rec_type, noise, age, dist,
                       vocal, is_sim):
        mask = pd.Series(True, index=df.index)

        # Language: check both 'language' and 'file_language' columns
        if lang:
            lang_mask = pd.Series(False, index=df.index)
            for col in _LANG_COLS:
                if col in df.columns:
                    lang_mask |= df[col].astype(str).isin(lang)
            mask &= lang_mask

        mask &= _apply_filter(df, "dialect", dialect or [])
        mask &= _apply_filter(df, "gender", gender or [])
        mask &= _apply_filter(df, "recording_type", rec_type or [])
        mask &= _apply_filter(df, "noise_level", noise or [])
        mask &= _apply_filter(df, "age_bucket", age or [])
        mask &= _apply_filter(df, "distance_bucket", dist or [])
        mask &= _apply_filter(df, "vocal_effort", vocal or [])

        if is_sim:
            is_sim_mask = pd.Series(False, index=df.index)
            if "is_simulated" in df.columns:
                for v in is_sim:
                    bool_v = (v.lower() == "true")
                    is_sim_mask |= (df["is_simulated"].astype(bool) == bool_v)
            mask &= is_sim_mask

        kept = df.loc[mask, "session_id"].astype(str).tolist()
        return kept

    # ── stats panel ───────────────────────────────────────────────────────────
    @app.callback(
        Output(IDS.stats_panel, "children"),
        Input(IDS.store_filtered_df, "data"),
    )
    def _update_stats(session_ids):
        if not session_ids:
            # No filter applied = show global stats
            sub = df
        else:
            sub = df[df["session_id"].astype(str).isin(set(session_ids))]

        n_rows = len(sub)
        n_speakers = sub["speaker_id"].nunique() if "speaker_id" in sub.columns else "?"

        # Language distribution
        lang_col = "language" if "language" in sub.columns else "file_language"
        if lang_col in sub.columns:
            lang_dist = sub[lang_col].value_counts().head(6)
            lang_str = "  ".join(f"{l}:{c}" for l, c in lang_dist.items())
        else:
            lang_str = "—"

        # Gender distribution
        if "gender" in sub.columns:
            gd = sub["gender"].value_counts()
            gender_str = "  ".join(f"{g}:{c}" for g, c in gd.items())
        else:
            gender_str = "—"

        # Mean SNR
        if "snr_estimate_db" in sub.columns:
            snr_vals = pd.to_numeric(sub["snr_estimate_db"], errors="coerce").dropna()
            snr_str = f"{snr_vals.mean():.1f} dB" if len(snr_vals) else "—"
        else:
            snr_str = "—"

        lines = [
            f"Rows: {n_rows:,}",
            f"Unique speakers: {n_speakers:,}",
            f"Languages: {lang_str}",
            f"Gender: {gender_str}",
            f"Mean SNR: {snr_str}",
        ]
        return [html.Div(line) for line in lines]

    # ── wav player + cross-lingual speaker links ──────────────────────────────
    @app.callback(
        Output(IDS.wav_player_panel, "children"),
        Output(IDS.store_aggregated, "data", allow_duplicate=True),
        Input(IDS.fig_pca, "clickData"),
        State(IDS.ctrl_model, "value"),
        State(IDS.store_aggregated, "data"),
        prevent_initial_call=True,
    )
    @trace
    def _on_point_click(click_data, model, prev_agg):
        if not click_data or not click_data.get("points"):
            return no_update, no_update
        pt = click_data["points"][0]
        cd = pt.get("customdata")
        if not (isinstance(cd, list) and len(cd) >= 2):
            return no_update, no_update

        session_id = str(cd[1])

        # Wav player
        player = _wav_player_div(session_id, _wav_dir)

        # Cross-lingual speaker linking: highlight all sessions with same speaker_id
        if "speaker_id" in df.columns:
            match = df[df["session_id"].astype(str) == session_id]
            if not match.empty:
                spk = str(match.iloc[0]["speaker_id"])
                # Filter to current model if model_name column exists
                df_scope = df
                if model and "model_name" in df.columns:
                    df_scope = df[df["model_name"] == model]
                linked = df_scope[df_scope["speaker_id"].astype(str) == spk]
                agg = linked["session_id"].astype(str).tolist()
            else:
                agg = prev_agg or []
        else:
            agg = prev_agg or []

        return player, agg

    # ── color-by propagation to labels ctrl ──────────────────────────────────
    @app.callback(
        Output(IDS.ctrl_labels, "value", allow_duplicate=True),
        Input(IDS.ctrl_color_by, "value"),
        prevent_initial_call=True,
    )
    def _color_by_to_labels(color_by):
        if not color_by:
            return no_update
        return [color_by]

    # ── filter → aggregation sync ─────────────────────────────────────────────
    # When filters narrow the visible set, push filtered session_ids into
    # store_aggregated so PCA scatter highlights them.
    @app.callback(
        Output(IDS.store_aggregated, "data", allow_duplicate=True),
        Input(IDS.store_filtered_df, "data"),
        State(IDS.store_aggregated, "data"),
        prevent_initial_call=True,
    )
    def _filter_to_agg(filtered_ids, prev_agg):
        # Only push when filters are actually active (non-empty)
        if filtered_ids is None:
            return no_update
        # Empty list = no filter active — clear any filter-driven selection
        # but don't clobber manual lasso selections (heuristic: if
        # prev_agg is very large it was from a filter, not a lasso)
        if not filtered_ids:
            return []
        return filtered_ids


__all__ = ["register"]
