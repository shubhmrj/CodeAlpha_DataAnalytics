from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Make sure VADER lexicon is available. Download once if missing.
# try:
#     _ = nltk.data.find("sentiment/vader_lexicon.zip")
# except LookupError:  # pragma: no cover – download only first time
#     nltk.download("vader_lexicon")

from nrclex import NRCLex


# ----------------------------- Thresholds & constants ----------------------------- #
SENTIMENT_THRESHOLD_POS = 0.05  # compound >= 0.05 –> positive
SENTIMENT_THRESHOLD_NEG = -0.05  # compound <= -0.05 –> negative


class SentimentAnalyzer:
    """Encapsulates sentiment & emotion analysis utilities."""

    def __init__(self):
        self._sia = SentimentIntensityAnalyzer()

    def classify_sentiment(self, text: str) -> str:
        """Return **positive**, **negative**, or **neutral** for *text*."""
        score = self._sia.polarity_scores(text)["compound"]
        if score >= SENTIMENT_THRESHOLD_POS:
            return "positive"
        if score <= SENTIMENT_THRESHOLD_NEG:
            return "negative"
        return "neutral"

    @staticmethod
    def detect_emotions(text: str) -> List[str]:
        """Return list of dominant emotions detected in *text*."""
        emotion_obj = NRCLex(text)
        return [e for e, v in emotion_obj.raw_emotion_scores.items() if v > 0]

    # ------------------------------------------------------------------
    # Bulk / DataFrame helpers
    # ------------------------------------------------------------------
    def analyse_dataframe(
        self, df: pd.DataFrame, text_col: str = "text"
    ) -> Tuple[pd.Series, pd.Series]:
        """Add *sentiment* & *emotions* columns to *df* and return value counts."""
        df = df.copy()
        df["sentiment"] = df[text_col].apply(self.classify_sentiment)
        df["emotions"] = df[text_col].apply(
            lambda t: ",".join(self.detect_emotions(t))
        )

        sentiment_counts = df["sentiment"].value_counts().sort_index()
        emotion_series = df["emotions"].str.split(",").explode()
        emotion_counts = emotion_series.value_counts()
        return sentiment_counts, emotion_counts


# ------------------------------------------------------------------------------
# Visualisation helper – kept here so GUI & CLI can share logic
# ------------------------------------------------------------------------------

def plot_bar(series: pd.Series, title: str, filename: str | Path) -> Path:
    """Create and save a simple bar plot for *series* and return path."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=series.index, y=series.values, palette="viridis")
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    filename = Path(filename)
    plt.savefig(filename)
    plt.close()
    return filename
