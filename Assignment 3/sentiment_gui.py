from __future__ import annotations

import threading
from pathlib import Path
from tkinter import Tkinter ,  Screen
#     BOTH,
#     BOTTOM,
#     END,
#     LEFT,
#     RIGHT,
#     VERTICAL,
#     Button,
#     Entry,
#     Frame,
#     Label,
#     Listbox,
#     PanedWindow,
#     Scrollbar,
#     Text,
#     Tk,
#     filedialog,
#     messagebox,
# )
from typing import List

import pandas as pd

from sentiment_core import SentimentAnalyzer, plot_bar


def run_in_thread(func):
    """Decorator to run blocking function *func* in background thread."""

    def _wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()

    return _wrapper


class SentimentApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment & Emotion Analyzer")
        self.geometry("800x600")

        # Core analyzer instance
        self._analyzer = SentimentAnalyzer()

        # ------------------------------------------------------------------
        # Layout – use top paned window: left = inputs, right = outputs
        # ------------------------------------------------------------------
        pw = PanedWindow(self, orient=VERTICAL)
        pw.pack(fill=BOTH, expand=True)

        # --------------------------- Text analysis pane -------------------
        text_frame = Frame(pw, padx=10, pady=10)
        pw.add(text_frame, stretch="always")

        Label(text_frame, text="Enter text for analysis:").pack(anchor="w")
        self.txt_input = Text(text_frame, height=4)
        self.txt_input.pack(fill=BOTH, expand=False)

        Button(text_frame, text="Analyze Text", command=self._on_analyse_text).pack(
            pady=5
        )

        # --------------------------- CSV analysis pane --------------------
        csv_frame = Frame(pw, padx=10, pady=10)
        pw.add(csv_frame, stretch="always")

        Button(csv_frame, text="Select CSV File", command=self._select_csv).pack(
            anchor="w"
        )
        self.lbl_file = Label(csv_frame, text="No file selected")
        self.lbl_file.pack(anchor="w")

        Button(csv_frame, text="Analyze CSV", command=self._on_analyse_csv).pack(
            pady=5, anchor="w"
        )

        # --------------------------- Results pane -------------------------
        res_frame = Frame(pw, padx=10, pady=10)
        pw.add(res_frame, stretch="always")

        Label(res_frame, text="Results:").pack(anchor="w")

        # Scrollable listbox to show output lines
        list_frame = Frame(res_frame)
        list_frame.pack(fill=BOTH, expand=True)

        scrollbar = Scrollbar(list_frame, orient=VERTICAL)
        self.listbox = Listbox(list_frame, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        self.listbox.pack(side=LEFT, fill=BOTH, expand=True)

        # Store path of last analysis plots (if any)
        self._last_plot_paths: List[Path] = []

    # ----------------------------------------------------------------------
    # GUI callbacks
    # ----------------------------------------------------------------------

    @run_in_thread
    def _on_analyse_text(self):
        text = self.txt_input.get("1.0", END).strip()
        if not text:
            tkinter.messagebox.showwarning("Input needed", "Please enter some text.")
            return

        sentiment = self._analyzer.classify_sentiment(text)
        emotions = self._analyzer.detect_emotions(text)
        self._display_results([f"Sentiment: {sentiment}", f"Emotions: {', '.join(emotions) or 'none'}"])

    def _select_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.lbl_file["text"] = file_path

    @run_in_thread
    def _on_analyse_csv(self):
        path_str = self.lbl_file["text"]
        file_path = Path(path_str)
        if not file_path.exists():
            messagebox.showerror("File missing", "Please select a valid CSV file first.")
            return

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Read error", f"Failed to read CSV: {exc}")
            return

        # Detect possible text column names (case-insensitive)
        possible_cols = [c for c in df.columns if c.lower() in {"text", "review", "comment"}]
        if not possible_cols:
            messagebox.showerror(
                "Missing column",
                "CSV must contain a column like 'text', 'review', or 'comment' for analysis.",
            )
            return

        text_col = possible_cols[0]
        if text_col != "text":
            df = df.rename(columns={text_col: "text"})

        sentiment_counts, emotion_counts = self._analyzer.analyse_dataframe(df, text_col="text")

        # Prepare output lines
        lines = ["Sentiment distribution:"] + [f"  {i}: {v}" for i, v in sentiment_counts.items()]
        lines += ["Top emotions (top 10):"] + [
            f"  {i}: {v}" for i, v in emotion_counts.head(10).items()
        ]
        self._display_results(lines)

        # Generate plots
        self._last_plot_paths.clear()
        self._last_plot_paths.append(plot_bar(sentiment_counts, "Sentiment Distribution", file_path.with_suffix("_sentiment.png")))
        self._last_plot_paths.append(plot_bar(emotion_counts.head(10), "Top Emotions (Top 10)", file_path.with_suffix("_emotions.png")))

        self._display_results([f"Plots saved to same folder as CSV."])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _display_results(self, lines):
        self.listbox.delete(0, END)
        for line in lines:
            self.listbox.insert(END, line)


def launch_gui():
    app = SentimentApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
