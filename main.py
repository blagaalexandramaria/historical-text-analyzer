"""
Main GUI application for the Historical Text Analyzer.

Responsibilities:
- file selection
- running analysis in background
- rendering results in tabs
- highlighting important terms
- displaying charts
"""

import os
import re
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from text_processing import read_text_file
from analysis_service import analyze_raw_texts


# Supported file formats for analysis
SUPPORTED_TYPES = (
    ("Text files", "*.txt"),
    ("Word documents", "*.docx"),
    ("PDF files", "*.pdf"),
    ("All files", "*.*"),
)


class HistoricalTextAnalyzerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Historical Text Similarity Analyzer")
        self.root.geometry("900x780")
        self.root.resizable(True, True)

        # Store selected files
        self.selected_file_1 = ""
        self.selected_file_2 = ""

        # Keep last analysis results for charts/highlighting reuse
        self.last_results: dict[str, object] | None = None
        self.last_term_groups: dict[str, list[str]] | None = None

        # Queue used to transfer results from worker thread to GUI thread
        self.analysis_queue: queue.Queue[dict[str, object]] = queue.Queue()
        self.is_analyzing = False

        self.build_ui()

    def build_ui(self) -> None:
        """Builds the main graphical interface."""
        title = tk.Label(
            self.root,
            text="Historical Text Similarity Analyzer",
            font=("Arial", 16, "bold"),
        )
        title.pack(pady=10)

        instruction = tk.Label(
            self.root,
            text="Select two files (.txt, .docx, .pdf) to compare.",
        )
        instruction.pack(pady=5)

        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=8)

        file1_button = tk.Button(
            file_frame,
            text="Select file 1",
            command=self.select_file_1,
            width=15,
        )
        file1_button.grid(row=0, column=0, padx=5, pady=4)

        file2_button = tk.Button(
            file_frame,
            text="Select file 2",
            command=self.select_file_2,
            width=15,
        )
        file2_button.grid(row=0, column=1, padx=5, pady=4)

        self.analyze_button = tk.Button(
            file_frame,
            text="Analyze",
            command=self.analyze_files,
            width=15,
        )
        self.analyze_button.grid(row=0, column=2, padx=5, pady=4)

        self.charts_button = tk.Button(
            file_frame,
            text="Charts",
            command=self.show_charts,
            width=12,
        )
        self.charts_button.grid(row=0, column=3, padx=5, pady=4)

        self.file1_label = tk.Label(
            self.root,
            text="File 1: not selected",
            fg="blue",
            wraplength=850,
            justify="left",
        )
        self.file1_label.pack(pady=2)

        self.file2_label = tk.Label(
            self.root,
            text="File 2: not selected",
            fg="blue",
            wraplength=850,
            justify="left",
        )
        self.file2_label.pack(pady=2)

        # Notebook tabs separate results into logical categories
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.output_boxes: dict[str, scrolledtext.ScrolledText] = {}
        for key, title in (
            ("similarity", "Similarity"),
            ("words", "Words"),
            ("themes", "Themes"),
            ("years", "Years"),
            ("propaganda", "Propaganda"),
            ("important", "Important"),
        ):
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=title)
            box = scrolledtext.ScrolledText(
                frame,
                wrap=tk.WORD,
                width=110,
                height=30,
            )
            box.pack(fill=tk.BOTH, expand=True)
            self.output_boxes[key] = box

        self.status_label = tk.Label(
            self.root,
            text="",
            fg="#8c6a3f",
        )
        self.status_label.pack(pady=4)

    def select_file_1(self) -> None:
        """Opens file picker for the first input file."""
        filepath = filedialog.askopenfilename(
            title="Select file 1",
            filetypes=SUPPORTED_TYPES,
        )
        if filepath:
            self.selected_file_1 = filepath
            self.file1_label.config(text=f"File 1: {filepath}")

    def select_file_2(self) -> None:
        """Opens file picker for the second input file."""
        filepath = filedialog.askopenfilename(
            title="Select file 2",
            filetypes=SUPPORTED_TYPES,
        )
        if filepath:
            self.selected_file_2 = filepath
            self.file2_label.config(text=f"File 2: {filepath}")

    def analyze_files(self) -> None:
        """
        Validates selected files and starts analysis in a background thread.

        Threading is used so the GUI does not freeze during processing.
        """
        if self.is_analyzing:
            return

        if not self.selected_file_1 or not self.selected_file_2:
            messagebox.showerror("Missing files", "Please select two files.")
            return

        if os.path.abspath(self.selected_file_1) == os.path.abspath(self.selected_file_2):
            messagebox.showerror("Same file", "Please select two different files.")
            return

        self.is_analyzing = True
        self.status_label.config(text="Analyzing... please wait.")
        self.analyze_button.config(state=tk.DISABLED)
        self.charts_button.config(state=tk.DISABLED)

        worker = threading.Thread(
            target=self._analyze_worker,
            args=(self.selected_file_1, self.selected_file_2),
            daemon=True,
        )
        worker.start()

        # Periodically check if background analysis has finished
        self.root.after(100, self._check_analysis_queue)

    def _analyze_worker(self, file1: str, file2: str) -> None:
        """
        Worker thread that:
        - reads files
        - runs NLP analysis
        - prepares term groups for highlighting
        - pushes results back into queue
        """
        try:
            raw_text1 = read_text_file(file1)
            raw_text2 = read_text_file(file2)
            analysis = analyze_raw_texts(raw_text1, raw_text2)

            term_groups = self.collect_important_terms(
                analysis["top1"],
                analysis["top2"],
                analysis["top_common"],
                analysis["classification1"]["top_terms"],
                analysis["classification2"]["top_terms"],
            )

            payload = {
                "raw_text1": raw_text1,
                "raw_text2": raw_text2,
                "term_groups": term_groups,
                **analysis,
            }
            self.analysis_queue.put(payload)
        except Exception as exc:
            self.analysis_queue.put({"error": str(exc)})

    def _check_analysis_queue(self) -> None:
        """
        Checks whether the background thread has produced results.

        Keeps polling until data appears in the queue.
        """
        try:
            result = self.analysis_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._check_analysis_queue)
            return

        self.is_analyzing = False
        self.status_label.config(text="")
        self.analyze_button.config(state=tk.NORMAL)
        self.charts_button.config(state=tk.NORMAL)

        if "error" in result:
            messagebox.showerror("Read error", str(result["error"]))
            return

        self._render_results(result)

    def _render_results(self, data: dict[str, object]) -> None:
        """
        Writes analysis results into GUI tabs.

        Also stores reduced results for chart generation
        and launches highlighting views.
        """
        raw_text1 = data["raw_text1"]  # type: ignore[assignment]
        raw_text2 = data["raw_text2"]  # type: ignore[assignment]
        similarity = data["similarity"]  # type: ignore[assignment]
        tfidf_similarity = data["tfidf_similarity"]  # type: ignore[assignment]
        common = data["common"]  # type: ignore[assignment]
        top1 = data["top1"]  # type: ignore[assignment]
        top2 = data["top2"]  # type: ignore[assignment]
        top_common = data["top_common"]  # type: ignore[assignment]
        unique1 = data["unique1"]  # type: ignore[assignment]
        unique2 = data["unique2"]  # type: ignore[assignment]
        years1 = data["years1"]  # type: ignore[assignment]
        years2 = data["years2"]  # type: ignore[assignment]
        themes1 = data["themes1"]  # type: ignore[assignment]
        themes2 = data["themes2"]  # type: ignore[assignment]
        classification1 = data["classification1"]  # type: ignore[assignment]
        classification2 = data["classification2"]  # type: ignore[assignment]
        term_groups = data["term_groups"]  # type: ignore[assignment]

        # Store a smaller subset of results for charts window
        self.last_results = {
            "top1": top1,
            "top2": top2,
            "top_common": top_common,
            "themes1": themes1,
            "themes2": themes2,
            "similarity": similarity,
        }
        self.last_term_groups = term_groups

        for box in self.output_boxes.values():
            box.delete("1.0", tk.END)

        def write_to(key: str, line: str = "") -> None:
            """Small helper to write text into the correct result tab."""
            self.output_boxes[key].insert(tk.END, line + "\n")

        write_to("similarity", "Jaccard similarity: %.2f%%" % similarity)
        write_to("similarity", "TF-IDF cosine similarity: %.2f%%" % tfidf_similarity)

        write_to("words", "=== Common Words ===")
        write_to("words", f"Count: {len(common)}")
        if common:
            preview = ", ".join(common[:50])
            write_to("words", f"Preview (first 50): {preview}")
        write_to("words", "")

        write_to("words", "=== Top Words (File 1) ===")
        for word, freq in top1:
            write_to("words", f"{word}: {freq}")
        write_to("words", "")

        write_to("words", "=== Top Words (File 2) ===")
        for word, freq in top2:
            write_to("words", f"{word}: {freq}")
        write_to("words", "")

        write_to("words", "=== Top Common Words ===")
        for word, freq in top_common:
            write_to("words", f"{word}: {freq}")
        write_to("words", "")

        write_to("words", "=== Unique Words (File 1) ===")
        for word, freq in unique1:
            write_to("words", f"{word}: {freq}")
        write_to("words", "")

        write_to("words", "=== Unique Words (File 2) ===")
        for word, freq in unique2:
            write_to("words", f"{word}: {freq}")

        write_to("years", "=== Years Mentioned (File 1) ===")
        for year, freq in years1[:15]:
            write_to("years", f"{year}: {freq}")
        write_to("years", "")

        write_to("years", "=== Years Mentioned (File 2) ===")
        for year, freq in years2[:15]:
            write_to("years", f"{year}: {freq}")

        write_to("themes", "=== Theme Scores (File 1) ===")
        for theme, score in sorted(themes1.items(), key=lambda item: item[1], reverse=True):
            write_to("themes", f"{theme}: {score}")
        write_to("themes", "")

        write_to("themes", "=== Theme Scores (File 2) ===")
        for theme, score in sorted(themes2.items(), key=lambda item: item[1], reverse=True):
            write_to("themes", f"{theme}: {score}")

        write_to("important", "=== Important Terms (Highlighted) ===")
        if term_groups["common"] or term_groups["propaganda"]:
            if term_groups["common"]:
                write_to("important", "Common terms: " + ", ".join(term_groups["common"]))
            if term_groups["propaganda"]:
                write_to("important", "Propaganda terms: " + ", ".join(term_groups["propaganda"]))
        else:
            write_to("important", "No important terms found.")

        write_to("propaganda", "=== Propaganda Classification (File 1) ===")
        write_to("propaganda", f"Label: {classification1['label']}")
        write_to(
            "propaganda",
            "Propaganda terms: "
            f"{classification1['propaganda_count']} | "
            f"Neutral terms: {classification1['neutral_count']}",
        )
        write_to(
            "propaganda",
            "Ratio (propaganda/neutral): "
            f"{classification1['ratio']:.2f} | "
            f"Density: {classification1['density']:.3f}",
        )
        if classification1["top_terms"]:
            terms_preview = ", ".join(
                f"{word}({count})" for word, count in classification1["top_terms"]
            )
            write_to("propaganda", f"Top propaganda terms: {terms_preview}")

        write_to("propaganda", "")
        write_to("propaganda", "=== Propaganda Classification (File 2) ===")
        write_to("propaganda", f"Label: {classification2['label']}")
        write_to(
            "propaganda",
            "Propaganda terms: "
            f"{classification2['propaganda_count']} | "
            f"Neutral terms: {classification2['neutral_count']}",
        )
        write_to(
            "propaganda",
            "Ratio (propaganda/neutral): "
            f"{classification2['ratio']:.2f} | "
            f"Density: {classification2['density']:.3f}",
        )
        if classification2["top_terms"]:
            terms_preview = ", ".join(
                f"{word}({count})" for word, count in classification2["top_terms"]
            )
            write_to("propaganda", f"Top propaganda terms: {terms_preview}")

        # Apply term highlighting in the words tab
        self.highlight_terms(term_groups)

        # Open an additional window with highlighted raw texts
        self.show_highlighted_texts(raw_text1, raw_text2, term_groups)

    def show_charts(self) -> None:
        """
        Displays charts in a separate window using matplotlib.

        Includes:
        - top terms per file
        - top common terms
        - theme comparison
        - similarity percentage
        """
        if not self.last_results:
            messagebox.showinfo("No data", "Run analysis first to generate charts.")
            return

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except Exception:
            messagebox.showerror(
                "Missing dependency",
                "Charts require matplotlib. Install with: pip install matplotlib",
            )
            return

        chart_window = tk.Toplevel(self.root)
        chart_window.title("Analysis Charts")
        chart_window.geometry("1000x760")

        palette = {
            "blue": "#4e79a7",
            "green": "#59a14f",
            "red": "#e15759",
            "orange": "#f28e2b",
            "purple": "#af7aa1",
            "gray": "#bab0ab",
        }

        figure = Figure(figsize=(9.5, 6.8), dpi=100)
        ax1 = figure.add_subplot(221)
        ax2 = figure.add_subplot(222)
        ax3 = figure.add_subplot(223)
        ax4 = figure.add_subplot(224)

        top1 = self.last_results["top1"]  # type: ignore[assignment]
        top2 = self.last_results["top2"]  # type: ignore[assignment]
        top_common = self.last_results["top_common"]  # type: ignore[assignment]
        themes1 = self.last_results["themes1"]  # type: ignore[assignment]
        themes2 = self.last_results["themes2"]  # type: ignore[assignment]
        similarity = self.last_results["similarity"]  # type: ignore[assignment]

        def barh(ax, data: list[tuple[str, int]], title: str) -> None:
            """Helper function for horizontal bar charts."""
            labels = [item[0] for item in data][::-1]
            values = [item[1] for item in data][::-1]
            ax.barh(labels, values, color=palette["blue"])
            ax.set_title(title)
            ax.tick_params(axis="y", labelsize=8)

        barh(ax1, top1, "Top Terms (File 1)")
        barh(ax2, top2, "Top Terms (File 2)")
        barh(ax3, top_common, "Top Common Terms")

        # Compare theme scores side by side for the two texts
        theme_labels = list(themes1.keys())
        values1 = [themes1[key] for key in theme_labels]
        values2 = [themes2[key] for key in theme_labels]
        x = range(len(theme_labels))
        ax4.bar([i - 0.2 for i in x], values1, width=0.4, label="File 1", color=palette["green"])
        ax4.bar([i + 0.2 for i in x], values2, width=0.4, label="File 2", color=palette["red"])
        ax4.set_title("Theme Scores")
        ax4.set_xticks(list(x))
        ax4.set_xticklabels(theme_labels, rotation=30, ha="right", fontsize=8)
        ax4.legend(fontsize=8)

        figure.tight_layout()

        tabs = ttk.Notebook(chart_window)
        tabs.pack(fill=tk.BOTH, expand=True)

        tab_charts = ttk.Frame(tabs)
        tab_similarity = ttk.Frame(tabs)
        tabs.add(tab_charts, text="Charts")
        tabs.add(tab_similarity, text="Similarity")

        charts_canvas = FigureCanvasTkAgg(figure, master=tab_charts)
        charts_canvas.draw()
        charts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        sim_figure = Figure(figsize=(6, 4), dpi=100)
        sim_ax = sim_figure.add_subplot(111)
        sim_ax.bar(["Similarity"], [similarity], color=palette["orange"])
        sim_ax.set_ylim(0, 100)
        sim_ax.set_title("Similarity (%)")
        sim_ax.bar_label(sim_ax.containers[0], fmt="%.1f%%", fontsize=10)
        sim_figure.tight_layout()

        sim_canvas = FigureCanvasTkAgg(sim_figure, master=tab_similarity)
        sim_canvas.draw()
        sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        button_frame = tk.Frame(chart_window)
        button_frame.pack(fill=tk.X, padx=10, pady=6)

        def save_chart() -> None:
            """Exports generated charts as PNG images."""
            filepath = filedialog.asksaveasfilename(
                title="Save chart image",
                defaultextension=".png",
                filetypes=(("PNG Image", "*.png"), ("All files", "*.*")),
            )
            if filepath:
                figure.savefig(filepath, dpi=200, bbox_inches="tight")
                sim_figure.savefig(filepath.replace(".png", "_similarity.png"), dpi=200, bbox_inches="tight")
                messagebox.showinfo("Saved", f"Charts saved to: {filepath}")

        save_button = tk.Button(
            button_frame,
            text="Save Charts as PNG",
            command=save_chart,
            width=20,
        )
        save_button.pack(side=tk.LEFT)

    def show_highlighted_texts(
        self,
        text1: str,
        text2: str,
        term_groups: dict[str, list[str]],
    ) -> None:
        """
        Opens a new window showing the original texts
        with highlighted important terms.
        """
        window = tk.Toplevel(self.root)
        window.title("Highlighted Texts")
        window.geometry("1000x700")

        tabs = ttk.Notebook(window)
        tabs.pack(fill=tk.BOTH, expand=True)

        tab1 = ttk.Frame(tabs)
        tab2 = ttk.Frame(tabs)
        tabs.add(tab1, text="File 1")
        tabs.add(tab2, text="File 2")

        box1 = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=120, height=40)
        box1.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        box2 = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, width=120, height=40)
        box2.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for box in (box1, box2):
            box.tag_config("highlight_common", background="#bde3ff")
            box.tag_config("highlight_propaganda", background="#ffd6c2")
            box.tag_config("highlight_top", background="#c6f6c6")

        self.insert_with_highlight(box1, text1, term_groups)
        self.insert_with_highlight(box2, text2, term_groups)

    def insert_with_highlight(
        self,
        box: scrolledtext.ScrolledText,
        text: str,
        term_groups: dict[str, list[str]],
    ) -> None:
        """
        Inserts raw text into text box and applies highlights to:
        - common terms
        - propaganda terms
        - top terms
        """
        box.delete("1.0", tk.END)
        common = {t.lower() for t in term_groups.get("common", [])}
        propaganda = {t.lower() for t in term_groups.get("propaganda", [])}
        top_terms = {t.lower() for t in term_groups.get("top", [])}

        # Split text into words, spaces and punctuation
        # so original formatting is preserved during highlighting
        pattern = re.compile(r"\w+|\s+|[^\w\s]")
        for match in pattern.finditer(text):
            token = match.group(0)
            key = token.lower()
            tag = None
            if key and key.isalnum():
                if key in common:
                    tag = "highlight_common"
                elif key in propaganda:
                    tag = "highlight_propaganda"
                elif key in top_terms:
                    tag = "highlight_top"

            if tag:
                box.insert(tk.END, token, tag)
            else:
                box.insert(tk.END, token)

    def collect_important_terms(
        self,
        top1: list[tuple[str, int]],
        top2: list[tuple[str, int]],
        top_common: list[tuple[str, int]],
        propaganda1: list[tuple[str, int]],
        propaganda2: list[tuple[str, int]],
    ) -> dict[str, list[str]]:
        """
        Combines several categories of relevant words into unified term groups.

        Used later for text highlighting.
        """
        common_terms = [word for word, _ in top_common[:10]]
        propaganda_terms = [word for word, _ in propaganda1[:10]] + [
            word for word, _ in propaganda2[:10]
        ]
        top_terms = [word for word, _ in top1[:10]] + [word for word, _ in top2[:10]]

        def uniq(items: list[str]) -> list[str]:
            """Removes duplicates while preserving order."""
            seen: set[str] = set()
            result: list[str] = []
            for item in items:
                if len(item) <= 2:
                    continue
                if item not in seen:
                    result.append(item)
                    seen.add(item)
            return result

        common = uniq(common_terms)
        propaganda = uniq(propaganda_terms)
        top = uniq(top_terms)
        important = uniq(common + propaganda + top)

        return {
            "common": common,
            "propaganda": propaganda,
            "top": top,
            "important": important,
        }

    def highlight_terms(self, term_groups: dict[str, list[str]]) -> None:
        """
        Highlights relevant terms inside the 'Words' output tab.
        """
        self.output_boxes["words"].tag_config(
            "highlight_common",
            foreground="black",
            background="#bde3ff",
        )
        self.output_boxes["words"].tag_config(
            "highlight_propaganda",
            foreground="black",
            background="#ffd6c2",
        )

        for term in term_groups["common"]:
            pattern = r"\m" + re.escape(term) + r"\M"
            start = "1.0"
            while True:
                match_index = self.output_boxes["words"].search(
                    pattern,
                    start,
                    stopindex=tk.END,
                    regexp=True,
                    nocase=True,
                )
                if not match_index:
                    break
                end = f"{match_index}+{len(term)}c"
                self.output_boxes["words"].tag_add("highlight_common", match_index, end)
                start = end

        for term in term_groups["propaganda"]:
            pattern = r"\m" + re.escape(term) + r"\M"
            start = "1.0"
            while True:
                match_index = self.output_boxes["words"].search(
                    pattern,
                    start,
                    stopindex=tk.END,
                    regexp=True,
                    nocase=True,
                )
                if not match_index:
                    break
                end = f"{match_index}+{len(term)}c"
                self.output_boxes["words"].tag_add("highlight_propaganda", match_index, end)
                start = end


def main() -> None:
    """Starts the Tkinter desktop application."""
    root = tk.Tk()
    app = HistoricalTextAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()