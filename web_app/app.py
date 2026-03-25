"""
Flask web application for the Historical Text Analyzer.

Responsibilities:
- upload and validate files
- run NLP analysis on uploaded texts
- cache analysis results
- store display data in session
- render result, propaganda, and chart pages
"""

import os
import sys
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, render_template, request, session


# Add project root to Python path so shared modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_service import analyze_raw_texts
from text_processing import read_text_file


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # limit uploads to 10 MB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Allowed upload file types
ALLOWED_EXTENSIONS = {".txt", ".docx", ".pdf"}


# Small in-memory cache for recent analysis results
_CACHE_LIMIT = 10
_analysis_cache: "OrderedDict[str, dict[str, object]]" = OrderedDict()


def _cache_set(key: str, value: dict[str, object]) -> None:
    """
    Stores an analysis result in cache.

    If cache exceeds limit, removes the oldest entry.
    """
    _analysis_cache[key] = value
    _analysis_cache.move_to_end(key)
    if len(_analysis_cache) > _CACHE_LIMIT:
        _analysis_cache.popitem(last=False)


def _cache_get(key: str) -> dict[str, object] | None:
    """
    Retrieves cached analysis result by key.

    Moves accessed item to the end to mimic LRU behavior.
    """
    if key in _analysis_cache:
        _analysis_cache.move_to_end(key)
        return _analysis_cache[key]
    return None


def allowed_file(filename: str) -> bool:
    """
    Checks whether uploaded file has a supported extension.
    """
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def process_upload(file_storage) -> str:
    """
    Saves uploaded file temporarily, reads its text content,
    then deletes the temporary file.

    This allows reuse of the same text-reading logic used by the desktop app.
    """
    suffix = Path(file_storage.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_storage.save(tmp.name)
        temp_path = tmp.name

    try:
        raw_text = read_text_file(temp_path)
    finally:
        os.remove(temp_path)

    return raw_text


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main page:
    - on GET: shows existing cached/session results if available
    - on POST: handles file upload, runs analysis, prepares paginated output
    """
    result = None
    error = None
    analysis = None

    # Load decorative image list for the landing page
    image_dir = PROJECT_ROOT / "web_app" / "static" / "images"
    if image_dir.exists():
        image_files = sorted(
            f.name for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        image_files = []

    def build_result(
        analysis_data: dict[str, object],
        common_limit: int,
        list_limit: int,
        page: int,
        list_page: int,
    ):
        """
        Builds a paginated result dictionary for templates.

        Splits:
        - common words into pages
        - top/unique/year lists into pages
        """
        common = analysis_data["common"]
        total_common = len(common)
        total_pages = max(1, (total_common + common_limit - 1) // common_limit)
        page = max(1, min(page, total_pages))
        start = (page - 1) * common_limit
        end = start + common_limit

        total_list = len(analysis_data["top1"])
        total_list_pages = max(1, (total_list + list_limit - 1) // list_limit)
        list_page = max(1, min(list_page, total_list_pages))
        list_start = (list_page - 1) * list_limit
        list_end = list_start + list_limit

        return {
            "similarity": f"{analysis_data['similarity']:.2f}%",
            "tfidf_similarity": f"{analysis_data['tfidf_similarity']:.2f}%",
            "common_count": total_common,
            "common_preview": common[start:end],
            "common_preview_limit": common_limit,
            "common_page": page,
            "common_total_pages": total_pages,
            "common_has_prev": page > 1,
            "common_has_next": page < total_pages,
            "list_page": list_page,
            "list_total_pages": total_list_pages,
            "list_has_prev": list_page > 1,
            "list_has_next": list_page < total_list_pages,
            "list_limit": list_limit,
            "top1": analysis_data["top1"][list_start:list_end],
            "top2": analysis_data["top2"][list_start:list_end],
            "top_common": analysis_data["top_common"][list_start:list_end],
            "unique1": analysis_data["unique1"][list_start:list_end],
            "unique2": analysis_data["unique2"][list_start:list_end],
            "years1": analysis_data["years1"][list_start:list_end],
            "years2": analysis_data["years2"][list_start:list_end],
            "themes1": sorted(analysis_data["themes1"].items(), key=lambda item: item[1], reverse=True),
            "themes2": sorted(analysis_data["themes2"].items(), key=lambda item: item[1], reverse=True),
            "class1": analysis_data["classification1"],
            "class2": analysis_data["classification2"],
        }

    if request.method == "POST":
        # Read uploaded files from form
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        # Validate upload presence and supported types
        if not file1 or not file2 or file1.filename == "" or file2.filename == "":
            error = "Please select two files."
        elif not allowed_file(file1.filename) or not allowed_file(file2.filename):
            error = "Unsupported file type. Use .txt, .docx, or .pdf."
        else:
            try:
                raw_text1 = process_upload(file1)
                raw_text2 = process_upload(file2)
            except Exception as exc:
                error = str(exc)
            else:
                # Run full NLP analysis
                analysis = analyze_raw_texts(raw_text1, raw_text2)

                # Cache analysis so pagination and extra pages can reuse it
                analysis_id = str(uuid.uuid4())
                _cache_set(analysis_id, analysis)
                session["analysis_id"] = analysis_id

                common_limit = int(request.form.get("common_limit", 20))
                list_limit = int(request.form.get("list_limit", 10))
                page = int(request.form.get("common_page", 1))
                list_page = int(request.form.get("list_page", 1))
                session["common_limit"] = common_limit
                session["list_limit"] = list_limit

                result = build_result(analysis, common_limit, list_limit, page, list_page)

                classification1 = analysis["classification1"]
                classification2 = analysis["classification2"]
                chart_limit = 8

                # Store lightweight chart data in session for charts page
                session["charts"] = {
                    "top1": {
                        "labels": [w for w, _ in analysis["top1"][:chart_limit]],
                        "values": [c for _, c in analysis["top1"][:chart_limit]],
                    },
                    "top2": {
                        "labels": [w for w, _ in analysis["top2"][:chart_limit]],
                        "values": [c for _, c in analysis["top2"][:chart_limit]],
                    },
                    "common": {
                        "labels": [w for w, _ in analysis["top_common"][:chart_limit]],
                        "values": [c for _, c in analysis["top_common"][:chart_limit]],
                    },
                    "themes": {
                        "labels": list(analysis["themes1"].keys()),
                        "values1": [analysis["themes1"][k] for k in analysis["themes1"].keys()],
                        "values2": [analysis["themes2"][k] for k in analysis["themes1"].keys()],
                    },
                    "similarity": analysis["similarity"],
                }

                # Store propaganda-focused results in session for dedicated page
                session["propaganda"] = {
                    "file1": {
                        "label": classification1["label"],
                        "propaganda_count": classification1["propaganda_count"],
                        "neutral_count": classification1["neutral_count"],
                        "ratio": classification1["ratio"],
                        "density": classification1["density"],
                        "top_terms": [
                            {"term": term, "count": count}
                            for term, count in classification1["top_terms"]
                        ],
                    },
                    "file2": {
                        "label": classification2["label"],
                        "propaganda_count": classification2["propaganda_count"],
                        "neutral_count": classification2["neutral_count"],
                        "ratio": classification2["ratio"],
                        "density": classification2["density"],
                        "top_terms": [
                            {"term": term, "count": count}
                            for term, count in classification2["top_terms"]
                        ],
                    },
                }

    elif session.get("analysis_id"):
        # Rebuild result from cached analysis if user changes pages/limits
        analysis_id = session.get("analysis_id")
        analysis = _cache_get(analysis_id)
        if analysis:
            common_limit = int(request.args.get("common_limit", session.get("common_limit", 20)))
            list_limit = int(request.args.get("list_limit", session.get("list_limit", 10)))
            page = int(request.args.get("common_page", 1))
            list_page = int(request.args.get("list_page", 1))
            session["common_limit"] = common_limit
            session["list_limit"] = list_limit
            result = build_result(analysis, common_limit, list_limit, page, list_page)

    return render_template("index.html", result=result, error=error, image_files=image_files)


@app.route("/propaganda/<file_id>")
def propaganda(file_id: str):
    """
    Dedicated page showing propaganda metrics for one selected file.
    """
    data = session.get("propaganda")
    if not data:
        return render_template("propaganda.html", result=None, error="No analysis data.")

    key = "file1" if file_id == "1" else "file2" if file_id == "2" else None
    if not key:
        return render_template("propaganda.html", result=None, error="Invalid file id.")

    return render_template("propaganda.html", result=data[key], error=None, file_id=file_id)


@app.route("/charts")
def charts():
    """
    Dedicated page for chart visualizations.
    """
    data = session.get("charts")
    if not data:
        return render_template("charts.html", result=None, error="No analysis data.")
    return render_template("charts.html", result=data, error=None)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)