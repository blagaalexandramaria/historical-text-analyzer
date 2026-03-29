# Historical Text Analyzer (NLP + GUI + Web App)

A Python application for analyzing historical texts using Natural Language Processing techniques, featuring both a desktop GUI and a web interface.
This project focuses on analyzing historical narratives and detecting propaganda patterns using interpretable NLP techniques.

The project includes:
- a desktop GUI (Tkinter)
- a web application (Flask)
- multiple NLP analysis modules

It can compare texts, extract themes, detect propaganda patterns, and visualize results.

---

## Features

### Text Processing
- Supports `.txt`, `.docx`, `.pdf`
- Text cleaning and normalization
- Tokenization 
- Stop-word removal (custom + fallback)

### NLP Analysis
- Jaccard similarity 
- TF-IDF cosine similarity
- Top words / common words / unique words
- Year extraction (historical timeline detection)
- Theme analysis (war, politics, revolution, etc.)

### Propaganda Detection
- Keyword-based classification
- Propaganda vs neutral text scoring
- Density and ratio analysis

### Interfaces
- Desktop GUI (Tkinter)
- Web app (Flask)
- File upload + folder-based processing

### Visualization
- Charts using Chart.js
- Highlighted terms
- Thematic distribution

---

## Screenshots

### Home Page
![Home](screenshots/home.png)

### Results Page
![Results](screenshots/results.png)

### Charts
![Charts](screenshots/charts.png)

### Propaganda Breakdown
![Propaganda](screenshots/propaganda.png)

### Highlighted Differences
![Highlight](screenshots/highlight.png)

---

## Project Structure
```bash
history-text/
│
├── main.py                  # Tkinter GUI application
├── analysis_service.py      # Core orchestration + caching
├── text_processing.py       # Text preprocessing
├── similarity.py            # Similarity + statistics
├── classification.py        # Propaganda detection
├── theme_analysis.py        # Theme scoring
├── themes.json              # Theme definitions
├── themes.py                # Fallback themes
├── stop_words.json          # Stop words
├── stop_words.py            # Fallback stop words
├── data/                    # Example texts
│
├── screenshots/             # Project images (README)
│   ├── home.png
│   ├── results.png
│   ├── charts.png
│   ├── propaganda.png
│   └── highlight.png
│
├── web_app/
│   ├── app.py              # Flask app
│   ├── templates/          # HTML pages
│   └── static/
│       ├── images/         # UI images
│       └── styles.css
│
├── requirements.txt
└── README.md
```
---

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/history-text.git
cd history-text
```
Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
```
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

### Desktop App (GUI)

```bash
python main.py
```
- Select folder or files
- Run analysis
- View results in interface

### Web Application

```bash
cd web_app
python app.py
```
Then open in browser:
`http://127.0.0.1:5000`

---

## Example Capabilities
- Compare two historical texts (e.g., WW1 Romania vs Russia)
- Identify key terms and differences
- Extract important years (timeline)
- Detect dominant themes
- Highlight propaganda-related language
  
---

## What I Learned
- Natural Language Processing pipelines
- Text preprocessing techniques
- Similarity metrics (Jaccard, TF-IDF)
- GUI development with Tkinter
- Web development with Flask
- Data visualization (Chart.js)
- Structuring a multi-module Python project

---

## Text Sources / Credits

The historical texts used for analysis:
- **Text 1 – Romania (World War I)
https://encyclopedia.1914-1918-online.net/article/romania-1-1/

- **Text 2 – Russia (War and Revolution, 1914–1921)
https://www.theworldwar.org/learn/educator-resource/war-and-revolution-russia-1914-1921

Used strictly for educational purposes.

---

## Image Credits

The images used in the interface were sourced from Pinterest:
- Clock.jpg
  https://es.pinterest.com/pin/595460382027290860/
- Train.jpg
  https://es.pinterest.com/pin/595460382027290855/
- Manuscripts.jpg
  https://es.pinterest.com/pin/595460382027290855/
- Giovanni-laterano-text.jpg
  https://es.pinterest.com/pin/595460382027290888/
- Statue.jpg
  https://es.pinterest.com/pin/595460382027290885/
- Filipino-art.jpg
  https://es.pinterest.com/pin/595460382027290875/
- Old-vintage-map.jpg
  https://es.pinterest.com/pin/595460382027290834/
- History-collage.jpg
  https://es.pinterest.com/pin/595460382027290830/
- Vintage-pictures.jpg
  https://es.pinterest.com/pin/595460382027290830/

---

## Possible Improvements

- Machine learning classification (scikit-learn)
- Named Entity Recognition (NER)
- More advanced propaganda detection
- Multi-text comparison
- REST API version

---

## Author

Alexandra Blaga
Computer Science Student

---

## License

This project is for educational purposes.
