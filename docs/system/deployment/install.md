
## DEPLOYMENT SETUP:

> Requirements.txt:

```
streamlit>=1.28.0
openai>=1.3.0
langchain>=0.0.340
sentence-transformers>=2.2.2
networkx>=3.1
plotly>=5.17.0
spacy>=3.7.0
transformers>=4.35.0
torch>=2.1.0
pypdf2>=3.0.0
speechrecognition>=3.10.0
pyttsx3>=2.90
whisper>=1.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
faiss-cpu>=1.7.0
```

Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

##  Install dependencies (Linux)

For **Ubuntu/Debian**:

```bash
sudo apt update
sudo apt install --no-install-recommends \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxcb-icccm4 \
    libxcb-icccm4-dev \
    libxcb-image0 \
    libxcb-image0-dev \
    libxcb-keysyms1 \
    libxcb-keysyms1-dev \
    libxcb-render-util0 \
    libxcb-render-util0-dev \
    libxcb-shape0 \
    libxcb-shape0-dev \
    libxcb-randr0 \
    libxcb-randr0-dev \
    libxcb-cursor-dev \
    libxkbcommon-x11-0


sudo apt-get install portaudio19-dev python3-pyaudio
```
