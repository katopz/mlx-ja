# Japanese to English Voice Translator (Gradio Interface)

This application provides a web interface using Gradio to (eventually) transcribe Japanese audio input and translate it to English.

Currently, it uses **mock functions** for transcription and translation. You will need to replace these with actual machine learning models for real functionality.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Running the Application

1.  **Navigate to the directory:**
    Make sure you are in the `mlx-ja` directory in your terminal.

2.  **Run the Python script:**
    ```bash
    python3 app.py
    ```

3.  **Access the interface:**
    The terminal will output a local URL (usually `http://127.0.0.1:7860` or similar). Open this URL in your web browser.

## How to Use

1.  Allow microphone access if prompted by your browser.
2.  Click the microphone icon under "Record Japanese Speech Here" to start recording.
3.  Speak clearly in Japanese.
4.  Click the checkmark icon to stop recording.
5.  Click the "▶️ Start Translation" button.
6.  The mock process will simulate transcription and translation, updating the status and showing the final (mock) output in the text box.
7.  You can click "⏹️ Stop Translation" to interrupt the process (useful for long-running actual models).

## Next Steps (Development)

1.  **Integrate Speech-to-Text:** Replace `transcribe_audio_mock` in `app.py` with a function that uses a real Japanese speech recognition model (e.g., using libraries like `Whisper`, `SpeechRecognition`, or cloud APIs).
2.  **Integrate Machine Translation:** Replace `translate_text_mock` in `app.py` with a function that uses a real Japanese-to-English translation model (e.g., using libraries like `transformers`, `googletrans`, or cloud APIs).
3.  **Error Handling:** Enhance error handling for model failures, invalid inputs, etc.
4.  **Dependencies:** Add the necessary libraries for your chosen models to `requirements.txt` and reinstall.
