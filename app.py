import gradio as gr
import time
import os
import sys # For error output
import numpy as np
import mlx.core as mx
import mlx_whisper
from transformers import AutoTokenizer, AutoConfig

# Attempt to import mlx-transformers components (commented out as we switch to standard transformers for NLLB)
# try:
#     from mlx_transformers.models.nllb import NLLBForConditionalGeneration as MLXNLLBModel
#     mlx_transformers_available = True
#     print("Successfully imported MLXNLLBModel from mlx_transformers.")
# except ImportError:
#     MLXNLLBModel = None
#     mlx_transformers_available = False
#     print("WARNING: Could not import NLLBForConditionalGeneration from mlx_transformers.", file=sys.stderr)
#     print("Translation functionality will be limited or unavailable if it relies on this.", file=sys.stderr)

# Import for standard Hugging Face transformers model
from transformers import AutoModelForSeq2SeqLM
import torch # PyTorch will be used by the standard transformers library

# --- Model Configuration ---\
STT_MODEL_NAME = "kaiinui/kotoba-whisper-v1.0-mlx"
TRANSLATION_MODEL_HF_ID = "Helsinki-NLP/opus-mt-ja-en"

# --- Global Variables for Loaded Models ---
stt_model = None
translator_model = None
translator_tokenizer = None

# --- Model Loading Functions ---
def load_models():
    global stt_model, translator_model, translator_tokenizer

    # Load STT Model
    print(f"Loading STT model: {STT_MODEL_NAME}...")
    try:
        # mlx_whisper loads the model on first use, but we can prime it.
        # A specific load function isn't explicitly provided by mlx_whisper's typical API surface,
        # it happens during the first transcribe call. We'll rely on this implicit loading.
        # To confirm, we could do a dummy transcribe here, but it's better to let it load on first actual use.
        stt_model = STT_MODEL_NAME # Store the name, transcribe will use it
        print(f"STT model '{STT_MODEL_NAME}' will be loaded on first use by mlx_whisper.")
    except Exception as e:
        print(f"ERROR: Could not initialize STT model ({STT_MODEL_NAME}): {e}", file=sys.stderr)
        stt_model = None

    # Load Translation Model (using standard Hugging Face transformers)
    print(f"Loading Translation tokenizer and model: {TRANSLATION_MODEL_HF_ID}...")
    try:
        translator_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_HF_ID) # Helsinki models usually don't need src_lang
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_HF_ID)
        # If using GPU and it's available:
        # if torch.cuda.is_available():
        #    translator_model.to("cuda")
        print(f"Translation model '{TRANSLATION_MODEL_HF_ID}' and tokenizer ready (using standard Transformers).")
    except Exception as e:
        print(f"ERROR: Could not load translation model or tokenizer ({TRANSLATION_MODEL_HF_ID}) using standard Transformers: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        translator_model = None
        translator_tokenizer = None

# --- Real ML Model Functions ---
def transcribe_audio(audio_filepath):
    """Transcribes audio using MLX Whisper."""
    if not stt_model:
        print("STT model not loaded. Cannot transcribe.", file=sys.stderr)
        return "Error: STT model not available."
    if not audio_filepath or not os.path.exists(audio_filepath):
        print(f"Audio filepath is None or does not exist: {audio_filepath}", file=sys.stderr)
        return "Error: Audio data not found or invalid."

    print(f"Transcribing: '{audio_filepath}' with {stt_model}")
    try:
        # Ensure audio is in a format mlx_whisper can handle.
        # mlx_whisper.transcribe can take a file path.
        result = mlx_whisper.transcribe(
            audio_filepath,
            path_or_hf_repo=stt_model, # Uses the globally set model name
            language="ja"
        )
        mx.eval() # Ensure transcription is computed
        transcribed_text = result["text"].strip()
        print(f"Transcription result: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        print(f"STT Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return "Error: Speech transcription failed."

def translate_text(text_to_translate, source_lang="ja", target_lang="en"):
    """Translates text using the loaded Hugging Face Seq2Seq model."""
    if not translator_model or not translator_tokenizer:
        print("Translation model or tokenizer not loaded. Cannot translate.", file=sys.stderr)
        return "Error: Translation model not available."
    if not text_to_translate or "Error:" in text_to_translate:
        return "Translation skipped due to transcription error or no input."

    print(f"Translating: '{text_to_translate}' from {source_lang} to {target_lang}")
    try:
        # Tokenize for standard Hugging Face model (expects PyTorch tensors by default)
        inputs = translator_tokenizer(text_to_translate, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask # Use attention mask

        # If using GPU and model is on GPU:
        # if torch.cuda.is_available() and translator_model.device.type == "cuda":
        #    input_ids = input_ids.to("cuda")
        #    if attention_mask is not None:
        #        attention_mask = attention_mask.to("cuda")

        # For Helsinki-NLP models, explicit decoder_start_token_id is usually not needed,
        # as the language pair is fixed.
        # Generate translation using the standard Hugging Face model
        output_tokens = translator_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=512 # You can adjust max_length
            # num_beams=5, # Can be added back if needed for quality
            # early_stopping=True # Can be added back if needed
        )
        # No mx.eval() needed for PyTorch tensors

        # Decode
        # output_tokens is a PyTorch tensor
        tokens_to_decode = output_tokens[0] # Assuming batch size 1 and taking the first sequence

        # Use batch_decode for robustness, even with a single sequence
        decoded_batch = translator_tokenizer.batch_decode([tokens_to_decode], skip_special_tokens=True)
        translated_text = decoded_batch[0] if decoded_batch else "" # Get the first string from the batch

        final_text = translated_text.strip() if translated_text else "(Empty translation)"
        print(f"Translation result: {final_text}")
        return final_text

    except Exception as e:
        print(f"Translation Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return "Error: Translation failed."
# --- End of Real ML Model Functions ---

# --- Gradio App Functions ---
def start_translation_process(audio_file_path, progress=gr.Progress(track_tqdm=True)):
    """
    Handles the voice translation process.
    """
    if audio_file_path is None:
        return "Please record audio first using the microphone.", "Status: Idle"

    progress(0, desc="Initializing...")
    # Status will be updated by progress.tqdm descriptions and the final return

    # 1. Transcribe Audio
    # In a real app, this involves calling a speech-to-text model.
    # The loop with progress.tqdm is for showing cancellable progress.
    japanese_text = ""
    # Wrap the potentially long-running step in a loop for progress and cancellation checks.
    # Here, we simulate it with a single iteration.
    for _ in progress.tqdm(range(1), desc="Transcribing Audio..."):
        japanese_text = transcribe_audio(audio_file_path) # Use real STT
        # In a real model call, you might not have a loop here,
        # but the 'cancels' feature will still work if the underlying Python code is interruptible.
        # If transcribe_audio was a long C extension call, it might not be interruptible by Python signals.

    if "Error:" in japanese_text:
        progress(1, desc="Transcription Error") # Update progress to final state
        return japanese_text, "Status: Transcription Error"

    progress(0.5, desc="Transcription Complete. Starting Translation...")
    # The output_area will be updated once at the end. Intermediate transcribed text won't show separately.

    # 2. Translate Text
    # In a real app, this involves calling a machine translation model.
    english_text = ""

    for _ in progress.tqdm(range(1), desc="Translating Text..."):
        english_text = translate_text(japanese_text) # Use the actual transcribed text

    final_output = f"Japanese (transcribed): {japanese_text}\n\nEnglish (translated): {english_text}"
    progress(1, desc="Process Complete")
    return final_output, "Status: Translation Complete"

def ui_stop_action():
    """
    Called when the stop_button is clicked.
    Its main purpose is to trigger the 'cancels' behavior on the start_event listener.
    It also provides immediate feedback by updating the status label.
    """
    # The actual stopping is handled by Gradio's 'cancels' mechanism.
    # This function just updates the UI to reflect that a stop was requested.
    return "Status: Translation process stop requested. Will halt if currently running."

# --- Gradio Interface Definition ---
# Using gr.Blocks for custom layout
with gr.Blocks(theme=gr.themes.Soft()) as app_interface:
    gr.Markdown(
        """
        # Japanese to English Voice Translator
        🎤 Record your voice in Japanese using the microphone widget below.
        ▶️ Click the "Start Translation" button to process the audio.
        ⏹️ Click the "Stop Translation" button to interrupt the process if needed.
        """
    )

    with gr.Column(): # Main container column
        # Audio input widget
        with gr.Row():
            mic_input = gr.Audio(
                source="microphone",
                type="filepath", # Saves audio to a temporary file path
                label="Record Japanese Speech Here",
                info="Click the microphone icon to record. Click the checkmark to finalize recording."
            )

        # Control buttons
        with gr.Row():
            start_button = gr.Button("▶️ Start Translation")
            stop_button = gr.Button("⏹️ Stop Translation")

        # Status display
        status_label = gr.Label("Status: Idle") # Use gr.Label for status messages

        # Output area
        output_area = gr.Textbox(
            label="Translation Output",
            lines=6,
            interactive=False, # Output is not meant to be edited by the user
            placeholder="Transcription and translation will appear here..."
        )

    # Event listeners for button clicks
    start_event_listener = start_button.click(
        fn=start_translation_process,
        inputs=[mic_input],
        outputs=[output_area, status_label]
        # The 'api_name' argument could be added here if you want to expose this as an API endpoint.
        # e.g., api_name="translate_japanese_voice_to_english"
    )

    stop_button.click(
        fn=ui_stop_action,
        inputs=None, # No direct inputs needed for the stop action itself
        outputs=[status_label], # Update the status label to show a stop was requested
        cancels=[start_event_listener] # This is crucial: it cancels the ongoing event triggered by start_button
    )

# To run the Gradio app:
if __name__ == "__main__":
    load_models() # Load models when the script is run
    # demo.launch(share=True) # share=True creates a public link (useful for sharing)
    # For local development:
    app_interface.queue().launch()
    # You can specify server_name="0.0.0.0" to make it accessible on your local network.
    # app_interface.queue().launch(server_name="0.0.0.0")
