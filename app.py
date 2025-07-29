import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import time # Import time for sleep

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- 1. API and Model Configuration ---
try:
    # Get the API key from the environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found. Please set it in your .env file.")
    genai.configure(api_key=api_key)
    
    # Initialize the Generative Model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
except Exception as e:
    print(f"Failed to configure Gemini API: {e}")
    exit(1) # Exit if API configuration fails

@app.route('/')
def index():
    """Renders the main HTML page for the voice assistant."""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Receives audio data from the frontend, processes it with Gemini,
    and returns the transcribed text and answer.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file_blob = request.files['audio']
    if audio_file_blob.filename == '':
        return jsonify({"error": "No selected audio file"}), 400

    if audio_file_blob:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_file_blob.save(temp_audio.name)
            audio_filename = temp_audio.name

        uploaded_audio_file = None # Initialize to None
        try:
            # --- 3. Process with Gemini ---
            # Upload the audio file to the Gemini API File Service
            uploaded_audio_file = genai.upload_file(path=audio_filename)

            # --- Wait for the file to become ACTIVE ---
            # This is the crucial part to prevent the "not in ACTIVE state" error
            while uploaded_audio_file.state.name == 'PROCESSING':
                print(f"File {uploaded_audio_file.name} is still processing. Waiting...")
                time.sleep(1) # Wait for 1 second before checking again
                uploaded_audio_file = genai.get_file(uploaded_audio_file.name) # Get updated status

            if uploaded_audio_file.state.name != 'ACTIVE':
                raise Exception(f"File upload failed or is not active: {uploaded_audio_file.state.name}")

            # A structured prompt for a better-formatted output
            prompt = """
            Please perform two tasks:
            1.  Transcribe the audio accurately.
            2.  Provide a helpful answer to the transcribed query.
            3.  Provide with some easy example 
            
            Format your response exactly as follows:

            **Transcription:**
            [Your transcription here]

            **Answer:**
            [Your answer here]
            """

            # Generate content using both the prompt and the audio file
            response = model.generate_content([prompt, uploaded_audio_file])

            # --- 4. Return the Output ---
            return jsonify({"response": response.text}), 200

        except Exception as e:
            print(f"An error occurred during Gemini processing: {e}")
            return jsonify({"error": f"An error occurred: {e}"}), 500
        finally:
            # Clean up the temporary audio file
            if os.path.exists(audio_filename):
                os.remove(audio_filename)
            # Also delete the uploaded file from Gemini's file service if it was uploaded
            if uploaded_audio_file and uploaded_audio_file.state.name == 'ACTIVE':
                try:
                    genai.delete_file(uploaded_audio_file.name)
                    print(f"Successfully deleted file {uploaded_audio_file.name} from Gemini File Service.")
                except Exception as e:
                    print(f"Error deleting file {uploaded_audio_file.name} from Gemini File Service: {e}")
    else:
        return jsonify({"error": "No audio data received"}), 400

if __name__ == '__main__':
    app.run(debug=True)
