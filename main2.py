# Import the necessary libraries
import os
import whisper  # Whisper is OpenAI's automatic speech recognition (ASR) system.
import streamlit as st  # Streamlit is a framework for building machine learning and data science web apps.
from pydub import AudioSegment  # PyDub is a simple and easy-to-use Python library for audio manipulation.
import openai  # OpenAI's API for GPT-3, a large-scale language model.
from dotenv import load_dotenv  # Python-dotenv reads key-value pairs from a .env file and can set them as environment variables.

# Load .env file where we store our OpenAI key
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Whisper & GPT-3",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

# Define the directories for uploading, downloading, and storing transcripts
upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"

# Create the directories if they do not exist
if not os.path.exists(upload_path):
    os.makedirs(upload_path)

if not os.path.exists(download_path):
    os.makedirs(download_path)

if not os.path.exists(transcript_path):
    os.makedirs(transcript_path)

# Decorate function with st.cache to ensure the function is only rerun when the inputs change
@st.cache(persist=True, allow_output_mutation=False, show_spinner=True, suppress_st_warning=True)
def to_mp3(audio_file, output_audio_file, upload_path, download_path):
    """
    Convert audio files to MP3 format. This function supports several audio file formats including WAV, MP3, OGG, WMA, AAC, FLAC, FLV, and MP4.
    """
    # Get the file extension and convert to lowercase
    file_extension = audio_file.name.split('.')[-1].lower()

    # Check the file extension and convert the audio to MP3 accordingly
    if file_extension == "wav":
        audio_data = AudioSegment.from_wav(os.path.join(upload_path, audio_file.name))
    elif file_extension == "mp3":
        audio_data = AudioSegment.from_mp3(os.path.join(upload_path, audio_file.name))
    elif file_extension == "ogg":
        audio_data = AudioSegment.from_ogg(os.path.join(upload_path, audio_file.name))
    elif file_extension == "wma":
        audio_data = AudioSegment.from_file(os.path.join(upload_path, audio_file.name), "wma")
    elif file_extension == "aac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path, audio_file.name), "aac")
    elif file_extension == "flac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path, audio_file.name), "flac")
    elif file_extension == "flv":
        audio_data = AudioSegment.from_flv(os.path.join(upload_path, audio_file.name))
    elif file_extension == "mp4":
        audio_data = AudioSegment.from_file(os.path.join(upload_path, audio_file.name), "mp4")

    # Export the audio data as MP3
    audio_data.export(os.path.join(download_path, output_audio_file), format="mp3")

    return output_audio_file

@st.cache(persist=True, allow_output_mutation=False, show_spinner=True, suppress_st_warning=True)
def process_audio(filename, model_type):
    """
    Transcribe the given audio file using Whisper ASR system
    """
    # Load the model
    model = whisper.load_model(model_type)

    # Transcribe the audio file
    result = model.transcribe(filename)

    return result["text"]

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(transcript_data, txt_file):
    """
    Save the transcribed data to a text file
    """
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

# Set the title for the Streamlit app
st.title("Whisper Transcription")

# Information about the supported file formats
st.info('✨ Supports all popular audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV 😉')

# Instructions for the user
st.text("First upload your audio file and then select the model type. \nThen click on the button to transcribe and classify the sentiment of the text in the audio.")

# Provide a file uploader for the user to upload an audio file
uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","ogg","wma","aac","flac","mp4","flv"])

audio_file = None

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file
    audio_bytes = uploaded_file.read()

    # Save the uploaded file to the upload directory
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())

    # Convert the uploaded file to MP3
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)

        # Open the converted audio file
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()

    print("Opening ",audio_file)

    st.markdown("---")

    # Create two columns
    col1, col2 = st.columns(2)

    # Play the audio file in the first column
    with col1:
        st.markdown("Feel free to play your uploaded audio file 🎼")
        st.audio(audio_bytes)

    # Select the model type in the second column
    with col2:
        whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    # When the user clicks the "Generate Transcript and Classfification" button
    if st.button("Generate Transcript and Classfification"):
        # Generate the transcript for the audio file
        with st.spinner(f"Generating Transcript... 💫"):
            transcript = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())
            print(transcript)

            # If a transcript is generated, display it and perform sentiment analysis
            if transcript is not None:
                st.header("Transcript:")
                st.text(transcript)

                # Get the OpenAI key from the environment variables
                openai.api_key = os.getenv("OPENAI_API_KEY")

                # Create a completion with GPT-3 to perform sentiment analysis
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=f"Where are you now? I'm sitting in my office. I doubt that. And why would you doubt that? If you were in your office right now, we'd be having this conversation face-to-face.\n\nsentiment analysis (very negativ, negativ, neutral, positive, very positive): negative\n\n\ {transcript} \n\nsentiment analysis (very negativ, negativ, neutral, positive, very positive): ",
                    temperature=0.7,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                 # Display the result of sentiment analysis
                st.header("Sentiment analysis:")
                st.caption("very negativ, negativ, neutral, positive, very positive")
                st.text("Classified as:" + response.choices[0].text)

            
            # Celebratory balloons if successful
            st.balloons()
            st.success('✅ Successful !!')

# If no file is uploaded, display a warning
else:
    st.warning('⚠ Please upload your audio file 😯')
