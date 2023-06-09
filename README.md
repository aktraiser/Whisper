# Whisper

[![My Project](http://img.youtube.com/vi/Hk4_Hks4xmI/0.jpg)](http://www.youtube.com/watch?v=Hk4_Hks4xmI "My Project")


This project uses OpenAI's automatic speech recognition system, Whisper, to transcribe audio files, and then uses the language model GPT-3 to perform sentiment analysis on the transcriptions.

The interactive web app was built with Streamlit and supports various audio formats.

Installation
Clone the repository:

bash
Copy code
```
git clone https://github.com/aktraiser/whisper.git
cd whisper
```
Install the dependencies with pip:

Copy code
```
pip install -r requirements.txt
```
Set your OpenAI API keys as environment variables. You can set them in a .env file:

makefile
Copy code
```
OPENAI_API_KEY=your_api_key
```
Usage
To start the app, run the following command in your terminal:

go
Copy code
```
streamlit run main2.py
```
You can then open your web browser and navigate to localhost:8501 to interact with the app.

Contributing
Contributions to this project are welcomed. Feel free to open an issue or submit a pull request.
