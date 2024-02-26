import gradio as gr
from gradio_rich_textbox import RichTextbox
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from lang_list import TEXT_SOURCE_LANGUAGE_NAMES , LANGUAGE_NAME_TO_CODE , text_source_language_codes
from gradio_client import Client
from dotenv import load_dotenv
import requests
from io import BytesIO  
import cohere
import os
import re
import pandas as pd
import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
import hashlib


title = "# Welcome to AyaTonic"
description = "Learn a New Language With Aya"
# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv('CO_API_KEY')
SEAMLESSM4T = os.getenv('SEAMLESSM4T')
df = pd.read_csv("lang_list.csv")
choices = df["name"].to_list()
inputlanguage = ""
producetext =  "\n\nProduce a complete expositional blog post in {target_language} based on the above :"
formatinputstring = "\n\nthe above text is a learning aid. you must use rich text format to rewrite the above and add 1 . a red color tags for nouns 2. a blue color tag for verbs 3. a green color tag for adjectives and adverbs:"
translatetextinst = "\n\nthe above text is a learning aid. you must use markdown format to translate the above into {inputlanguage} :'"
# Regular expression patterns for each color
patterns = {
    "red": r'<span style="color: red;">(.*?)</span>',
    "blue": r'<span style="color: blue;">(.*?)</span>',
    "green": r'<span style="color: green;">(.*?)</span>',
}

# Dictionaries to hold the matches
matches = {
    "red": [],
    "blue": [],
    "green": [],
}

co = cohere.Client(COHERE_API_KEY)
audio_client = Client(SEAMLESSM4T)

def get_language_code(language_name):
    """
    Extracts the first two letters of the language code based on the language name.
    """
    try:
        code = df.loc[df['name'].str.lower() == language_name.lower(), 'code'].values[0]
        return code
    except IndexError:
        print(f"Language name '{language_name}' not found.")
        return None

def translate_text(text, inputlanguage, target_language):
    """
    Translates text.
    """
    # Ensure you format the instruction string within the function body
    instructions = translatetextinst.format(inputlanguage=inputlanguage)
    producetext_formatted = producetext.format(target_language=target_language)
    prompt = f"{text}{producetext_formatted}\n{instructions}"
    response = co.generate(
        model='c4ai-aya',
        prompt=prompt,
        max_tokens=2986,
        temperature=0.6,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    return response.generations[0].text

class LongAudioProcessor:
    def __init__(self, audio_client, api_key=None):
        self.client = audio_client
        self.process_audio_to_text = process_audio_to_text
        self.api_key = api_key

    def process_long_audio(self, audio_path, inputlanguage, outputlanguage, chunk_length_ms=20000):
        """
        Process audio files longer than 29 seconds by chunking them into smaller segments.
        """
        audio = AudioSegment.from_file(audio_path)
        chunks = make_chunks(audio, chunk_length_ms)
        full_text = ""
        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk{i}.wav"
            with open(chunk_name, 'wb') as file:
                chunk.export(file, format="wav")
            try:
                result = self.process_audio_to_text(chunk_name, inputlanguage=inputlanguage, outputlanguage=outputlanguage)
                full_text += " " + result.strip()
            except Exception as e:
                print(f"Error processing {chunk_name}: {e}")
            finally:
                if os.path.exists(chunk_name):
                    os.remove(chunk_name)
        return full_text.strip()
class TaggedPhraseExtractor:
    def __init__(self, text=''):
        self.text = text
        self.patterns = patterns 

    def set_text(self, text):
        """Set the text to search within."""
        self.text = text

    def add_pattern(self, color, pattern):
        """Add a new color and its associated pattern."""
        self.patterns[color] = pattern

    def extract_phrases(self):
        """Extract phrases for all colors and patterns added, including the three longest phrases."""
        matches = {}
        for color, pattern in self.patterns.items():
            found_phrases = re.findall(pattern, self.text)
            sorted_phrases = sorted(found_phrases, key=len, reverse=True)
            matches[color] = sorted_phrases[:3]
        return matches

    def print_phrases(self):
        """Extract phrases and print them, including the three longest phrases."""
        matches = self.extract_phrases()
        for color, data in matches.items():
            print(f"Phrases with color {color}:")
            for phrase in data['all_phrases']:
                print(f"- {phrase}")
            print(f"\nThree longest phrases for color {color}:")
            for phrase in data['top_three_longest']:
                print(f"- {phrase}")
            print()
            
def process_audio_to_text(audio_path, inputlanguage="English", outputlanguage="English"):
    """
    Convert audio input to text using the Gradio client.
    """
    audio_client = Client(SEAMLESSM4T)
    result = audio_client.predict(
        audio_path,
        inputlanguage,  
        outputlanguage,  
        api_name="/s2tt"
    )
    print("Audio Result: ", result)
    return result[0]


def process_text_to_audio(text, translatefrom="English", translateto="English", filename_prefix="audio", base_url="https://huggingface.co/spaces/MultiTransformer/AyaTonic"):
    """
    Convert text input to audio, ensuring the audio file is correctly saved and returned as a file path or URL.
    """
    try:
        # Generate audio from text
        audio_response = audio_client.predict(
            text,
            translatefrom,  
            translateto, 
            api_name="/t2st"
        )
        if "error" in audio_response:
            raise ValueError(f"API Error: {audio_response['error']}")
        audio_url = audio_response[0]
        if not audio_url.startswith('http'):
            raise ValueError("Invalid URL returned from audio generation API")

        response = requests.get(audio_url)
        if response.status_code != 200:
            raise ValueError("Failed to download audio from URL")

        audio_data = response.content  
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        filename = f"{filename_prefix}_{text_hash}.wav"
        
        directory = "audio_files"
        os.makedirs(directory, exist_ok=True)
        
        file_path = os.path.join(directory, filename)
        with open(file_path, 'wb') as file:
            file.write(audio_data)
        
        full_url = f"{base_url}/{directory}/{filename}"

        return full_url
    except Exception as e:
        print(f"Error processing text to audio: {e}")
        return None
    
def save_audio_data_to_file(audio_data, directory="audio_files", filename="output_audio.wav"):
    """
    Save audio data to a file and return the file path.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as file:
        file.write(audio_data)
    return file_path

# Ensure the function that reads the audio file checks if the path is a file
def read_audio_file(file_path):
    """
    Read and return the audio file content if the path is a file.
    """
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            return file.read()
    else:
        raise ValueError(f"Expected a file path, got a directory: {file_path}")


def initialize_ocr_models():
    """
    Load the detection and recognition models along with their processors.
    """
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    return det_processor, det_model, rec_model, rec_processor

class OCRProcessor:
    def __init__(self, lang_code=["en"]): 
        self.lang_code = lang_code
        self.det_processor, self.det_model, self.rec_model, self.rec_processor = initialize_ocr_models()

    def process_image(self, image):
        """
        Process a PIL image and return the OCR text.
        """
        predictions = run_ocr([image], [self.lang_code], self.det_model, self.det_processor, self.rec_model, self.rec_processor)
        return predictions[0] 

    def process_pdf(self, pdf_path):
        """
        Process a PDF file and return the OCR text.
        """
        predictions = run_ocr([pdf_path], [self.lang_code], self.det_model, self.det_processor, self.rec_model, self.rec_processor)
        return predictions[0]
    
def process_input(image=None, file=None, audio=None, text="", translateto = "English", translatefrom = "English" ):
    lang_code = get_language_code(translatefrom)
    ocr_processor = OCRProcessor(lang_code)
    final_text = text
    print("Image :", image)
    if image is not None:
        ocr_prediction = ocr_processor.process_image(image)
        for idx in range(len((list(ocr_prediction)[0][1]))):
            final_text += " "
            final_text += list((list(ocr_prediction)[0][1])[idx])[1][1]
    if file is not None:
        if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            pil_image = Image.open(file)
            ocr_prediction = ocr_processor.process_image(pil_image)
            for idx in range(len((list(ocr_prediction)[0][1]))):
                final_text += " "
                final_text += list((list(ocr_prediction)[0][1])[idx])[1][1]
        elif file.name.lower().endswith('.pdf'):
            ocr_prediction = ocr_processor.process_pdf(file.name)
            for idx in range(len((list(ocr_prediction)[0][1]))):
                final_text += " "
                final_text += list((list(ocr_prediction)[0][1])[idx])[1][1]
        else:
            final_text += "\nUnsupported file type."
    print("OCR Text: ", final_text)
    if audio is not None:
        long_audio_processor = LongAudioProcessor(audio_client)
        audio_text = long_audio_processor.process_long_audio(audio, inputlanguage=translatefrom, outputlanguage=translateto)
        final_text += "\n" + audio_text

    final_text_with_producetext = final_text + producetext.format(target_language=translateto)

    response = co.generate(
        model='c4ai-aya',
        prompt=final_text_with_producetext,
        max_tokens=1024,
        temperature=0.5
    )
    # add graceful handling for errors (overflow)
    generated_text = response.generations[0].text
    print("Generated Text: ", generated_text)
    generated_text_with_format = generated_text + "\n" + formatinputstring
    response = co.generate(
        model='command-nightly',
        prompt=generated_text_with_format,
        max_tokens=4000,
        temperature=0.5
    )
    processed_text = response.generations[0].text

    audio_output = process_text_to_audio(processed_text, translateto, translateto)
    extractor = TaggedPhraseExtractor(final_text)
    matches = extractor.extract_phrases()

    top_phrases = []
    for color, phrases in matches.items():
        top_phrases.extend(phrases)

    while len(top_phrases) < 3:
        top_phrases.append("")

    audio_outputs = []
    translations = []
    for phrase in top_phrases:
        if phrase:
            translated_phrase = translate_text(phrase, translatefrom=translatefrom, translateto=translateto)
            translations.append(translated_phrase)
            target_audio = process_text_to_audio(phrase, translatefrom=translateto, translateto=translateto)
            native_audio = process_text_to_audio(translated_phrase, translatefrom=translatefrom, translateto=translatefrom)
            audio_outputs.append((target_audio, native_audio))
        else:
            translations.append("")
            audio_outputs.append(("", ""))

    return final_text, audio_output, top_phrases, translations, audio_outputs



inputs = [
    
    gr.Dropdown(choices=choices, label="Your Native Language"),
    gr.Dropdown(choices=choices, label="Language To Learn"),
    gr.Audio(sources="microphone", type="filepath", label="Mic Input"),
    gr.Image(type="pil", label="Camera Input"),
    gr.Textbox(lines=2, label="Text Input"),
    gr.File(label="File Upload")
]

outputs = [
    RichTextbox(label="Processed Text"),
    gr.Audio(label="Audio"),
    gr.Textbox(label="Focus 1"),
    gr.Textbox(label="Translated Phrases 1"),
    gr.Audio(label="Audio Output (Native Language) 1"),
    gr.Audio(label="Audio Output (Target Language) 1"),
    gr.Textbox(label="Focus 2"),
    gr.Textbox(label="Translated Phrases 2"),
    gr.Audio(label="Audio Output (Native Language) 2"),
    gr.Audio(label="Audio Output (Target Language) 2"),
    gr.Textbox(label="Focus 3"),
    gr.Textbox(label="Translated Phrases 3"),
    gr.Audio(label="Audio Output (Native Language) 3"),
    gr.Audio(label="Audio Output (Target Language) 3")
]


def update_outputs(inputlanguage, target_language, audio, image, text, file):
    processed_text, audio_output_path, top_phrases, translations, audio_outputs = process_input(
        image=image, file=file, audio=audio, text=text, 
        translateto=target_language, translatefrom=inputlanguage 
    )

    output_tuple = (
        processed_text,  # RichTextbox content
        audio_output_path,  # Main audio output
        top_phrases[0] if len(top_phrases) > 0 else "",  # Focus 1
        translations[0] if len(translations) > 0 else "",  # Translated Phrases 1
        audio_outputs[0][0] if len(audio_outputs) > 0 else "",  # Audio Output (Native Language) 1
        audio_outputs[0][1] if len(audio_outputs) > 0 else "",  # Audio Output (Target Language) 1
        top_phrases[1] if len(top_phrases) > 1 else "",  # Focus 2
        translations[1] if len(translations) > 1 else "",  # Translated Phrases 2
        audio_outputs[1][0] if len(audio_outputs) > 1 else "",  # Audio Output (Native Language) 2
        audio_outputs[1][1] if len(audio_outputs) > 1 else "",  # Audio Output (Target Language) 2
        top_phrases[2] if len(top_phrases) > 2 else "",  # Focus 3
        translations[2] if len(translations) > 2 else "",  # Translated Phrases 3
        audio_outputs[2][0] if len(audio_outputs) > 2 else "",  # Audio Output (Native Language) 3
        audio_outputs[2][1] if len(audio_outputs) > 2 else ""   # Audio Output (Target Language) 3
    )

    return output_tuple

def interface_func(inputlanguage, target_language, audio, image, text, file):
    return update_outputs(inputlanguage, target_language, audio, image, text, file)

iface = gr.Interface(fn=interface_func, inputs=inputs, outputs=outputs, title=title, description=description)

if __name__ == "__main__":
    iface.launch()