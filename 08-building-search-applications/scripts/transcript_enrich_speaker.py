""" This script will get the speaker name from the YouTube video metadata and the first minute of the transcript using the OpenAI Functions entity extraction."""

import json
import os
import glob
import threading
import logging
import queue
import time
import argparse
import openai
from openai.embeddings_utils import get_embedding
from rich.progress import Progress
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type,
)
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

API_KEY = os.environ["LMSTUDIO_API_KEY"]
RESOURCE_ENDPOINT = os.environ["LMSTUDIO_ENDPOINT"]
TRANSCRIPT_FOLDER = "transcripts"
SEGMENT_MIN_LENGTH_MINUTES = 3
PROCESSING_THREADS = 10

MODEL_NAME = "QuantFactory/Meta-Llama-3-8B-GGUF"

openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
if args.verbose:
    logger.setLevel(logging.DEBUG)

TRANSCRIPT_FOLDER = args.folder if args.folder else '/home/pquerido/ai/course/generative-ai-for-beginners/08-building-search-applications/scripts/transcripts_the_ai_show'
if not TRANSCRIPT_FOLDER:
    logger.error("Transcript folder not provided")
    exit(1)

get_speaker_name = {
    "name": "get_speaker_name",
    "description": "Get the speaker's name for the session.",
    "parameters": {
        "type": "object",
        "properties": {
            "speakers": {
                "type": "string",
                "description": "The speaker's name.",
            }
        },
        "required": ["speaker_name"],
    },
}


openai_functions = [get_speaker_name]


# these maps are used to make the function name string to the function call
definition_map = {"get_speaker_name": get_speaker_name}

q = queue.Queue()

errors = 0


class Counter:
    """thread safe counter"""

    def __init__(self):
        """initialize the counter"""
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        """increment the counter"""
        with self.lock:
            self.value += 1
            return self.value


counter = Counter()


@retry(
    wait=wait_random_exponential(min=6, max=10),
    stop=stop_after_attempt(4),
    retry=retry_if_not_exception_type(openai.InvalidRequestError),
)
def get_speaker_info(text):
    """Gets the OpenAI functions from the text."""

    function_name = None
    arguments = None

    response_1 = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that talks European Portuguese, and are capable of making API calls.",
            },
            {"role": "user", "content": "What's the speaker's name given the following text? " + text},
        ],
        functions=openai_functions,
        function_call={"name": "get_speaker_name"},
        temperature=0.0,
    )

    # The assistant's response includes a function call. We extract the arguments from this function call

    result = response_1.get("choices")[0].get("message")

    if result.get("function_call"):
        function_name = result.get("function_call").get("name")
        arguments = json.loads(result.get("function_call").get("arguments"))

    return function_name, arguments


def clean_text(text):
    """clean the text"""
    text = text.replace("\n", " ")  # remove new lines
    text = text.replace("&#39;", "'")
    text = text.replace(">>", "")  # remove '>>'
    text = text.replace("  ", " ")  # remove double spaces
    text = text.replace("[inaudible]", "")  # [inaudible]

    return text


def get_first_segment(file_name):
    """Gets the first segment from the filename"""

    text = ""
    current_seconds = None
    segment_begin_seconds = None
    segment_finish_seconds = None

    vtt = file_name.replace(".json", ".json.vtt")

    with open(vtt, "r", encoding="utf-8") as json_file:
        json_vtt = json.load(json_file)

        for segment in json_vtt:
            current_seconds = segment.get("start")

            if segment_begin_seconds is None:
                segment_begin_seconds = current_seconds
                # calculate the finish time from the segment_begin_time
                segment_finish_seconds = (
                    segment_begin_seconds + SEGMENT_MIN_LENGTH_MINUTES * 60
                )

            if current_seconds < segment_finish_seconds:
                # add the text to the transcript
                text += clean_text(segment.get("text")) + " "

    return text


def process_queue(progress, task):
    """process the queue"""
    while not q.empty():
        filename = q.get()
        progress.update(task, advance=1)
        if errors > 100:
            logger.error("Too many errors. Exiting...")
            exit(1)

        with open(filename, "r", encoding="utf-8") as json_file:
            metadata = json.load(json_file)

            base_text = 'The title is: ' +  metadata['title'] + " " + metadata["description"] + " " + get_first_segment(filename)
            # replace new line with empty string
            base_text = base_text.replace("\n", " ")

            function_name, arguments = get_speaker_info(base_text)
            speakers = arguments.get("speakers", "")
            if speakers == "":
                print(f"From function call: {filename}\t---MISSING SPEAKER---")
                continue
            else:
                print(f"From function call: {filename}\t{speakers}")

            metadata["speaker"] = speakers
            json.dump(metadata, open(filename, "w", encoding="utf-8"))

        q.task_done()
        time.sleep(0.2)


logger.debug("Transcription folder %s", TRANSCRIPT_FOLDER)
logger.debug("Starting Speaker Update")

# load all the transcript json files into the queue
folder = os.path.join(TRANSCRIPT_FOLDER, "*.json")

for filename in glob.glob(folder):
    # load the json file
    q.put(filename)


logger.debug("Starting speaker name update. Files to be processed: %s", q.qsize())
start_time = time.time()
with Progress() as progress:
    task1 = progress.add_task("[blue]Enriching Speaker Data...", total=q.qsize())
    # create multiple threads to process the queue
    threads = []
    for i in range(PROCESSING_THREADS):
        t = threading.Thread(target=process_queue, args=(progress, task1))
        t.start()
        threads.append(t)

    # wait for all threads to finish
    for t in threads:
        t.join()

finish_time = time.time()
logger.debug(
    "Finished speaker name update. Total time taken: %s", finish_time - start_time
)
