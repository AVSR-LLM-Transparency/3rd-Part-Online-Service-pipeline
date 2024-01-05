import queue
import re
import sys
import time
import os
import csv
import warnings

from google.cloud import speech
import pyaudio

import openai 



# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms 
warnings.filterwarnings("ignore")

# Colors of printing 
GREEN = "\033[0;32m"
BLUE="\033[0;94m"
RESET="\033[0m"



def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(
        self: object,
        rate: int,
        chunk_size: int,
    ) -> None:
        """Creates a resumable microphone stream.

        Args:
        self: The class instance.
        rate: The audio file's sampling rate.
        chunk_size: The audio file's chunk size.

        returns: None
        """
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self: object) -> object:
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
        self: The class instance.
        in_data: The audio data as a bytes object.
        args: Additional arguments.
        kwargs: Additional arguments.

        returns: None
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Stream Audio from microphone to API and to local buffer

        Args:
            self: The class instance.

        returns:
            The data from the audio stream.
        """
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


def main():
    # Google Cloud Speech Recogntion Initialization
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=False
    )

    # OpenAI ChatGPT Initialization
    openai.api_key = "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"
    messages = [ {"role": "system", "content":  
              "You are a intelligent assistant."} ]
    
    # Preparation for experiment
    print('\n')
    print('Hello, dear my friend! Welcome to our experiment!')
    username=input('Please enter your name (could use a nickname if you want):')

    file_name=username+'.csv'
    file_path=os.path.join(os.path.abspath(".."),'data',file_name)
    with open(file_path,'w') as file:
        csv_writer=csv.writer(file)
        labels=['Round','SR_time','GPT_time','SR_result','GPT_response']
        csv_writer.writerow(labels)

    # Instructions of experiment
    print('\n')
    print(f"Hi! Dear {username}~")
    print("\r")
    print("Now we will start the test round 0, just to familiarize you with the experiment procedure")
    print("Please follow my simple instructions~  And the pattern is all the same during the experiment")
    print("\r")
    print("Please hold the microphone close to your mouth~")
    print("And every time we will ask you to press ENTER on the keyborad to start or stop talking~")

    # Testing Round 0
    print("\n")
    print("TEST Round 0 start!!!")
    print("\r")

    space=input("Please press ENTER to start talking...")
    stream = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

    stream.__enter__()
    stream.audio_input = []
    audio_generator = stream.generator() 

    print("\r")
    space=input("I am listening...(Press ENTER to stop talking)")

    print("\r")
    print("I am understanding...")
    print("\r")

    # Speech recognition
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        sys.stdout.write(BLUE)
        print('{}:'.format(username),end=' ')
        print(transcript)
        sys.stdout.write(RESET)

        # ChatGPT
        print("\r")
        print("I am thinking...")
        print("\r")

        messages.append( {"role": "user", "content": transcript}, ) 
        chat = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=messages )
        reply = chat.choices[0].message.content 
        messages.append({"role": "assistant", "content": reply}) 

        sys.stdout.write(GREEN)
        print(f"Assistant: {reply}")
        sys.stdout.write(RESET)

        break

    stream.__exit__(None,None,None)
    print("\r")
    print("Now TEST Round 0 ends, The experiment officially start!!!")
    print("\n")


    # Main Body of experiment
    rounds=5

    for Round in range(1,rounds+1):
        print(f"Round {Round} start!!!")
        print("\r")

        space=input("Please press ENTER to start talking...")
        stream = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

        stream.__enter__()
        stream.audio_input = []
        audio_generator = stream.generator() 

        print("\r")
        space=input("I am listening...(Press ENTER to stop talking)")

        SR_start=time.time()
        print("\r")
        print("I am understanding...")
        print("\r")

        # Speech recognition
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        transcript=""
        reply=""

        for response in responses:

            if not response.results:
                continue

            result = response.results[0]

            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            sys.stdout.write(BLUE)
            print('{}:'.format(username),end=' ')
            print(transcript)
            sys.stdout.write(RESET)
            SR_end=time.time()

            # ChatGPT
            print("\r")
            print("I am thinking...")
            print("\r")

            GPT_start=time.time()
            messages.append( {"role": "user", "content": transcript}, ) 
            chat = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=messages )
            reply = chat.choices[0].message.content 
            messages.append({"role": "assistant", "content": reply}) 

            sys.stdout.write(GREEN)
            print(f"Assistant: {reply}")
            sys.stdout.write(RESET)
            GPT_end=time.time()

            break
        
        stream.__exit__(None,None,None)
        print("\r")
        print("Saving data... Next round will be ready soon...")
        print("\n")

        # Saving data
        SR_time=round(SR_end-SR_start,3)
        GPT_time=round(GPT_end-GPT_start,3)
        content=[Round,SR_time,GPT_time,transcript,reply]

        with open(file_path,'a') as file:
            csv_writer=csv.writer(file)
            csv_writer.writerow(content)
    

    # End of experiment
    print('\n')
    print('Experiment is over!')
    print('Thank you so much for your cooperation and patience! Merry Christmas in advance~')



# Main Running
if __name__ == "__main__":
    main()




        




            


            





    



