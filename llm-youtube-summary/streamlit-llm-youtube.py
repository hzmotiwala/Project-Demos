import streamlit as st
import requests
import toml
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# Read the OpenAI API key from secrets.toml

def read_openai_api_key():
    try:
        # secrets = toml.load("secrets.toml")
        #return secrets["openai"]["api_key"]
        api_key=st.secrets["OPENAI_API_KEY"]
        return api_key
    except Exception as e:
        st.error(f"Error reading OpenAI API key: {e}")
        return None


def get_transcript_from_youtube(youtube_url):
    try:
        video_id = youtube_url.split("v=")[1]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in range(len(transcript_list)):
            transcript += transcript_list[i]['text'] + " "
        st.write(transcript)
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None



def summarize_video_content(youtube_url):
    openai_api_key = read_openai_api_key()
    if openai_api_key:
        try:
            # Fetch the transcript of the YouTube video
            transcript = get_transcript_from_youtube(youtube_url)
            if transcript:

                 # Check if the total number of tokens in the transcript exceeds the threshold
                max_tokens_threshold = 5000
                if len(transcript.split()) > max_tokens_threshold:
                    st.error("Transcript is too long. Please pick a shorter video.")
                    return None

                # Load the summarization model
                summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
                
                # Split the transcript into smaller chunks
                chunk_size = 1000  # Adjust the chunk size as needed
                transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
                
                # Summarize each chunk and concatenate the summaries
                summaries = []
                for chunk in transcript_chunks:
                    summary = summarization_model(chunk, max_length=500, min_length=100, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                
                # Concatenate the summaries of all chunks
                final_summary = " ".join(summaries)
                
                return final_summary
            else:
                st.warning("Failed to fetch transcript. Please check the YouTube URL.")
                return None
        except Exception as e:
            st.error(f"Error summarizing the video content: {e}")
            return None
    else:
        st.warning("OpenAI API key not found.")
        return None


# Streamlit UI
def main():
    st.title("YouTube Video Summarizer")
    youtube_url = st.text_input("Enter the YouTube URL:")
    if st.button("Summarize"):
        if youtube_url:
            st.write("Full Transcript:")
            st.write()
            summary = summarize_video_content(youtube_url)
            if summary:
                st.write("Summarized Content:")
                st.write(summary)
        else:
            st.warning("Please enter a YouTube URL.")

if __name__ == "__main__":
    main()
