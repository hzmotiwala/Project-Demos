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
                # Load the GPT model for summarization
                summarization_model = pipeline("summarization")

                # Summarize the transcript
                summary = summarization_model(transcript, max_length=1000, min_length=100, do_sample=False)
                return summary[0]['summary_text']
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
