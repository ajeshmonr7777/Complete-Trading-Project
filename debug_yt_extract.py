from youtube_transcript_api import YouTubeTranscriptApi
import sys

# Test with a known accessible video (e.g., Python tutorial)
test_video_id = "rfscVS0vtbw"  # specific video ID
if len(sys.argv) > 1:
    test_video_id = sys.argv[1]

print(f"Testing extraction for Video ID: {test_video_id}")

try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(test_video_id)
    print("Success! Transcript methods found:")
    for transcript in transcript_list:
        print(f" - {transcript.language} ({transcript.language_code})")
        
    # Try fetching one
    t = transcript_list.find_transcript(['en'])
    print(f"Fetched {len(t.fetch())} lines.")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
