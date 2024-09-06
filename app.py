import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import io
from pydub import AudioSegment
import tempfile
import os
import wave
import collections
import contextlib
import struct
import pandas as pd

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield (voiced_frames[0].timestamp, 
                       voiced_frames[-1].timestamp + voiced_frames[-1].duration,
                       b''.join([f.bytes for f in voiced_frames]))
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        yield (voiced_frames[0].timestamp, 
               voiced_frames[-1].timestamp + voiced_frames[-1].duration,
               b''.join([f.bytes for f in voiced_frames]))
    if voiced_frames:
        yield (voiced_frames[0].timestamp, 
               voiced_frames[-1].timestamp + voiced_frames[-1].duration,
               b''.join([f.bytes for f in voiced_frames]))

def convert_to_wav_16khz(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_filename = temp_file.name
    
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(temp_filename, format="wav")
    
    return temp_filename

def analyze_audio(audio_file):
    # Convert to 16kHz WAV
    wav_file = convert_to_wav_16khz(audio_file)
    
    # Read the WAV file
    with contextlib.closing(wave.open(wav_file, 'rb')) as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
    
    # Initialize WebRTC VAD
    vad = webrtcvad.Vad(3)  # Aggressiveness mode 3
    
    # Generate frames
    frames = frame_generator(30, pcm_data, sample_rate)
    frames = list(frames)  # Convert generator to list
    
    # Collect voice segments
    segments = list(vad_collector(sample_rate, 30, 300, vad, frames))
    
    # Process segments
    processed_segments = []
    for i, segment in enumerate(segments):
        start_time, end_time, audio_bytes = segment
        processed_segments.append({
            "type": "speech",
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
            "audio": audio_bytes
        })
    
    # Add silence segments
    all_segments = []
    last_end = 0
    for segment in processed_segments:
        if segment["start"] > last_end:
            all_segments.append({
                "type": "silence",
                "start": last_end,
                "end": segment["start"],
                "duration": segment["start"] - last_end
            })
        all_segments.append(segment)
        last_end = segment["end"]
    
    # Add final silence if needed
    total_duration = len(pcm_data) / (2 * sample_rate)  # 2 bytes per sample
    if last_end < total_duration:
        all_segments.append({
            "type": "silence",
            "start": last_end,
            "end": total_duration,
            "duration": total_duration - last_end
        })
    
    # Clean up temporary file
    os.unlink(wav_file)
    
    return all_segments, pcm_data, sample_rate

def calculate_silence_between_speakers(segments):
    silence_data = []
    last_speaker = None
    last_end_time = 0

    for segment in segments:
        if segment['type'] == 'speech':
            if last_speaker and segment['speaker'] != last_speaker:
                silence_duration = segment['start'] - last_end_time
                if last_speaker == 'USER':
                    silence_data.append({
                        'From': last_speaker,
                        'To': segment['speaker'],
                        'Silence Duration': f"{silence_duration:.2f}s"
                    })
            last_speaker = segment['speaker']
            last_end_time = segment['end']

    return silence_data

def main():
    st.title("Audio Analyzer")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a"])
    
    # Add checkbox and input field for minimum segment length
    col1, col2 = st.columns(2)
    with col1:
        ignore_short_segments = st.checkbox("Ignore short segments", value=True)
    with col2:
        min_segment_length = st.number_input("Minimum segment length (seconds)", value=0.5, min_value=0.0, step=0.1)
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        segments, audio, sr = analyze_audio(uploaded_file)
        
        # Filter segments based on minimum length if checkbox is checked
        if ignore_short_segments:
            segments = [seg for seg in segments if seg['duration'] >= min_segment_length]
        
        st.write("## Audio Segments")
        for i, segment in enumerate(segments):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            
            with col1:
                st.write(f"Segment {i+1}: {segment['type'].capitalize()}")
            
            with col2:
                st.write(f"{segment['duration']:.2f}s")
            
            with col3:
                if segment['type'] == 'speech':
                    # Convert to float32 audio data
                    segment_audio = np.frombuffer(segment['audio'], dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Convert to WAV
                    with io.BytesIO() as buffer:
                        sf.write(buffer, segment_audio, sr, format="wav")
                        st.audio(buffer.getvalue(), format="audio/wav")
                else:
                    st.write("N/A")
            
            with col4:
                if segment['type'] == "speech":
                    segment['speaker'] = st.selectbox(f"Speaker for segment {i+1}", ["BOT", "USER"])
                else:
                    segment['speaker'] = "N/A"
                    st.write("N/A")
            
            st.write("---")
        
        # Calculate and display silence between speakers
        silence_data = calculate_silence_between_speakers(segments)
        if silence_data:
            st.write("## Silence Between Speakers")
            df = pd.DataFrame(silence_data)
            st.table(df)
        else:
            st.write("No silence between different speakers detected.")

if __name__ == "__main__":
    main()