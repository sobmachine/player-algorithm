#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mutagen.flac import FLAC
import os
import sys
import time
import shutil
import vlc
import numpy as np
import librosa
from rich.console import Console
from rich.table import Table
import json

console = Console()
CACHE_FILE = "playlist_cache.json"

##########################
# Utilities
##########################

def format_time(ms):
    seconds = int(ms / 1000)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}:{secs:02d}"

def find_all_flacs(root_folder):
    flac_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.lower().endswith(".flac"):
                flac_files.append(os.path.join(dirpath, f))
    return sorted(flac_files)

##########################
# Mood generation
##########################

def auto_generate_mood(file_path):
    """Generate mood vector from audio features"""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        return [
            float(np.clip(tempo/300, 0, 1)),                # valence
            float(np.clip(rms, 0, 1)),                      # energy
            float(np.clip(spectral_centroid/10000, 0, 1)),  # danceability
            float(np.clip(tempo/300, 0, 1))                 # happiness
        ]
    except Exception:
        return [0.5, 0.5, 0.5, 0.5]  # fallback neutral mood

##########################
# Metadata reading
##########################

def get_flac_metadata(file_path):
    """Read metadata from FLAC using mutagen"""
    audio = FLAC(file_path)
    title = audio.get("title", ["Unknown Title"])[0]
    artist = audio.get("artist", ["Unknown Artist"])[0]
    album = audio.get("album", ["Unknown Album"])[0]
    full_title = f"{album} -> {title} — {artist}"
    return {"title": title, "artist": artist, "album": album, "full_title": full_title}

##########################
# Playlist display
##########################

def display_playlist(songs, current_index, window=5):
    total = len(songs)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", width=5)
    table.add_column("Title", style="green")
    table.add_column("Mood Vector", style="yellow")
    table.add_column("Artist", style="blue")
    table.add_column("Album", style="magenta")

    start = max(0, current_index - window//2)
    end = min(total, start + window)
    start = max(0, end - window)

    for i in range(start, end):
        song = songs[i]
        mood_str = ", ".join(f"{v:.2f}" for v in song["mood_vector"])
        table.add_row(str(i+1), song["title"], mood_str, song["artist"], song["album"])
    
    console.clear()
    console.print("\n🎧 [bold yellow]Mood-Aware Shuffle[/bold yellow]\n")
    console.print(table)
    console.print("\n")

##########################
# Playback
##########################

def play_song(song):
    player = vlc.MediaPlayer(song["file"])
    player.play()
    time.sleep(0.01)

    total = player.get_length()
    retries = 0
    while total <= 0 and retries < 50:
        time.sleep(0.01)
        total = player.get_length()
        retries += 1
    if total <= 0:
        total = 1

    try:
        width = shutil.get_terminal_size().columns
        last_display = ""
        while True:
            state = player.get_state()
            if state in [vlc.State.Ended, vlc.State.Error]:
                break
            current = player.get_time()
            progress = min(current / total, 1.0)

            bar_width = width - 70 - len(song['full_title'])
            filled = int(progress * bar_width)
            bar = "=" * filled + " " * (bar_width - filled)
            progress_text = f"[{bar}] {int(progress*100):3d}% | {format_time(current)}/{format_time(total)} | {song['full_title']}"
            
            if progress_text != last_display:
                print(f"\r{progress_text}", end='', flush=True)
                last_display = progress_text
            time.sleep(0.05)
        print()
    except KeyboardInterrupt:
        print()
    finally:
        player.stop()

##########################
# Caching
##########################

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cache(songs):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)

##########################
# Mood-based Playlist
##########################

def ask_user_mood():
    """Ask user for a 4-value mood or randomize"""
    console.print("[bold cyan]Enter your current mood as 4 numbers (0–1):[/bold cyan]")
    console.print("Format: valence, energy, danceability, happiness")
    console.print("Or press [bold yellow]Enter[/bold yellow] to randomize.\n")
    inp = input("Your mood: ").strip()
    if not inp:
        mood = np.random.rand(4).tolist()
        console.print(f"[italic green]Randomized mood:[/italic green] {mood}\n")
        return mood
    try:
        parts = [float(x) for x in inp.split(",")]
        if len(parts) != 4:
            raise ValueError
        return [np.clip(x, 0, 1) for x in parts]
    except ValueError:
        console.print("[bold red]Invalid format — using neutral mood.[/bold red]")
        return [0.5, 0.5, 0.5, 0.5]

def mood_similarity(user_mood, song_mood):
    """Euclidean distance in mood space"""
    return np.linalg.norm(np.array(user_mood) - np.array(song_mood))

def calculate_probabilities(songs, user_mood, temperature=0.5):
    """Convert distances to probabilities"""
    distances = np.array([mood_similarity(user_mood, s["mood_vector"]) for s in songs])
    similarity = np.exp(-distances / temperature)
    probabilities = similarity / similarity.sum()
    return probabilities

def generate_playlist_order(songs, user_mood, temperature=0.5):
    """Random but mood-aware ordering"""
    probabilities = calculate_probabilities(songs, user_mood, temperature)
    order = np.random.choice(len(songs), size=len(songs), replace=False, p=probabilities)
    return order.tolist()
    

##########################
# Main
##########################

def main():
    root_folder = sys.argv[1] if len(sys.argv) > 1 else "."
    force_rescan = "--rescan" in sys.argv

    songs = None
    if not force_rescan:
        songs = load_cache()

    if not songs:
        files = find_all_flacs(root_folder)
        songs = []
        for f in files:
            metadata = get_flac_metadata(f)
            mood = auto_generate_mood(f)
            songs.append({
                "file": f,
                "title": metadata["title"],
                "artist": metadata["artist"],
                "album": metadata["album"],
                "full_title": metadata["full_title"],
                "mood_vector": mood
            })
        save_cache(songs)

    console.print(f"[bold magenta]Loaded {len(songs)} songs.[/bold magenta]\n")
    user_mood = ask_user_mood()
    playlist_order = generate_playlist_order(songs, user_mood, temperature=0.6)

    for idx in playlist_order:
        display_playlist(songs, idx)
        play_song(songs[idx])

if __name__ == "__main__":
    main()
