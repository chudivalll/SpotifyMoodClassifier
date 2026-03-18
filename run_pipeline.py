#!/usr/bin/env python3
"""
Spotify Mood Classifier — main entry point.

Usage:
    python run_pipeline.py                          # run on sample data (no API keys needed)
    python run_pipeline.py --mode live --playlist <ID>  # fetch from Spotify API
    python run_pipeline.py --data my_tracks.csv     # run on your own CSV
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.ml_pipeline import SpotifyMLPipeline

import argparse

parser = argparse.ArgumentParser(description='Spotify Mood Classifier')
parser.add_argument('--mode', choices=['sample', 'live'], default='sample',
                    help='sample = bundled dataset (default), live = Spotify API')
parser.add_argument('--playlist', type=str, default=None,
                    help='playlist ID or URL (for live mode)')
parser.add_argument('--data', type=str, default=None,
                    help='path to your own CSV with track data')
args = parser.parse_args()

if args.data:
    pipeline = SpotifyMLPipeline(data_path=args.data)
    pipeline.run(mode='sample')
elif args.mode == 'live':
    if not args.playlist:
        print("Error: --playlist is required for live mode")
        print("Example: python run_pipeline.py --mode live --playlist 6jgCEkpKSc7LQI8ZWdAVr6")
        sys.exit(1)
    pipeline = SpotifyMLPipeline(playlist_id=args.playlist)
    pipeline.run(mode='live')
else:
    pipeline = SpotifyMLPipeline()
    pipeline.run(mode='sample')
