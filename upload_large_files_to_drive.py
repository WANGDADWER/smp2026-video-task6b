#!/usr/bin/env python3
"""
Upload large feature files to Google Drive (for README download links).

Usage (run on a machine with a browser):
  python upload_large_files_to_drive.py

This uploads:
  - features/vit_base_frame8.csv (684 MB)
  - features/clip_vitl14_frame8_temporal.csv (988 MB)

To your Google Drive and prints shareable download links.
Paste those links into README.md under "External Resources".
"""

from __future__ import annotations

import os, sys
from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# ── Paths ────────────────────────────────────────────────────────────────
FEATURE_DIR = Path("/user_home/jingweikai/smp_vedio/submission_package/features")
FILES = [
    ("vit_base_frame8.csv", "ViT-base frame embeddings (684 MB)"),
    ("clip_vitl14_frame8_temporal.csv", "CLIP ViT-L/14 temporal embeddings (988 MB)"),
]


def main():
    print("=" * 60)
    print("Upload Large Feature Files to Google Drive")
    print("=" * 60)
    print()
    print("This will upload 2 large feature files (~1.7 GB total)")
    print("to your Google Drive (wyc294723847@gmail.com).")
    print()
    print("A browser window will open for authentication.")
    print()

    # Authenticate
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Create folder
    folder_name = "smp2026-video-task6b-features"
    folder_list = drive.ListFile({
        'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if folder_list:
        folder_id = folder_list[0]['id']
        print(f"Using existing folder: {folder_name} (id={folder_id})")
    else:
        folder = drive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        })
        folder.Upload()
        folder_id = folder['id']
        print(f"Created folder: {folder_name} (id={folder_id})")

    # Upload files
    links = {}
    for filename, description in FILES:
        filepath = FEATURE_DIR / filename
        if not filepath.exists():
            print(f"\n[SKIP] {filename} not found at {filepath}")
            continue

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"\n[UPLOAD] {filename} ({size_mb:.0f} MB) — {description}")
        print("  Uploading... (this may take several minutes)")

        drive_file = drive.CreateFile({
            'title': filename,
            'parents': [{'id': folder_id}]
        })
        drive_file.SetContentFile(str(filepath))
        drive_file.Upload()

        # Make shareable
        drive_file.InsertPermission({
            'type': 'anyone',
            'value': 'anyone',
            'role': 'reader'
        })

        # Get download link
        file_id = drive_file['id']
        download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
        links[filename] = download_link
        print(f"  Done! Download link: {download_link}")

    # Print all links
    print("\n" + "=" * 60)
    print("ADD THESE LINKS TO README.md:")
    print("=" * 60)
    for filename, link in links.items():
        print(f"\n| {filename} | {FILES[[f[0] for f in FILES].index(filename)][1]} |")
        print(f"| Download | [{link}]({link}) |")

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
