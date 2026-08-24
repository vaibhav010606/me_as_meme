#!/usr/bin/env python3
"""
Meme Catalog Builder
Scans the meme folder, parses filenames to extract emotion/action tags,
and generates meme_catalog.json for the Meme Expression Matcher web app.
"""

import os
import re
import json
import sys

# The root folder containing the memes
MEME_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(MEME_DIR, "meme_catalog.json")

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}

# ──────────────────────────────────────────────
# Emotion keyword mapping
# Maps words found in filenames → face-api.js emotion categories
# face-api.js emotions: happy, sad, angry, surprised, fearful, disgusted, neutral
# Additional custom categories: confused, crying, flexing, pointing, proud, embarrassed, bored
# ──────────────────────────────────────────────

EMOTION_KEYWORDS = {
    # HAPPY
    "happy": ["happy"],
    "smile": ["happy"],
    "smiling": ["happy"],
    "laugh": ["happy"],
    "laughing": ["happy"],
    "joy": ["happy"],
    "joyful": ["happy"],
    "celebration": ["happy"],
    "celebrate": ["happy"],
    "champagne": ["happy"],
    "party": ["happy"],
    "dancing": ["happy"],
    "dance": ["happy"],
    "danser": ["happy"],
    "content": ["happy"],
    "heureux": ["happy"],
    "brilliant": ["happy"],
    "winking": ["happy"],
    "wink": ["happy"],
    "kissing": ["happy"],
    "love": ["happy"],
    "crush": ["happy", "sad"],
    "thumbs up": ["happy"],
    "approve": ["happy"],
    "approved": ["happy"],
    "yes": ["happy"],
    "win": ["happy"],
    "winner": ["happy"],
    "glow": ["happy"],
    "glowing": ["happy"],
    "proud": ["happy"],
    "flex": ["happy", "flexing"],
    "strong": ["happy", "flexing"],
    "chad": ["happy", "proud"],
    "fine": ["happy", "neutral"],
    "nice": ["happy"],
    "good": ["happy"],
    "great": ["happy"],
    "best": ["happy"],
    "better": ["happy"],
    "wooo": ["happy", "surprised"],
    "hooray": ["happy"],

    # SAD
    "sad": ["sad"],
    "crying": ["sad"],
    "cry": ["sad"],
    "crie": ["sad"],
    "pleure": ["sad"],
    "pleurer": ["sad"],
    "tears": ["sad"],
    "tear": ["sad"],
    "larme": ["sad"],
    "weeping": ["sad"],
    "weep": ["sad"],
    "depressed": ["sad"],
    "depression": ["sad"],
    "lonely": ["sad"],
    "alone": ["sad"],
    "heartbreak": ["sad"],
    "broken": ["sad"],
    "funeral": ["sad"],
    "funerals": ["sad"],
    "tombe": ["sad"],
    "rip": ["sad"],
    "pain": ["sad"],
    "suffering": ["sad"],
    "withered": ["sad"],
    "triste": ["sad"],
    "sombre": ["sad"],
    "dark": ["sad"],
    "tragic": ["sad"],
    "regret": ["sad"],
    "miss": ["sad"],
    "missing": ["sad"],
    "wolverine": ["sad"],
    "volume up": ["sad"],

    # ANGRY
    "angry": ["angry"],
    "anger": ["angry"],
    "mad": ["angry"],
    "rage": ["angry"],
    "furious": ["angry"],
    "fight": ["angry"],
    "fighting": ["angry"],
    "bagarre": ["angry"],
    "frappe": ["angry"],
    "strike": ["angry"],
    "beats": ["angry"],
    "beating": ["angry"],
    "slap": ["angry"],
    "punch": ["angry"],
    "destroy": ["angry"],
    "violence": ["angry"],
    "yelling": ["angry"],
    "yell": ["angry"],
    "scream": ["angry"],
    "screaming": ["angry"],
    "attack": ["angry"],
    "agression": ["angry"],
    "kill": ["angry"],
    "weapon": ["angry"],
    "weapons": ["angry"],
    "gun": ["angry"],
    "knife": ["angry"],
    "sword": ["angry"],
    "vein": ["angry"],
    "veine": ["angry"],
    "forehead": ["angry"],
    "headache": ["angry"],
    "douleur": ["angry"],
    "cursed": ["angry"],
    "toxic": ["angry"],
    "tabassage": ["angry"],

    # SURPRISED
    "surprised": ["surprised"],
    "surprise": ["surprised"],
    "shocked": ["surprised"],
    "shock": ["surprised"],
    "omg": ["surprised"],
    "wtf": ["surprised"],
    "what": ["surprised"],
    "whoa": ["surprised"],
    "wow": ["surprised"],
    "unexpected": ["surprised"],
    "unsettled": ["surprised"],
    "disturbed": ["surprised"],
    "wait": ["surprised"],
    "twist": ["surprised"],
    "plot twist": ["surprised"],
    "reverse": ["surprised"],
    "impossible": ["surprised"],

    # FEARFUL
    "scared": ["fearful"],
    "scary": ["fearful"],
    "fear": ["fearful"],
    "fearful": ["fearful"],
    "terrified": ["fearful"],
    "terror": ["fearful"],
    "horror": ["fearful"],
    "peur": ["fearful"],
    "effrayé": ["fearful"],
    "creepy": ["fearful"],
    "nightmare": ["fearful"],
    "panic": ["fearful"],
    "run": ["fearful"],
    "running away": ["fearful"],
    "hide": ["fearful"],
    "hiding": ["fearful"],

    # DISGUSTED
    "disgusted": ["disgusted"],
    "disgust": ["disgusted"],
    "gross": ["disgusted"],
    "cringe": ["disgusted"],
    "ew": ["disgusted"],
    "eww": ["disgusted"],
    "ugly": ["disgusted"],
    "worst": ["disgusted"],
    "hate": ["disgusted"],
    "trash": ["disgusted"],

    # NEUTRAL / THINKING / CONFUSED
    "neutral": ["neutral"],
    "blank": ["neutral"],
    "empty": ["neutral"],
    "template": ["neutral"],
    "thinking": ["neutral", "confused"],
    "pense": ["neutral", "confused"],
    "think": ["neutral", "confused"],
    "confused": ["confused"],
    "confusion": ["confused"],
    "explain": ["confused"],
    "trying": ["confused"],
    "understand": ["confused"],
    "question": ["confused"],
    "doubt": ["confused"],
    "huh": ["confused"],
    "hmm": ["confused"],

    # POINTING / GESTURES
    "pointing": ["pointing"],
    "point": ["pointing"],
    "doigt": ["pointing"],
    "finger": ["pointing"],
    "showing": ["pointing"],
    "look": ["pointing"],
    "regarde": ["pointing"],

    # FLEXING / POWER POSES
    "flexing": ["flexing"],
    "muscle": ["flexing"],
    "buff": ["flexing"],
    "swole": ["flexing"],
    "power": ["flexing"],
    "dominant": ["flexing"],
    "gigachad": ["flexing"],

    # FACEPALM / DISAPPOINTMENT
    "facepalm": ["disappointed"],
    "disappointed": ["disappointed"],
    "disappointment": ["disappointed"],
    "sigh": ["disappointed"],
    "really": ["disappointed"],
    "seriously": ["disappointed"],
    "bruh": ["disappointed"],
    "weak": ["disappointed"],

    # SPECIFIC MEME EXPRESSIONS
    "stare": ["neutral", "surprised"],
    "staring": ["neutral", "surprised"],
    "squinting": ["angry", "confused"],
    "eyes": ["surprised"],
    "wide eyes": ["surprised", "fearful"],
    "side eye": ["disgusted", "confused"],
    "smirk": ["happy", "proud"],
    "grin": ["happy"],
    "grimace": ["disgusted", "fearful"],
    "poker face": ["neutral"],
    "dead inside": ["sad", "neutral"],
    "sarcastic": ["happy", "disgusted"],
    "ironic": ["happy", "confused"],

    # ACTIONS
    "write": ["neutral", "pointing"],
    "read": ["neutral", "confused"],
    "draw": ["neutral"],
    "choose": ["confused"],
    "choisir": ["confused"],
    "decision": ["confused"],
    "upgrade": ["happy"],
    "downgrade": ["sad"],
    "trade": ["neutral", "confused"],
    "offer": ["happy"],
    "paid": ["surprised", "confused"],
    "money": ["happy", "surprised"],
    "fire": ["fearful", "angry"],
    "burning": ["fearful"],
    "explosion": ["surprised", "fearful"],

    # SPECIFIC MEME CHARACTERS (expression associations)
    "drake": ["happy", "disgusted"],
    "doge": ["happy", "surprised"],
    "pepe": ["sad", "happy"],
    "wojak": ["sad"],
    "soyjak": ["surprised", "pointing"],
    "thanos": ["angry", "sad"],
    "pikachu": ["surprised"],
    "spongebob": ["happy", "confused"],
    "shrek": ["happy"],
    "gru": ["surprised", "angry"],
    "oogway": ["neutral", "happy"],
    "rickroll": ["happy"],
    "stonks": ["happy"],
}

# Also map some multi-word phrases
PHRASE_KEYWORDS = {
    "this is fine": ["happy", "fearful"],
    "it ain't much": ["happy", "proud"],
    "change my mind": ["neutral", "proud"],
    "am i a joke": ["sad", "angry"],
    "trust nobody": ["fearful", "angry"],
    "you guys are getting paid": ["surprised", "confused"],
    "draw 25": ["angry", "confused"],
    "not sure if": ["confused"],
    "one does not simply": ["neutral", "serious"],
    "write that down": ["surprised", "happy"],
    "understandable have a great day": ["neutral", "happy"],
    "is this a": ["confused"],
    "what if i told you": ["neutral", "surprised"],
    "well yes but actually no": ["confused"],
    "wait that's illegal": ["surprised", "angry"],
    "outstanding move": ["happy", "surprised"],
    "reality is often disappointing": ["sad", "disappointed"],
    "it's all": ["surprised"],
    "always has been": ["surprised", "neutral"],
    "corporate needs you to find the difference": ["confused", "neutral"],
    "same picture": ["confused", "neutral"],
    "impossible": ["surprised"],
    "have a great day": ["happy"],
    "distracted boyfriend": ["surprised", "confused"],
    "galaxy brain": ["happy", "proud"],
    "woman yelling at cat": ["angry", "neutral"],
    "expanding brain": ["happy", "surprised"],
    "first half": ["surprised"],
    "had us": ["surprised"],
    "madman": ["surprised", "angry"],
    "task failed successfully": ["confused", "happy"],
    "they don't know": ["sad"],
    "what did it cost": ["sad"],
    "you were supposed to": ["angry", "sad"],
}


def clean_filename(filename):
    """Remove extension and clean up the filename for tag extraction."""
    name = os.path.splitext(filename)[0]
    # Replace common separators with spaces
    name = name.replace('_', ' ').replace('-', ' ').replace(',', ' ')
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def extract_tags(filename):
    """Extract emotion/action tags from a meme filename."""
    cleaned = clean_filename(filename).lower()
    tags = set()
    emotions = set()

    # Check multi-word phrases first
    for phrase, emotion_list in PHRASE_KEYWORDS.items():
        if phrase in cleaned:
            tags.add(phrase.replace(' ', '_'))
            for em in emotion_list:
                emotions.add(em)

    # Check individual keywords
    words = re.split(r'[\s\-_,;:!?()]+', cleaned)
    for word in words:
        word = word.strip().lower()
        if word in EMOTION_KEYWORDS:
            for em in EMOTION_KEYWORDS[word]:
                emotions.add(em)
            tags.add(word)

    # If no emotions found, default to neutral
    if not emotions:
        emotions.add("neutral")

    return sorted(tags), sorted(emotions)


def scan_memes(root_dir):
    """Scan directory recursively for meme images and build catalog."""
    catalog = []
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                skipped += 1
                continue
            if ext == '.svg':
                skipped += 1
                continue

            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir).replace('\\', '/')

            tags, emotions = extract_tags(filename)

            catalog.append({
                "path": rel_path,
                "filename": filename,
                "tags": tags,
                "emotions": emotions,
            })

    print(f"[OK] Scanned {len(catalog)} meme images ({skipped} non-image files skipped)")
    return catalog


def main():
    print("=" * 60)
    print("  Meme Catalog Builder")
    print("=" * 60)
    print(f"  Scanning: {MEME_DIR}")
    print()

    catalog = scan_memes(MEME_DIR)

    # Stats
    emotion_counts = {}
    for meme in catalog:
        for em in meme["emotions"]:
            emotion_counts[em] = emotion_counts.get(em, 0) + 1

    print("\n  Emotion Distribution:")
    for em, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count // 5, 40)
        print(f"    {em:15s} {count:4d}  {bar}")

    # Write catalog
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Catalog saved to: {OUTPUT_FILE}")
    print(f"    Total memes cataloged: {len(catalog)}")


if __name__ == "__main__":
    main()
