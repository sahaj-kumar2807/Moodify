from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import time
import requests
from dotenv import load_dotenv


def get_spotify_access_token():

    global spotify_token

    if spotify_token:
        return spotify_token

    url = "https://accounts.spotify.com/api/token"

    response = requests.post(
        url,
        data={
            "grant_type": "client_credentials"
        },
        auth=(
            SPOTIFY_CLIENT_ID,
            SPOTIFY_CLIENT_SECRET
        )
    )

    if response.status_code != 200:
        print("Spotify authentication failed:")
        print(response.text)
        return None

    spotify_token = response.json()["access_token"]

    return spotify_token


def find_spotify_track(song_name, artist_name):

    cache_key = (
        f"{song_name.strip().lower()}|"
        f"{artist_name.strip().lower()}"
    )

    if cache_key in spotify_cache:
        return spotify_cache[cache_key]

    token = get_spotify_access_token()

    if not token:
        return None

    url = "https://api.spotify.com/v1/search"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "q": f'track:"{song_name}" artist:"{artist_name}"',
        "type": "track",
        "limit": 1
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=5
    )

    if response.status_code != 200:
        print("Spotify search failed:")
        print(response.text)
        return None

    tracks = response.json()["tracks"]["items"]

    if not tracks:
        spotify_cache[cache_key] = None
        return None

    track = tracks[0]

    result = {
        "spotify_id": track["id"],
        "spotify_url": track["external_urls"]["spotify"],
        "spotify_uri": track["uri"]
    }

    spotify_cache[cache_key] = result

    return result
# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

# Spotify data is cached so repeated lookups are fast.
spotify_token = None
spotify_cache = {}

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# =========================================================
# LOAD MUSIC DATA
# =========================================================

Music_Player = pd.read_csv(
    "./dataset/data_moods.csv"
)


# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

MODEL_PATH = "./models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

landmarker = None


def get_landmarker():

    global landmarker

    if landmarker is None:

        options = FaceLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=MODEL_PATH
            ),

            running_mode=VisionRunningMode.VIDEO,

            num_faces=1,

            output_face_blendshapes=True
        )

        landmarker = (
            FaceLandmarker.create_from_options(
                options
            )
        )

        print("===================================")
        print("       MOODIFY MEDIAPIPE READY")
        print("===================================")

    return landmarker


# =========================================================
# MOOD INFORMATION
# =========================================================

MOOD_INFO = {

    "Happy": {
        "emoji": "😊",
        "description": "Your vibe is looking good."
    },

    "Sad": {
        "emoji": "😢",
        "description": "Let's find something to lift your mood."
    },

    "Angry": {
        "emoji": "😠",
        "description": "You look a little intense right now."
    },

    "Surprise": {
        "emoji": "😮",
        "description": "Something caught your attention."
    },

    "Neutral": {
        "emoji": "😐",
        "description": "Calm, balanced and steady."
    }

}


# =========================================================
# MOOD → MUSIC MOOD
# =========================================================

def get_music_mood(mood):

    if mood == "Happy":
        return "Happy"

    elif mood == "Sad":
        return "Sad"

    elif mood == "Angry":
        return "Calm"

    elif mood == "Surprise":
        return "Energetic"

    else:
        return "Calm"
# =========================================================
# GET MUSIC RECOMMENDATIONS
# =========================================================

def get_music_recommendations(mood):

    songs = Music_Player[
        Music_Player["mood"] == mood
    ]

    songs = songs.sort_values(
        by="popularity",
        ascending=False
    ).head(5)

    recommendations = songs[
        [
            "album",
            "artist",
            "name",
            "popularity",
            "release_date"
        ]
    ].to_dict(
        orient="records"
    )

    # IMPORTANT:
    # Do NOT call Spotify here.
    # This function runs during every /detect_mood request.
    # Spotify lookups are handled separately so emotion detection stays fast.

    return recommendations


# =========================================================
# MOOD ENGINE
# =========================================================

def determine_mood(scores):

    # =====================================================
    # GET FACIAL FEATURES
    # =====================================================

    smile = (
        scores.get("mouthSmileLeft", 0) +
        scores.get("mouthSmileRight", 0)
    ) / 2

    frown = (
        scores.get("mouthFrownLeft", 0) +
        scores.get("mouthFrownRight", 0)
    ) / 2

    brow_down = (
        scores.get("browDownLeft", 0) +
        scores.get("browDownRight", 0)
    ) / 2

    brow_inner_up = scores.get(
        "browInnerUp",
        0
    )

    eye_wide = (
        scores.get("eyeWideLeft", 0) +
        scores.get("eyeWideRight", 0)
    ) / 2

    cheek_squint = (
        scores.get("cheekSquintLeft", 0) +
        scores.get("cheekSquintRight", 0)
    ) / 2

    eye_squint = (
        scores.get("eyeSquintLeft", 0) +
        scores.get("eyeSquintRight", 0)
    ) / 2

    mouth_press = (
        scores.get("mouthPressLeft", 0) +
        scores.get("mouthPressRight", 0)
    ) / 2

    nose_sneer = (
        scores.get("noseSneerLeft", 0) +
        scores.get("noseSneerRight", 0)
    ) / 2

    jaw_open = scores.get(
        "jawOpen",
        0
    )


    # =====================================================
    # HAPPY
    # =====================================================

    happy_score = (
        smile * 0.70 +
        cheek_squint * 0.30
    )


    # =====================================================
    # SAD
    # =====================================================

    sad_score = (
        frown * 0.65 +
        brow_inner_up * 0.25 +
        mouth_press * 0.10
    )


    # =====================================================
    # ANGRY
    # =====================================================

    angry_score = (
        brow_down * 0.60 +
        eye_squint* 0.10 +
        mouth_press * 1
    )


    # =====================================================
    # SURPRISE
    # =====================================================

    surprise_score = (
        jaw_open * 0.50 +
        eye_wide * 0.30 +
        brow_inner_up * 0.50
    )


    # =====================================================
    # DEBUG
    # =====================================================

    print("\n========== FACE FEATURES ==========")

    print("Smile:", round(smile, 2))
    print("Frown:", round(frown, 2))
    print("Brow Down:", round(brow_down, 2))
    print("Brow Inner Up:", round(brow_inner_up, 2))
    print("Eye Wide:", round(eye_wide, 2))
    print("Eye Squint:", round(eye_squint, 2))
    print("Jaw Open:", round(jaw_open, 2))
    # print("Mouth Press:", round(mouth_press, 2))
    # print("Nose Sneer:", round(nose_sneer, 2))

    # print("\n========== MOOD SCORES ==========")

    # print("Happy:", round(happy_score, 2))
    # print("Sad:", round(sad_score, 2))
    # print("Angry:", round(angry_score, 2))
    # print("Surprise:", round(surprise_score, 2))

    print("===================================")


    # =====================================================
    # PRIORITY 1 — SURPRISE
    # =====================================================

    if (
        jaw_open > 0.20
        and brow_inner_up > 0.15
    ):

        return "Surprise", min(
            surprise_score,
            1.0
        )


    # =====================================================
# =====================================================
# PRIORITY 2 — SAD
# =====================================================

    if (
        smile < 0.15
        and eye_squint > 0.27
        and frown > 0.05
        and brow_down < 0.18
):

        sad_confidence = min(
            (
                eye_squint * 0.45 +
                frown * 0.35 +
                (1.0 - brow_down) * 0.20
            ),
            1.0
        )

        return "Sad", sad_confidence

    


    # =====================================================
    # PRIORITY 3 — ANGRY
    # =====================================================

    if (
        brow_down > 0.20
        and smile < 0.20
        and jaw_open < 0.40
):

        angry_confidence = min(
            (
                brow_down * 0.70 +
                mouth_press * 0.20 +
                nose_sneer * 0.10
            ),
            1.0
        )

        return "Angry", angry_confidence


    # =====================================================
    # PRIORITY 4 — HAPPY
    # =====================================================

    if smile > 0.25:

        return "Happy", min(
            happy_score,
            1.0
        )


    # =====================================================
    # OTHERWISE NEUTRAL
    # =====================================================

    return "Neutral", 0.60


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# SPOTIFY TRACK LOOKUP
# =========================================================
# This route is separate from /detect_mood.
# It is only called when the frontend actually needs Spotify data.

@app.route(
    "/spotify_track",
    methods=["POST"]
)
def spotify_track():

    song_name = request.form.get("song_name", "").strip()
    artist_name = request.form.get("artist_name", "").strip()

    if not song_name or not artist_name:

        return jsonify({
            "error": "Song name and artist are required"
        }), 400

    result = find_spotify_track(
        song_name,
        artist_name
    )

    if result is None:

        return jsonify({
            "spotify_id": None,
            "spotify_url": None,
            "spotify_uri": None
        })

    return jsonify(result)


# =========================================================
# REAL-TIME MOOD DETECTION
# =========================================================

@app.route(
    "/detect_mood",
    methods=["POST"]
)
def detect_mood():

    # -----------------------------------------------------
    # CHECK FRAME
    # -----------------------------------------------------

    if "frame" not in request.files:

        return jsonify({

            "error":
                "No frame received"

        }), 400


    file = request.files["frame"]


    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_bytes = file.read()


    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        return jsonify({

            "error":
                "Could not read frame"

        }), 400


    # -----------------------------------------------------
    # BGR → RGB
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # MEDIAPIPE IMAGE
    # -----------------------------------------------------

    mp_image = mp.Image(

        image_format=
            mp.ImageFormat.SRGB,

        data=rgb_frame

    )


    # -----------------------------------------------------
    # TIMESTAMP
    # -----------------------------------------------------

    timestamp = int(
        time.time() * 1000
    )


    # -----------------------------------------------------
    # DETECT
    # -----------------------------------------------------

    landmarker_instance = get_landmarker()

    result = (
        landmarker_instance.detect_for_video(
            mp_image,
            timestamp
        )
    )


    # -----------------------------------------------------
    # NO FACE
    # -----------------------------------------------------

    if not result.face_blendshapes:

        return jsonify({

            "face_detected": False,

            "mood": "Neutral",

            "emoji": "👀",

            "description":
                "Make sure your face is visible.",

            "confidence": 0,

            "recommendations": []

        })


    # -----------------------------------------------------
    # GET BLENDSHAPES
    # -----------------------------------------------------

    blendshapes = (
        result.face_blendshapes[0]
    )


    scores = {}


    for category in blendshapes:

        scores[
            category.category_name
        ] = category.score


    # -----------------------------------------------------
    # DETERMINE MOOD
    # -----------------------------------------------------

    mood, confidence = (
        determine_mood(scores)
    )


    # -----------------------------------------------------
    # GET MOOD INFORMATION
    # -----------------------------------------------------

    info = MOOD_INFO[mood]


    # -----------------------------------------------------
    # GET MUSIC
    # -----------------------------------------------------

    music_mood = get_music_mood(
        mood
    )


    recommendations = (
        get_music_recommendations(
            music_mood
        )
    )


    # -----------------------------------------------------
    # SEND TO FRONTEND
    # -----------------------------------------------------

    return jsonify({

        "face_detected":
            True,

        "mood":
            mood,

        "emoji":
            info["emoji"],

        "description":
            info["description"],

        "confidence":
            round(
                confidence * 100,
                1
            ),

        "recommendations":
            recommendations

    })


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=8080
    )