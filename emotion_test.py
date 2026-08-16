from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time
import os


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# MUSIC DATA
# =========================================================

Music_Player = pd.read_csv(
    "./dataset/data_moods.csv"
)


# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

MODEL_PATH = "./models/face_landmarker.task"


BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


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


# =========================================================
# MOOD DESCRIPTIONS
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
# MOOD → MUSIC
# =========================================================

def get_music_mood(mood):

    if mood == "Happy":
        return "Happy"

    if mood == "Sad":
        return "Sad"

    if mood == "Angry":
        return "Calm"

    if mood == "Surprise":
        return "Energetic"

    return "Calm"


# =========================================================
# GET SONGS
# =========================================================

def get_music_recommendations(mood):

    songs = Music_Player[
        Music_Player["mood"] == mood
    ]

    songs = songs.sort_values(
        by="popularity",
        ascending=False
    ).head(5)

    return songs[
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


# =========================================================
# MOOD ENGINE
# =========================================================

def determine_mood(scores):

    smile_left = scores.get(
        "mouthSmileLeft",
        0
    )

    smile_right = scores.get(
        "mouthSmileRight",
        0
    )

    frown_left = scores.get(
        "mouthFrownLeft",
        0
    )

    frown_right = scores.get(
        "mouthFrownRight",
        0
    )

    jaw_open = scores.get(
        "jawOpen",
        0
    )

    brow_down_left = scores.get(
        "browDownLeft",
        0
    )

    brow_down_right = scores.get(
        "browDownRight",
        0
    )

    brow_inner_up = scores.get(
        "browInnerUp",
        0
    )

    eye_wide_left = scores.get(
        "eyeWideLeft",
        0
    )

    eye_wide_right = scores.get(
        "eyeWideRight",
        0
    )


    # Average facial scores

    smile = (
        smile_left +
        smile_right
    ) / 2


    frown = (
        frown_left +
        frown_right
    ) / 2


    brow_down = (
        brow_down_left +
        brow_down_right
    ) / 2


    eye_wide = (
        eye_wide_left +
        eye_wide_right
    ) / 2


    # =====================================================
    # HAPPY
    # =====================================================

    if smile > 0.35:

        confidence = min(
            smile,
            1.0
        )

        return "Happy", confidence


    # =====================================================
    # SURPRISE
    # =====================================================

    if (
        jaw_open > 0.40
        and brow_inner_up > 0.20
        and eye_wide > 0.20
    ):

        confidence = min(
            (
                jaw_open +
                brow_inner_up +
                eye_wide
            ) / 3,
            1.0
        )

        return "Surprise", confidence


    # =====================================================
    # ANGRY
    # =====================================================

    if (
        brow_down > 0.30
        and smile < 0.25
    ):

        confidence = min(
            brow_down,
            1.0
        )

        return "Angry", confidence


    # =====================================================
    # SAD
    # =====================================================

    if (
        frown > 0.20
        and brow_inner_up > 0.15
    ):

        confidence = min(
            (
                frown +
                brow_inner_up
            ) / 2,
            1.0
        )

        return "Sad", confidence


    # =====================================================
    # NEUTRAL
    # =====================================================

    return "Neutral", 0.75


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# DETECT MOOD
# =========================================================

@app.route(
    "/detect_mood",
    methods=["POST"]
)
def detect_mood():

    if "frame" not in request.files:

        return jsonify({
            "error": "No frame received"
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
            "error": "Could not read frame"
        }), 400


    # -----------------------------------------------------
    # BGR → RGB
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # MEDIA PIPE IMAGE
    # -----------------------------------------------------

    mp_image = mp.Image(

        image_format=(
            mp.ImageFormat.SRGB
        ),

        data=rgb_frame
    )


    # -----------------------------------------------------
    # TIMESTAMP
    # -----------------------------------------------------

    timestamp = int(
        time.time() * 1000
    )


    # -----------------------------------------------------
    # FACE LANDMARKER
    # -----------------------------------------------------

    result = (
        landmarker.detect_for_video(
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

            "emoji": "😐",

            "description":
                "Position your face in the camera.",

            "confidence": 0,

            "recommendations": []

        })


    # -----------------------------------------------------
    # BLENDSHAPES
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
    # MOOD INFORMATION
    # -----------------------------------------------------

    info = MOOD_INFO[mood]


    # -----------------------------------------------------
    # MUSIC
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
    # RESPONSE
    # -----------------------------------------------------

    return jsonify({

        "face_detected": True,

        "mood": mood,

        "emoji": info["emoji"],

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