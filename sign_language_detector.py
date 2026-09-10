import cv2
import mediapipe as mp
import math
import pyttsx3

# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

# =========================================================
# OFFLINE VOICE ENGINE
# =========================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

last_spoken_gesture = "UNKNOWN"

# =========================================================
# HELPERS
# =========================================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def finger_open(hand, tip, pip, wrist=0):
    return (
        distance(hand[tip], hand[wrist])
        >
        distance(hand[pip], hand[wrist])
    )


def finger_closed(hand, tip, pip, wrist=0):
    return (
        distance(hand[tip], hand[wrist])
        <
        distance(hand[pip], hand[wrist])
    )


def open_palm(hand):
    return (
        finger_open(hand, 8, 6)
        and finger_open(hand, 12, 10)
        and finger_open(hand, 16, 14)
        and finger_open(hand, 20, 18)
    )

# =========================================================
# ONE HAND SIGNS
# =========================================================

def mini_heart(hand):

    middle_closed = finger_closed(hand, 12, 10)
    ring_closed = finger_closed(hand, 16, 14)
    pinky_closed = finger_closed(hand, 20, 18)

    thumb_index = distance(
        hand[4],
        hand[8]
    )

    return (
        middle_closed
        and ring_closed
        and pinky_closed
        and thumb_index < 0.10
    )


def peace(hand):

    return (
        finger_open(hand, 8, 6)
        and finger_open(hand, 12, 10)
        and finger_closed(hand, 16, 14)
        and finger_closed(hand, 20, 18)
    )


def ok_sign(hand):

    thumb_index = distance(
        hand[4],
        hand[8]
    )

    return (
        thumb_index < 0.10
        and finger_open(hand, 12, 10)
        and finger_open(hand, 16, 14)
        and finger_open(hand, 20, 18)
    )


def telephone(hand):

    return (
        finger_open(hand, 4, 3)
        and finger_closed(hand, 8, 6)
        and finger_closed(hand, 12, 10)
        and finger_closed(hand, 16, 14)
        and finger_open(hand, 20, 18)
    )


def i_love_you(hand):

    index = finger_open(hand, 8, 6)
    middle = finger_closed(hand, 12, 10)
    ring = finger_closed(hand, 16, 14)
    pinky = finger_open(hand, 20, 18)

    return (
        index
        and middle
        and ring
        and pinky
    )

# =========================================================
# WATER
# INDEX + MIDDLE + RING OPEN
# PINKY + THUMB CLOSED
# =========================================================

def water_sign(hand):

    index_open = finger_open(
        hand, 8, 6
    )

    middle_open = finger_open(
        hand, 12, 10
    )

    ring_open = finger_open(
        hand, 16, 14
    )

    pinky_closed = finger_closed(
        hand, 20, 18
    )

    thumb_closed = (
        distance(hand[4], hand[0])
        <
        distance(hand[3], hand[0]) * 1.20
    )

    return (
        index_open
        and middle_open
        and ring_open
        and pinky_closed
        and thumb_closed
    )

# =========================================================
# YES
# =========================================================

def yes_sign(hand):

    return (
        finger_closed(hand, 8, 6)
        and finger_closed(hand, 12, 10)
        and finger_closed(hand, 16, 14)
        and finger_closed(hand, 20, 18)
    )

# =========================================================
# NO
# =========================================================

def no_sign(hand):

    wrist = hand[0]

    thumb_open = (
        distance(hand[4], wrist)
        >
        distance(hand[3], wrist)
    )

    index_open = (
        distance(hand[8], wrist)
        >
        distance(hand[6], wrist)
    )

    middle_closed = (
        distance(hand[12], wrist)
        <
        distance(hand[10], wrist)
    )

    ring_closed = (
        distance(hand[16], wrist)
        <
        distance(hand[14], wrist)
    )

    pinky_closed = (
        distance(hand[20], wrist)
        <
        distance(hand[18], wrist)
    )

    return (
        thumb_open
        and index_open
        and middle_closed
        and ring_closed
        and pinky_closed
    )

# =========================================================
# TWO HAND SIGNS
# =========================================================

def thanks(h1, h2):

    if not (open_palm(h1) and open_palm(h2)):
        return False

    middle_up1 = h1[12].y < h1[0].y
    middle_up2 = h2[12].y < h2[0].y

    palm_distance = distance(
        h1[0],
        h2[0]
    )

    middle_distance = distance(
        h1[12],
        h2[12]
    )

    return (
        middle_up1
        and middle_up2
        and palm_distance < 0.30
        and middle_distance < 0.25
    )


def heart(h1, h2):

    thumb_gap = distance(
        h1[4],
        h2[4]
    )

    index_gap = distance(
        h1[8],
        h2[8]
    )

    wrist_gap = distance(
        h1[0],
        h2[0]
    )

    return (
        thumb_gap < 0.20
        and index_gap < 0.20
        and wrist_gap < 0.60
    )


def hug(h1, h2):

    if not (open_palm(h1) and open_palm(h2)):
        return False

    x1 = h1[0].x
    x2 = h2[0].x

    y1 = h1[0].y
    y2 = h2[0].y

    horizontal_distance = abs(x1 - x2)
    vertical_distance = abs(y1 - y2)

    return (
        horizontal_distance > 0.15
        and horizontal_distance < 0.75
        and vertical_distance < 0.45
    )


def help_sign(hands):

    if len(hands) != 2:
        return False

    h1 = hands[0]
    h2 = hands[1]

    open1 = open_palm(h1)
    open2 = open_palm(h2)

    fist1 = yes_sign(h1)
    fist2 = yes_sign(h2)

    return (
        (open1 and fist2)
        or
        (open2 and fist1)
    )

# =========================================================
# MAIN
# =========================================================

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(image)

        gesture = "UNKNOWN"

        hands = result.hand_landmarks

        # =================================================
        # TWO HAND SIGNS
        # =================================================

        if len(hands) == 2:

            h1 = hands[0]
            h2 = hands[1]

            if thanks(h1, h2):
                gesture = "THANKS"

            elif heart(h1, h2):
                gesture = "HEART"

            elif hug(h1, h2):
                gesture = "HUG"

            elif help_sign(hands):
                gesture = "HELP"

        # =================================================
        # ONE HAND SIGNS
        # =================================================

        elif len(hands) == 1:

            hand = hands[0]

            if mini_heart(hand):
                gesture = "SARANGHAE"

            elif peace(hand):
                gesture = "PEACE"

            elif ok_sign(hand):
                gesture = "OK"

            elif telephone(hand):
                gesture = "TELEPHONE"

            elif i_love_you(hand):
                gesture = "I LOVE YOU"

            elif water_sign(hand):
                gesture = "WATER"

            elif no_sign(hand):
                gesture = "NO"

            elif yes_sign(hand):
                gesture = "YES"

        # =================================================
        # VOICE OUTPUT
        # =================================================

        if (
            gesture != "UNKNOWN"
            and gesture != last_spoken_gesture
        ):

            engine.say(gesture)
            engine.runAndWait()

            last_spoken_gesture = gesture

        elif gesture == "UNKNOWN":

            last_spoken_gesture = "UNKNOWN"

        # =================================================
        # DRAW DOTS
        # =================================================

        h, w, _ = frame.shape

        for hand in hands:

            for point in hand:

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

        # =================================================
        # SHOW GESTURE
        # =================================================

        cv2.putText(
            frame,
            gesture,
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            4
        )

        cv2.imshow(
            "Sign Language Translator",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()