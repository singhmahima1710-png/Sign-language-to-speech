
"""
realtime_recognition.py
Sign Language -> Text -> Speech

Controls:
    SPACE : commit current word to the sentence
    b     : delete last character, or last completed word
    c     : clear everything
    s     : speak the COMPLETE sentence and current word
    q     : quit
"""

import os
import time

import cv2
import joblib
import pyttsx3

from utils import (
    get_hands_detector,
    extract_combined_features,
    draw_landmarks,
)

MODEL_PATH = os.path.join("model", "sign_classifier.joblib")

STABLE_FRAMES = 10
CONFIDENCE_THRESHOLD = 0.60


def speak_text(engine, sentence, current_word=""):
    """Speak all completed words plus the current unfinished word."""
    parts = []

    if sentence and sentence.strip():
        parts.append(sentence.strip())

    if current_word and current_word.strip():
        parts.append(current_word.strip())

    full_text = " ".join(parts).strip()

    if not full_text:
        print("Nothing to speak. Spell a word first.")
        return

    # Diagnostic: shows exactly what will be spoken.
    print(f"TEXT SENT TO SPEECH: [{full_text}]")

    try:
        engine.stop()
        engine.say(full_text)
        engine.runAndWait()
        print("Speech completed.")
    except Exception as exc:
        print(f"Speech error: {exc}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Train the classifier using option 4 in main.py.")
        return

    try:
        clf = joblib.load(MODEL_PATH)
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 155)
    except Exception as exc:
        print(f"ERROR loading model or speech engine: {exc}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        tts_engine.stop()
        return

    detector = None

    try:
        detector = get_hands_detector(
            static_image_mode=False,
            max_num_hands=1,
        )

        sentence = ""
        current_word = ""

        last_pred = None
        stable_count = 0
        waiting_for_change = False
        accepted_label = None

        print("\nSign -> Speech started.")
        print("SPACE=commit | b=backspace | c=clear | s=speak | q=quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            pred_label = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                draw_landmarks(frame, hand_landmarks)

                features = extract_combined_features(
                    frame, hand_landmarks
                ).reshape(1, -1)

                probabilities = clf.predict_proba(features)[0]
                best_index = int(probabilities.argmax())

                pred_label = str(clf.classes_[best_index])
                confidence = float(probabilities[best_index])

            valid = (
                pred_label is not None
                and confidence >= CONFIDENCE_THRESHOLD
            )

            if not valid:
                last_pred = None
                stable_count = 0
                waiting_for_change = False
                accepted_label = None

            elif waiting_for_change:
                # Wait for a different stable prediction before
                # accepting another letter.
                if pred_label != accepted_label:
                    if pred_label == last_pred:
                        stable_count += 1
                    else:
                        last_pred = pred_label
                        stable_count = 1

                    if stable_count >= STABLE_FRAMES:
                        waiting_for_change = False
                        stable_count = 0
                        last_pred = pred_label
                else:
                    last_pred = pred_label
                    stable_count = 0

            else:
                if pred_label == last_pred:
                    stable_count += 1
                else:
                    last_pred = pred_label
                    stable_count = 1

                if stable_count >= STABLE_FRAMES:
                    current_word += pred_label
                    accepted_label = pred_label
                    waiting_for_change = True
                    stable_count = 0

                    print(
                        f"Accepted: {pred_label} | "
                        f"Word: {current_word}"
                    )

            display_pred = (
                f"{pred_label} ({confidence:.2f})"
                if pred_label is not None else "-"
            )

            status = (
                "Change sign for next letter"
                if waiting_for_change else "Hold sign steady"
            )

            cv2.putText(
                frame, f"Prediction: {display_pred}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Word: {current_word}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cv2.putText(
                frame, f"Sentence: {sentence}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            cv2.putText(
                frame, status, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
            )

            cv2.imshow("Sign -> Speech - press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                # Append the word exactly once, with one space.
                word = current_word.strip()

                if word:
                    sentence = (
                        f"{sentence.strip()} {word}".strip()
                    )
                    current_word = ""
                    print(f"Sentence buffer: [{sentence}]")
                else:
                    print("No current word to commit.")

            elif key == ord("b"):
                if current_word:
                    current_word = current_word[:-1]
                    print(f"Word: {current_word}")
                elif sentence.strip():
                    words = sentence.strip().split()
                    words.pop()
                    sentence = " ".join(words)
                    print(f"Sentence: {sentence}")

            elif key == ord("c"):
                sentence = ""
                current_word = ""
                print("All text cleared.")

            elif key == ord("s"):
                # Speak the entire sentence and current word together.
                speak_text(tts_engine, sentence, current_word)

    except Exception as exc:
        print(f"Recognition error: {exc}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass

        try:
            tts_engine.stop()
        except Exception:
            pass

        print("Recognition stopped.")


if __name__ == "__main__":
    main()
