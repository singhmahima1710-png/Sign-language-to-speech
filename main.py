"""
main.py - menu launcher for the whole project.

    python main.py
"""

import subprocess
import sys

MENU = """
==============================================
  Sign Language -> Speech  (MediaPipe + OpenCV)
==============================================
  1. Classical CV demo (skin mask, contour, hull)
  2. Collect training data (webcam)
  3. Build dataset from image folder (e.g. Kaggle ASL)
  4. Train classifier
  5. Run real-time sign -> speech
  6. Feature ablation experiment (for report)
  0. Exit
"""


def run(script, *args):
    subprocess.call([sys.executable, script, *args])


def main():
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()
        if choice == "1":
            run("classical_demo.py")
        elif choice == "2":
            run("data_collection.py")
        elif choice == "3":
            root = input("Path to dataset folder: ").strip().strip('"')
            n = input("Max images per class [300]: ").strip() or "300"
            run("build_dataset_from_images.py", root, "--max-per-class", n)
        elif choice == "4":
            run("train_model.py")
        elif choice == "5":
            run("realtime_recognition.py")
        elif choice == "6":
            run("evaluate_ablation.py")
        elif choice == "0":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
