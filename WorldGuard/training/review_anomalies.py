"""Human review CLI for labelling flagged anomaly clips.

Loops over unlabelled embedding files and prompts the user to label each one.
Labels are appended to a JSON file for use in Stage 2 classifier training.

Usage:
    python training/review_anomalies.py \
        --embeddings-dir outputs/embeddings \
        --labels-file data/feedback/labels.json \
        --camera-id ucsd

Controls:
    t  → true anomaly  (label=1)
    f  → false positive (label=0)
    s  → skip (label not saved)
    q  → quit and save progress
"""

import argparse
import json
import os


def load_existing_labels(labels_file: str) -> set[str]:
    """Return the set of clip filenames already labelled."""
    labelled = set()
    if not os.path.isfile(labels_file):
        return labelled
    with open(labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                labelled.add(json.loads(line)["clip"])
    return labelled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review and label anomaly clips for Stage 2 training"
    )
    parser.add_argument("--embeddings-dir", required=True,
                        help="Directory containing *_emb.pt files")
    parser.add_argument("--labels-file", required=True,
                        help="Path to append labels JSON (created if absent)")
    parser.add_argument("--camera-id", required=True,
                        help="Camera ID to filter embeddings (e.g. ucsd)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.labels_file)), exist_ok=True)

    # Find all embedding files for this camera
    all_files = sorted(
        f for f in os.listdir(args.embeddings_dir)
        if f.startswith(args.camera_id) and f.endswith("_emb.pt")
    )

    if not all_files:
        print(f"No embedding files found for camera '{args.camera_id}' in {args.embeddings_dir}")
        print("Run inference/score_video.py first to generate embeddings.")
        return

    labelled = load_existing_labels(args.labels_file)
    pending = [f for f in all_files if f not in labelled]

    print(f"\nCamera: {args.camera_id}")
    print(f"Total embeddings : {len(all_files)}")
    print(f"Already labelled : {len(labelled)}")
    print(f"Pending review   : {len(pending)}")
    print("\nControls: [t] true anomaly  [f] false positive  [s] skip  [q] quit\n")

    if not pending:
        print("All clips already labelled.")
        return

    saved = 0
    with open(args.labels_file, "a") as out:
        for i, clip_file in enumerate(pending):
            print(f"[{i+1}/{len(pending)}] {clip_file}")

            while True:
                key = input("  Label: ").strip().lower()
                if key in ("t", "f", "s", "q"):
                    break
                print("  Invalid — enter t / f / s / q")

            if key == "q":
                print(f"\nQuitting. {saved} labels saved this session.")
                return

            if key == "s":
                print("  Skipped.\n")
                continue

            label = 1 if key == "t" else 0
            label_str = "true_anomaly" if label == 1 else "false_positive"
            record = {"clip": clip_file, "label": label, "camera_id": args.camera_id}
            out.write(json.dumps(record) + "\n")
            out.flush()
            saved += 1
            print(f"  Saved as {label_str}.\n")

    print(f"\nDone. {saved} labels saved to {args.labels_file}")
    print("Run training/train_feedback.py to train the Stage 2 classifier.")


if __name__ == "__main__":
    main()
