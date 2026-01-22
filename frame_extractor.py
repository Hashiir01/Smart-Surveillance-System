import cv2
import os

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mpg']  # Added .mpg
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def extract_frames(video_path, output_folder, label, frame_rate=10):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_path) if is_video_file(f)]

    total_frames_extracted = 0
    print(f"\n📂 Processing '{label}' videos from: {video_path}")
    print(f"🔍 Found {len(video_files)} video file(s).")

    for idx, video in enumerate(video_files):
        video_file_path = os.path.join(video_path, video)

        if os.path.getsize(video_file_path) < 1000:
            print(f"[!] Skipped: Empty or corrupt file → {video_file_path}")
            continue

        cap = cv2.VideoCapture(video_file_path)

        if not cap.isOpened():
            print(f"[!] Skipped: Cannot open video → {video_file_path}")
            continue

        count = 0
        frame_index = 0
        success, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0 or fps != fps:  # handles NaN
            print(f"[!] Skipped: Invalid FPS → {video_file_path}")
            cap.release()
            continue

        interval = int(fps / frame_rate)
        if interval == 0:
            interval = 1  # Ensure at least 1 frame between saves

        while success:
            if frame_index % interval == 0:
                save_path = os.path.join(output_folder, f"{label}_{idx}_{count}.jpg")
                cv2.imwrite(save_path, frame)
                count += 1
            success, frame = cap.read()
            frame_index += 1

        cap.release()
        print(f"[{idx + 1}/{len(video_files)}] ✅ {video} → {count} frames extracted.")
        total_frames_extracted += count

    print(f"🎉 Done! Total frames extracted for '{label}': {total_frames_extracted}")

if __name__ == "__main__":
    extract_frames(
        "C:\\Users\\User\\Desktop\\sss\\Dataset\\Violence",
        "C:\\Users\\User\\Desktop\\sss\\Dataset\\frames\\violence",
        "violence"
    )
    extract_frames(
        "C:\\Users\\User\\Desktop\\sss\\Dataset\\NonViolence",
        "C:\\Users\\User\\Desktop\\sss\\Dataset\\frames\\non_violence",
        "non_violence"
    )
