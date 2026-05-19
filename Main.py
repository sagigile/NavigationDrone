import cv2
import re
import os
import pandas as pd


def parse_srt_telemetry(srt_path):
    """פונקציה שמנתחת את קובץ ה-SRT ומחזירה רשימה של נתוני טלמטריה לפי פריימים"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # פיצול לפי בלוקים של כתוביות
    blocks = content.strip().split('\n\n')
    telemetry_data = []

    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 4:
            continue

        # חילוץ מספר הפריים וזמנים
        frame_idx = int(lines[0])
        time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3})', lines[1])

        # חילוץ נתוני ה-GPS והגובה באמצעות Regular Expressions
        data_line = lines[-1]  # השורה שמכילה את המטא-דאטה בסוגריים מרובעים
        lat_match = re.search(r'latitude:\s*([\d\.]+)', data_line)
        lon_match = re.search(r'longitude:\s*([\d\.]+)', data_line)
        alt_match = re.search(r'rel_alt:\s*([\d\.]+)', data_line)

        if lat_match and lon_match and alt_match:
            telemetry_data.append({
                'frame_idx': frame_idx,
                'timestamp': time_match.group(1) if time_match else '',
                'latitude': float(lat_match.group(1)),
                'longitude': float(lon_match.group(1)),
                'rel_alt': float(alt_match.group(1))
            })

    return pd.DataFrame(telemetry_data)


def extract_sampled_frames(video_path, telemetry_df, output_dir, sample_rate_seconds=1):
    """דגימת פריימים מהווידאו בהתאם לזמנים בקובץ ה-SRT"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # חישוב קפיצה בפריימים (למשל: פריים אחד בכל שנייה)
    frame_step = int(fps * sample_rate_seconds)

    sampled_records = []

    for i in range(0, len(telemetry_df), frame_step):
        row = telemetry_df.iloc[i]
        frame_to_read = int(row['frame_idx'])

        # מעבר לפריים הספציפי בווידאו
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
        ret, frame = cap.read()

        if ret:
            frame_name = f"frame_{frame_to_read:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)

            # שמירת הרשומה המסונכרנת
            record = row.to_dict()
            record['frame_filename'] = frame_name
            sampled_records.append(record)

    cap.release()
    return pd.DataFrame(sampled_records)


# --- הרצה ראשונית ---
srt_file = "DJI_20260427152735_0019_D.SRT"
video_file = "DJI_20260427152735_0019_D.MP4"
output_images_folder = "./sampled_frames"

print("מנתח קובץ SRT...")
full_telemetry = parse_srt_telemetry(srt_file)

print(f"נמצאו {len(full_telemetry)} פריימים מתועדים ב-SRT. מתחיל בדגימת וידאו...")
dataset_summary = extract_sampled_frames(video_file, full_telemetry, output_images_folder, sample_rate_seconds=1)

# שמירת קובץ ה-CSV שישמש כאינדקס לפרויקט
dataset_summary.to_csv("flight_dataset_index.csv", index=False)
print(f"הסתיים! {len(dataset_summary)} פריימים נשמרו בתיקייה '{output_images_folder}'")
print("נוצר קובץ אינדקס מסונכרן: flight_dataset_index.csv")

# הגדרת המטא-דאטה הגיאוגרפי של המפה שגזרת
MAP_METADATA = {
    'map_path': 'reference_map.png',
    'top_left_lat': 32.111088,
    'top_left_lon': 35.198945,
    'bottom_right_lat': 32.103812,
    'bottom_right_lon': 35.210203
}


def pixel_to_gps(pixel_x, pixel_y, img_width, img_height, metadata):
    """פונקציה הממירה קואורדינטת פיקסל במפת הרפרנס לקואורדינטת GPS עולמית"""
    # חישוב יחס ליניארי בין הפיקסלים לטווח הגיאוגרפי
    lat_span = metadata['top_left_lat'] - metadata['bottom_right_lat']
    lon_span = metadata['bottom_right_lon'] - metadata['top_left_lon']

    # פיקסל (0,0) הוא הפינה השמאלית העליונה
    lat = metadata['top_left_lat'] - (pixel_y / img_height) * lat_span
    lon = metadata['top_left_lon'] + (pixel_x / img_width) * lon_span

    return lat, lon