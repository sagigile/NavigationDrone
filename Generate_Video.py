import cv2
import numpy as np
import pandas as pd
import os

# =================================================================
# 1. הגדרת המטא-דאטה והפונקציות הגיאומטריות
# =================================================================
MAP_METADATA = {
    'top_left_lat': 32.109231,
    'top_left_lon': 35.201031,
    'bottom_right_lat': 32.099423,
    'bottom_right_lon': 35.218093
}


def gps_to_pixel(lat, lon, img_w, img_h, meta):
    """ממיר קואורדינטות GPS בחזרה לפיקסלים על המפה"""
    lat_span = meta['top_left_lat'] - meta['bottom_right_lat']
    lon_span = meta['bottom_right_lon'] - meta['top_left_lon']
    x = ((lon - meta['top_left_lon']) / lon_span) * img_w
    y = ((meta['top_left_lat'] - lat) / lat_span) * img_h
    return int(x), int(y)


# =================================================================
# 2. טעינת הנתונים והמפה
# =================================================================
print("טוען את נתוני הניווט ואת מפת הרפרנס...")
map_img = cv2.imread('reference_map.png')
try:
    df = pd.read_csv('localization_results_summary.csv')
except FileNotFoundError:
    print("שגיאה: קובץ התוצאות localization_results_summary.csv לא נמצא. ודא שהרצת את הקוד הקודם במלואו.")
    exit()

if map_img is None:
    print("שגיאה: לא נמצא קובץ reference_map.png!")
    exit()

# התאמת גודל המפה לפורמט של וידאו HD
TARGET_WIDTH = 1280
scale_factor = TARGET_WIDTH / map_img.shape[1]
map_img = cv2.resize(map_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
img_h, img_w, _ = map_img.shape

# =================================================================
# 3. הגדרת כותב הוידאו (OpenCV VideoWriter)
# =================================================================
output_filename = 'flight_navigation_dynamic.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # קודק סטנדרטי ל-MP4
fps = 10.0  # 10 פריימים בשנייה כדי שיהיה נוח לראות את ההתקדמות
out = cv2.VideoWriter(output_filename, fourcc, fps, (img_w, img_h))

print(f"מתחיל לרנדר את הסרטון: {output_filename} (סה\"כ {len(df)} פריימים)")

# =================================================================
# 4. לולאת רינדור פריימים לסרטון
# =================================================================
path_history = []

for idx, row in df.iterrows():
    # יצירת עותק נקי של המפה לפריים הנוכחי
    frame = map_img.copy()

    # חילוץ מיקום ה-GPS האמיתי והוספה להיסטוריית המסלול
    true_lat, true_lon = row['true_lat'], row['true_lon']
    true_px = gps_to_pixel(true_lat, true_lon, img_w, img_h, MAP_METADATA)
    path_history.append(true_px)

    # שרטוט המסלול האמיתי שעברנו עד כה (קו אדום)
    if len(path_history) > 1:
        for i in range(1, len(path_history)):
            cv2.line(frame, path_history[i - 1], path_history[i], (0, 0, 255), 2)

    # שרטוט מיקום הרחפן הנוכחי (עיגול ירוק מלא)
    cv2.circle(frame, true_px, 8, (0, 255, 0), -1)
    cv2.circle(frame, true_px, 10, (0, 0, 0), 2)  # גבול שחור לבולטות

    # אם ה-AI הצליח לשערך מיקום בפריים הזה, נצייר את החץ ואת האיקס
    if row['status'] == 'Success':
        est_lat, est_lon = row['est_lat'], row['est_lon']
        est_px = gps_to_pixel(est_lat, est_lon, img_w, img_h, MAP_METADATA)

        # שרטוט קו המבט (הוקטור) מהרחפן לקרקע
        cv2.line(frame, true_px, est_px, (0, 255, 255), 3)

        # שרטוט איקס מגנטה בנקודת המבט
        cv2.drawMarker(frame, est_px, (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)

        err_text = f"Error: {row['error_meters']:.1f}m"
        color_text = (0, 255, 0) if row['error_meters'] < 200 else (0, 165, 255)
    else:
        err_text = "Error: N/A (AI Failed)"
        color_text = (0, 0, 255)

    # הוספת שכבת טקסט מקצועית (Telemetry) בפינת המסך (אנגלית למניעת שיבושי פונט)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # שקיפות

    cv2.putText(frame, "Deep Visual Navigation Demo", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {idx + 1}/{len(df)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, err_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)

    # כתיבת הפריים לוידאו
    out.write(frame)

# שחרור משאבים
out.release()
print("הרינדור הסתיים בהצלחה! קובץ הוידאו flight_navigation_dynamic.mp4 ממתין לך בתיקיית הפרויקט.")