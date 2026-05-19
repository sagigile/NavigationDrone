import cv2
import numpy as np
import pandas as pd
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import os

# =================================================================
# 1. הגדרת המטא-דאטה המכויל שלך (ודא שאלו הערכים המדויקים שלך!)
# =================================================================
MAP_METADATA = {
    'top_left_lat': 32.109231,
    'top_left_lon': 35.201031,
    'bottom_right_lat': 32.099423,
    'bottom_right_lon': 35.218093
}


def pixel_to_gps(pixel_x, pixel_y, img_w, img_h, meta):
    lat_span = meta['top_left_lat'] - meta['bottom_right_lat']
    lon_span = meta['bottom_right_lon'] - meta['top_left_lon']
    lat = meta['top_left_lat'] - (pixel_y / img_h) * lat_span
    lon = meta['top_left_lon'] + (pixel_x / img_w) * lon_span
    return lat, lon


# =================================================================
# 2. אתחול מודלים וטעינת מפת הרפרנס הגדולה
# =================================================================
print("=== שלב 1: טעינת מודלי למידה עמוקה ואופטימיזציית מפה ===")
disk = KF.DISK.from_pretrained("depth").eval()
matcher = KF.LightGlue(features='disk').eval()

map_img = cv2.imread('reference_map.png')
df = pd.read_csv('flight_dataset_index.csv')

if map_img is None or df.empty:
    print("שגיאה: ודא שקובץ המפה reference_map.png וקובץ flight_dataset_index.csv קיימים!")
    exit()

# אופטימיזציה ל-CPU לטובת מהירות הלולאה
TARGET_MAP_WIDTH = 1200
if map_img.shape[1] > TARGET_MAP_WIDTH:
    scale_factor = TARGET_MAP_WIDTH / map_img.shape[1]
    map_img = cv2.resize(map_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
img_h, img_w, _ = map_img.shape

# הכנת מפת הרפרנס כטנסור קבוע בזיכרון
img2_tensor = K.image_to_tensor(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB), keepdim=False).float() / 255.0

# =================================================================
# 3. לולאת הרצה אוטומטית על כל דאטהסט הפריימים
# =================================================================
print(f"\n=== שלב 2: מתחיל הרצת ניווט ויזואלי על {len(df)} פריימים ===")

results = []
W_d, H_d = 960, 540  # רזולוציית פריימים אחידה לריצה מהירה

# חילוץ מאפיינים ממפת הרפרנס פעם אחת בלבד לפני הלולאה (חוסך המון זמן!)
with torch.no_grad():
    features2 = disk(img2_tensor, n=2000, pad_if_not_divisible=True)[0]
    kps2, descs2 = features2.keypoints, features2.descriptors

for idx, row in df.iterrows():
    frame_path = f"sampled_frames/{row['frame_filename']}"
    if not os.path.exists(frame_path):
        continue

    # טעינת פריhorn והכנתו
    drone_img = cv2.imread(frame_path)
    drone_img = cv2.resize(drone_img, (W_d, H_d))
    img1_tensor = K.image_to_tensor(cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB), keepdim=False).float() / 255.0

    status = "Failed"
    est_lat, est_lon, err_meters = np.nan, np.nan, np.nan

    with torch.no_grad():
        try:
            # חילוץ והתאמה
            features1 = disk(img1_tensor, n=2000, pad_if_not_divisible=True)[0]
            kps1, descs1 = features1.keypoints, features1.descriptors

            input_dict = {
                "image0": {"keypoints": kps1[None, ...], "descriptors": descs1[None, ...], "image": img1_tensor},
                "image1": {"keypoints": kps2[None, ...], "descriptors": descs2[None, ...], "image": img2_tensor},
            }

            correspondences = matcher(input_dict)
            matches = correspondences["matches"][0].cpu().numpy()

            if len(matches) >= 4:
                pts_drone = np.array([kps1.cpu().numpy()[m[0]] for m in matches], dtype=np.float32)
                pts_map = np.array([kps2.cpu().numpy()[m[1]] for m in matches], dtype=np.float32)

                H, mask = cv2.findHomography(pts_drone, pts_map, cv2.RANSAC, 10.0)

                if H is not None and np.sum(mask) >= 4:
                    # הקרנת מרכז הפריים
                    center_pixel = np.array([W_d / 2.0, H_d / 2.0], dtype=np.float32).reshape(-1, 1, 2)
                    projected = cv2.perspectiveTransform(center_pixel, H)
                    est_x, est_y = projected[0][0]

                    # המרה ל-GPS וחישוב שגיאה
                    est_lat, est_lon = pixel_to_gps(est_x, est_y, img_w, img_h, MAP_METADATA)

                    lat_mid = np.radians((row['latitude'] + est_lat) / 2.0)
                    d_lat = np.radians(row['latitude'] - est_lat) * 6371000.0
                    d_lon = np.radians(row['longitude'] - est_lon) * 6371000.0 * np.cos(lat_mid)
                    err_meters = np.sqrt(d_lat ** 2 + d_lon ** 2)
                    status = "Success"
        except Exception as e:
            pass

    # הדפסת התקדמות קלה בטרמינל
    if idx % 10 == 0 or idx == len(df) - 1:
        print(
            f"עיבוד: פריים {idx + 1}/{len(df)} | סטטוס: {status} | שגיאה: {f'{err_meters:.1f}מ`' if status == 'Success' else 'N/A'}")

    results.append({
        'frame_filename': row['frame_filename'],
        'true_lat': row['latitude'],
        'true_lon': row['longitude'],
        'est_lat': est_lat,
        'est_lon': est_lon,
        'error_meters': err_meters,
        'status': status
    })

# שמירת כל התוצאות לקובץ אקסל/CSV עבור הדוח הסופי
res_df = pd.DataFrame(results)
res_df.to_csv('localization_results_summary.csv', index=False)

# =================================================================
# 4. חישוב סטטיסטיקות והצגת הגרף האקדמי המסכם
# =================================================================
success_runs = res_df[res_df['status'] == 'Success']
success_rate = (len(success_runs) / len(res_df)) * 100
mean_error = success_runs['error_meters'].mean()

print("\n=============================================")
print("📊 סיכום מדדי ביצוע אקדמיים (Metrics) 📊")
print(f"אחוז הצלחה בנעילת מיקום (Success Rate): {success_rate:.1f}%")
print(f"מרחק שגיאה ממוצע לאורך כל המסלול: {mean_error:.2f} מטרים")
print("קובץ ריכוז נתונים נשמר בהצלחה: localization_results_summary.csv")
print("=============================================")

# שרטוט הגרף הסופי להגשה
map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14, 10))
plt.imshow(map_rgb)

lat_span = MAP_METADATA['top_left_lat'] - MAP_METADATA['bottom_right_lat']
lon_span = MAP_METADATA['bottom_right_lon'] - MAP_METADATA['top_left_lon']

# 1. שרטוט מסלול האמת (Ground Truth) באדום
true_xs = [int(((lon - MAP_METADATA['top_left_lon']) / lon_span) * img_w) for lon in res_df['true_lon']]
true_ys = [int(((MAP_METADATA['top_left_lat'] - lat) / lat_span) * img_h) for lat in res_df['true_lat']]
plt.plot(true_xs, true_ys, color='red', linewidth=3, label='Drone Path (GPS Ground Truth)', zorder=2)

# 2. שרטוט נקודות השערוך של ה-AI במגנטה (רק אלו שהצליחו)
est_xs = [int(((lon - MAP_METADATA['top_left_lon']) / lon_span) * img_w) for lon in success_runs['est_lon']]
est_ys = [int(((MAP_METADATA['top_left_lat'] - lat) / lat_span) * img_h) for lat in success_runs['est_lat']]
plt.scatter(est_xs, est_ys, color='magenta', marker='x', s=40, label='AI Estimated View Center', zorder=3)

plt.xlim(0, img_w)
plt.ylim(img_h, 0)
plt.title(
    f"Visual Navigation Performance Summary\nSuccess Rate: {success_rate:.1f}% | Mean Vector Offset: {mean_error:.1f}m",
    fontsize=12)
plt.legend(loc='upper right')
plt.axis('on')

plt.savefig('final_navigation_benchmark.png', bbox_inches='tight', dpi=300)
plt.show()