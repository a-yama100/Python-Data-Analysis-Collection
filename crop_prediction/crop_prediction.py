# E:\programming\Project\Portfolio\crop_prediction\crop_prediction.py

import ee
import matplotlib.pyplot as plt

# Earth Engineの初期化
ee.Initialize()

# 対象となる地域（フレズノ, カリフォルニア州）の矩形範囲を定義
fresno_geometry = ee.Geometry.Rectangle([-120.0, 36.5, -119.5, 37.0])

# USDAのCropland Data Layerを取得
dataset = ee.ImageCollection('USDA/NASS/CDL') \
    .filter(ee.Filter.date('2020-01-01', '2020-12-31')) \
    .filterBounds(fresno_geometry) \
    .first()

# 'cropland'バンドを選択
cropLandcover = dataset.select('cropland')

# 画像のURLを取得（これをブラウザで開くことで画像を確認できる）
url = cropLandcover.getDownloadURL({
    'scale': 30,
    'crs': 'EPSG:4326',
    'fileFormat': 'GeoTIFF',
    'region': fresno_geometry
})

print(url)

# 画像の前処理スクリプトの作成
import numpy as np
import rasterio
from rasterio.mask import mask
from skimage import exposure

# 画像の読み込み
with rasterio.open("2020.cropland.tif") as src:
    # 全てのバンドを読み込む
    image = src.read()

    # 画像のデータ型をfloatに変更
    image = image.astype(float)

    # クラウドカバー（雲）の除去（仮実装。具体的な手法に基づいて実装する必要があります）
    image[image == 0] = np.nan  # 仮に0をクラウドとみなし、NaNで埋める

    # 画像の正規化
    p2, p98 = np.percentile(image[~np.isnan(image)], (2, 98))  # 2-98%の強度を取得
    image_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

    # 画像のクロッピング
    # fresno_geometryを基にクロッピング（矩形の範囲をGeoJSON形式に変換する必要があります）
    fresno_geojson = {
        "type": "Polygon",
        "coordinates": [[[-120.0, 36.5], [-119.5, 36.5], [-119.5, 37.0], [-120.0, 37.0], [-120.0, 36.5]]]
    }
    out_image, out_transform = mask(src, [fresno_geojson], crop=True)

# 新しい画像として保存
with rasterio.open('2020.cropland_processed.tif', 'w', **src.meta) as dest:
    dest.write(out_image)

print("Image processing completed.")

# 画像の平均色や中央値色を取得するためのスクリプトを作成
import numpy as np
import rasterio

# 画像の読み込み
with rasterio.open('2020.cropland_processed.tif') as src:
    image = src.read()
    image = image[~np.isnan(image)]  # NaNの除去

    # ピクセルの平均と中央値の計算
    mean_color = np.mean(image)
    median_color = np.median(image)

print(f"Average Color: {mean_color}")
print(f"Median Color: {median_color}")

# 収穫時期のラベル付けされたデータセットの作成
import pandas as pd
import random
from datetime import date, timedelta

# 日付範囲の設定
start_date = date(2020, 1, 1)
end_date = date(2020, 12, 31)
delta = timedelta(days=1)

# 空のリストを作成
dates = []
colors = []
labels = []

# データセットの生成
while start_date <= end_date:
    dates.append(start_date)

    # 月に3-4回の収穫が最適な日をランダムに選択
    if random.randint(1, 10) <= 3:  
        colors.append(random.uniform(0.7, 1.0))  # 収穫が最適な場合は色の平均値が高く
        labels.append(1)
    else:
        colors.append(random.uniform(0.3, 0.7))  # 収穫が最適でない場合は色の平均値が低く
        labels.append(0)

    start_date += delta

# データフレームの作成
df = pd.DataFrame({
    'date': dates,
    'average_color': colors,
    'optimal_harvest': labels
})

df.head()
# モデルのトレーニング、評価、未来の収穫時期の予測
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# データセットの生成 (既存のコードから)
start_date = date(2020, 1, 1)
end_date = date(2020, 12, 31)
delta = timedelta(days=1)
dates = []
colors = []
labels = []

while start_date <= end_date:
    dates.append(start_date)
    if np.random.randint(1, 10) <= 3:
        colors.append(np.random.uniform(0.7, 1.0))
        labels.append(1)
    else:
        colors.append(np.random.uniform(0.3, 0.7))
        labels.append(0)
    start_date += delta

df = pd.DataFrame({
    'date': dates,
    'average_color': colors,
    'optimal_harvest': labels
})

# 1. データの前処理
# 特徴量とターゲット変数の分割
X = df[['average_color']]
y = df['optimal_harvest']

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. モデルのトレーニング
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 3. モデルの評価
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 予測結果のグラフを表示 & 保存
plt.figure(figsize=(10, 6))
plt.scatter(X_test['average_color'], y_test, color='blue', label='Actual', alpha=0.6)
plt.scatter(X_test['average_color'], y_pred, color='red', label='Predicted', marker='x')
plt.title("Predictions vs Actual Harvesting")
plt.xlabel("Average Color")
plt.ylabel("Optimal Harvesting (1 = Optimal, 0 = Not Optimal)")
plt.legend()
plt.tight_layout()
plt.savefig('predictions_vs_actual.png')  # グラフを保存
plt.show()

# 4. 未来の収穫時期の予測
# 今回は、2021年のデータを生成して予測します
future_dates = []
future_colors = []

start_date = date(2021, 1, 1)
end_date = date(2021, 12, 31)
while start_date <= end_date:
    future_dates.append(start_date)
    future_colors.append(np.random.uniform(0.3, 1.0))  # ランダムな色の平均値
    start_date += delta

future_df = pd.DataFrame({
    'date': future_dates,
    'average_color': future_colors
})

future_predictions = clf.predict(future_df[['average_color']])
future_df['optimal_harvest_prediction'] = future_predictions

print(future_df.head())

# 未来の収穫時期の予測結果のグラフを表示 & 保存
plt.figure(figsize=(10, 6))
plt.plot(future_df['date'], future_df['optimal_harvest_prediction'], label='Predicted Harvesting', color='green')
plt.title("Predicted Harvesting for 2021")
plt.xlabel("Date")
plt.ylabel("Optimal Harvesting (1 = Optimal, 0 = Not Optimal)")
plt.tight_layout()
plt.savefig('future_harvesting_2021.png')  # グラフを保存
plt.show()