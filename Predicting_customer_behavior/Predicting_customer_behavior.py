# E:\programming\Project\Portfolio\Predicting_customer_behavior\Predicting_customer_behavior.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# CSVファイルからデータを読み込む
customer_info = pd.read_csv("customer_info.csv")
customer_history = pd.read_csv("customer_history.csv")
customer_exit = pd.read_csv("customer_exit.csv")

# データのクリーニング: 今回は欠損値の確認と補完を行います。
# 顧客の基本情報の欠損値の確認
missing_values_customer_info = customer_info.isnull().sum()

# 顧客の利用履歴の欠損値の確認
missing_values_customer_history = customer_history.isnull().sum()

# 退会者の情報の欠損値の確認
missing_values_customer_exit = customer_exit.isnull().sum()

# 今回はシンプルに欠損値を中央値や最頻値で補完します（実際のデータ分析では、補完の方法はデータの性質や目的に応じて選びます）
# 顧客の基本情報の年齢の欠損値を中央値で補完
customer_info['年齢'].fillna(customer_info['年齢'].median(), inplace=True)

# 退会者の情報の退会理由の欠損値を最頻値で補完
mode_reason = customer_exit['退会理由'].mode()[0]
customer_exit['退会理由'].fillna(mode_reason, inplace=True)

missing_values_after_cleaning = {
    "customer_info": customer_info.isnull().sum(),
    "customer_history": customer_history.isnull().sum(),
    "customer_exit": customer_exit.isnull().sum()
}

missing_values_after_cleaning

# 基本的な統計分析
# 顧客の基本情報の基本統計量
stats_customer_info = customer_info.describe(include='all')

# 顧客の利用履歴の基本統計量
stats_customer_history = customer_history.describe(include='all')

# 退会者の情報の基本統計量
stats_customer_exit = customer_exit.describe(include='all')

print("基本統計量 - 顧客の基本情報:\n", stats_customer_info)
print("\n基本統計量 - 顧客の利用履歴:\n", stats_customer_history)
print("\n基本統計量 - 退会者の情報:\n", stats_customer_exit)

# 傾向の分析
# 年齢別の顧客数
age_distribution = customer_info['年齢'].value_counts().sort_index()

# 退会理由別の顧客数
exit_reason_distribution = customer_exit['退会理由'].value_counts()

# トレーナー利用の有無別の利用回数
trainer_usage_distribution = customer_history['トレーナー利用'].value_counts()

print("\n年齢別の顧客数:\n", age_distribution)
print("\n退会理由別の顧客数:\n", exit_reason_distribution)
print("\nトレーナー利用の有無別の利用回数:\n", trainer_usage_distribution)

# 予測モデルの作成
# 顧客の利用回数を基に、退会するかどうかを予測

# 利用回数を計算
usage_count = customer_history.groupby('顧客ID').size()

# 退会情報をマージ
data = customer_info.merge(usage_count.rename('利用回数'), left_on='顧客ID', right_index=True, how='left')
data['退会'] = data['退会日'].notna().astype(int)

# データをトレーニングセットとテストセットに分割
X = data[['利用回数']]
y = data['退会']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# NaN を含む行を削除 (トレーニングデータ)
X_train = X_train.dropna()
y_train = y_train[X_train.index]

# NaN を含む行を削除 (テストデータ)
X_test = X_test.dropna()
y_test = y_test[X_test.index]

# ロジスティック回帰モデルをトレーニング
model = LogisticRegression()
model.fit(X_train, y_train)

# テストセットでの予測
y_pred = model.predict(X_test)

# 分析結果の評価
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 結果の可視化
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted')
plt.xlabel('利用回数')
plt.ylabel('退会')
plt.legend()
plt.title('実際の退会 vs 予測の退会')
plt.show()