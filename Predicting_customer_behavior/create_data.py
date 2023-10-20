# E:\programming\Project\Portfolio\Predicting_customer_behavior\create_data.py

import pandas as pd
import random
import datetime

import pandas as pd

# Re-run the modified data generation process

# 顧客の基本情報の生成
num_customers = 1000
names = ["Customer" + str(i) for i in range(1, num_customers + 1)]
ages = [random.randint(18, 65) for _ in range(num_customers)]
genders = [random.choice(["男", "女"]) for _ in range(num_customers)]
registration_dates = [(datetime.date(2022, 1, 1) + datetime.timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d') for _ in range(num_customers)]

# ランダムに退会日を生成するが、トレーナーを利用している人の退会率を低く設定
exit_dates = []
trainer_usage_list = []
for i in range(num_customers):
    trainer_usage_chance = random.random()
    trainer_usage = "はい" if trainer_usage_chance > 0.6 else "いいえ"  # トレーナー利用率を40%に設定
    trainer_usage_list.append(trainer_usage)
    if trainer_usage == "はい":
        exit_date_chance = 0.1  # トレーナーを利用している人の退会率を10%に設定
    else:
        exit_date_chance = 0.3  # トレーナーを利用していない人の退会率を30%に設定
    exit_date = (datetime.date(2022, 1, 1) + datetime.timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d') if random.random() < exit_date_chance else None
    exit_dates.append(exit_date)

customer_df = pd.DataFrame({
    "顧客ID": range(1, num_customers + 1),
    "名前": names,
    "年齢": ages,
    "性別": genders,
    "登録日": registration_dates,
    "退会日": exit_dates,
    "トレーナー利用": trainer_usage_list
})

# 顧客の利用履歴の生成
num_records = 5000
customer_ids_history = [random.randint(1, num_customers) for _ in range(num_records)]
usage_dates = [(datetime.date(2022, 1, 1) + datetime.timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d') for _ in range(num_records)]
training_times = []
trainer_usage_history = []
for i in range(num_records):
    customer_id = customer_ids_history[i]
    trainer_usage = customer_df.loc[customer_df["顧客ID"] == customer_id, "トレーナー利用"].values[0]
    trainer_usage_history.append(trainer_usage)
    if trainer_usage == "はい":
        training_time = round(random.uniform(1, 2.5), 1)  # トレーナーを利用している人のトレーニング時間は1〜2.5時間
    else:
        training_time = round(random.uniform(0.5, 1.5), 1)  # トレーナーを利用していない人のトレーニング時間は0.5〜1.5時間
    training_times.append(training_time)

history_df = pd.DataFrame({
    "顧客ID": customer_ids_history,
    "利用日": usage_dates,
    "トレーニング時間": training_times,
    "トレーナー利用": trainer_usage_history
})

# 退会者の情報の生成
exit_reasons_list = ["料金が高い", "トレーニング内容", "トレーナーの対応", "移住", "健康問題"]
exit_customer_ids = customer_df[customer_df["退会日"].notna()]["顧客ID"].tolist()
exit_reasons = [random.choice(exit_reasons_list) for _ in exit_customer_ids]

exit_df = pd.DataFrame({
    "顧客ID": exit_customer_ids,
    "退会理由": exit_reasons
})

# Display the first few rows of each dataframe
customer_df.head(), history_df.head(), exit_df.head()
