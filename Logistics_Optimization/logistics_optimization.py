# E:\programming\Project\Portfolio\Logistics_Optimization\logistics_optimization.py

import pandas as pd

# データの読み込み
factory_df = pd.read_csv("tbl_factory.csv")
warehouse_df = pd.read_csv("tbl_warehouse.csv")
cost_df = pd.read_csv("rel_cost.csv")
transaction_df = pd.read_csv("tbl_transaction.csv")

# 1. 各工場と倉庫のペアについて、2019年の輸送回数と輸送コストを計算
transaction_cost = pd.merge(transaction_df, cost_df, on=['factory_id', 'warehouse_id'], how='left')
transaction_cost['total_cost'] = transaction_cost['quantity'] * transaction_cost['cost']
total_cost = transaction_cost.groupby(['factory_id', 'warehouse_id'])['total_cost'].sum().reset_index()

# 2. 輸送回数が多い工場と倉庫のペアを特定
high_freq_pairs = transaction_df.groupby(['factory_id', 'warehouse_id']).size().reset_index()
high_freq_pairs.columns = ['factory_id', 'warehouse_id', 'frequency']
high_freq_pairs = high_freq_pairs.sort_values(by='frequency', ascending=False).head()

# 3. 高コストの輸送ルートを特定
high_cost_routes = total_cost.sort_values(by='total_cost', ascending=False).head()

print("High Frequency Pairs:")
print(high_freq_pairs)
print("\nHigh Cost Routes:")
print(high_cost_routes)

# 倉庫と工場のダミーの位置情報 (緯度, 経度)
location_info = {
    "Tokyo": (35.6895, 139.6917),
    "Osaka": (34.6937, 135.5023),
    "Nagoya": (35.1815, 136.9066),
    "Fukuoka": (33.5902, 130.4017),
    "Sapporo": (43.0618, 141.3545)
}

# 位置情報をDataFrameに追加
factory_df['latitude'] = factory_df['factory_location'].map(lambda x: location_info[x][0])
factory_df['longitude'] = factory_df['factory_location'].map(lambda x: location_info[x][1])

warehouse_df['latitude'] = warehouse_df['warehouse_location'].map(lambda x: location_info[x][0])
warehouse_df['longitude'] = warehouse_df['warehouse_location'].map(lambda x: location_info[x][1])

# (省略)

# 期間別のコストの変動を分析
transaction_cost['month'] = pd.to_datetime(transaction_cost['transaction_date']).dt.month
monthly_costs = transaction_cost.groupby('month')['total_cost'].sum()

print("\nMonthly Costs:")
print(monthly_costs)

# (以下、在庫量や生産量のダミーデータを生成して分析する必要があります。)
# この部分は、実際のビジネス要件やデータに基づいて詳細に検討する必要があります。
