# E:\programming\Project\Portfolio\Logistics_Optimization\create_data.py

import pandas as pd
import random

# 1. 生産工場のデータ
factories = ["Factory_" + str(i) for i in range(5)]
factory_df = pd.DataFrame({
    "factory_id": range(1, len(factories) + 1),
    "factory_name": factories,
    "factory_location": [random.choice(["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Sapporo"]) for _ in factories]
})
factory_df.to_csv("tbl_factory.csv", index=False)

# 2. 倉庫のデータ
warehouses = ["Warehouse_" + str(i) for i in range(5)]
warehouse_df = pd.DataFrame({
    "warehouse_id": range(1, len(warehouses) + 1),
    "warehouse_name": warehouses,
    "warehouse_location": [random.choice(["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Sapporo"]) for _ in warehouses]
})
warehouse_df.to_csv("tbl_warehouse.csv", index=False)

# 3. 倉庫と工場間の輸送コスト
costs = []
for warehouse in warehouse_df["warehouse_id"]:
    for factory in factory_df["factory_id"]:
        # 地理的な偏りの導入
        if (factory_df.loc[factory-1, "factory_location"] == "Tokyo" and warehouse_df.loc[warehouse-1, "warehouse_location"] == "Nagoya") or \
           (factory_df.loc[factory-1, "factory_location"] == "Osaka" and warehouse_df.loc[warehouse-1, "warehouse_location"] == "Fukuoka"):
            cost_value = random.randint(1000, 3000)
        else:
            cost_value = random.randint(3000, 5000)
        
        costs.append({
            "warehouse_id": warehouse,
            "factory_id": factory,
            "cost": cost_value
        })

# 4. 2019年の工場の部品輸送実績
transactions = []
for month in range(1, 13):
    for day in range(1, 29):  # 簡単のため、すべての月を28日としています
        for factory in factory_df["factory_id"]:
            for warehouse in warehouse_df["warehouse_id"]:
                # トレンドの導入: Factory_1 と Warehouse_1 の取引を他のペアよりも頻繁にする
                if factory == 1 and warehouse == 1:
                    quantity_value = random.randint(50, 100)
                else:
                    quantity_value = random.randint(1, 50)
                transactions.append({
                    "transaction_date": f"2019-{month:02}-{day:02}",
                    "factory_id": factory,
                    "warehouse_id": warehouse,
                    "quantity": quantity_value
                })

transaction_df = pd.DataFrame(transactions)
transaction_df.to_csv("tbl_transaction.csv", index=False)
