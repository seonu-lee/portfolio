from sqlalchemy import create_engine
import pandas as pd


csv_path = "PaySim\datasets\PaySim.csv"
df = pd.read_csv(csv_path)
# SQLite DB 생성/연결 (local DB 파일 생성됨)
engine = create_engine("sqlite:///paysim.db")
# PaySim이라는 테이블명으로 저장
df.to_sql("PaySim", engine, if_exists="replace", index=False)

# 수정된 SQL 쿼리 (DestStats 뒤의 쉼표 제거)
query_mart = """
WITH UserStats AS (
    SELECT 
        nameOrig,
        COUNT(step) AS user_frequency,
        SUM(amount) AS user_monetary,
        AVG(amount) AS user_avg_amount
    FROM PaySim
    GROUP BY nameOrig
), 

DestStats AS (
    SELECT 
        nameDest,
        COUNT(*) AS dest_received_count,
        SUM(isFraud) AS dest_fraud_history_count
    FROM PaySim
    GROUP BY nameDest
) 

SELECT 
    p.step,
    (p.step % 24) AS hour,
    p.type,
    p.amount,
    CASE 
        WHEN p.amount <= 100000 THEN 'Small'
        WHEN p.amount <= 1000000 THEN 'Medium'
        ELSE 'Large'
    END AS amount_segment,
    p.nameOrig,
    p.nameDest,
    p.isFraud,
    u.user_frequency,
    u.user_monetary,
    u.user_avg_amount,
    d.dest_received_count,
    d.dest_fraud_history_count,
    (CASE WHEN u.user_avg_amount > 0 THEN p.amount / u.user_avg_amount ELSE 0 END) AS amount_ratio_to_avg
FROM PaySim p
LEFT JOIN UserStats u ON p.nameOrig = u.nameOrig
LEFT JOIN DestStats d ON p.nameDest = d.nameDest;
"""

# 데이터 불러오기 및 저장
df_mart = pd.read_sql(query_mart, engine)
df_mart.to_csv('PaySim/datasets/paysim_data_mart.csv', index=False)

print("데이터 마트가 'paysim_data_mart.csv' 파일로 성공적으로 저장되었습니다.")