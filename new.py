import pickle
import pandas as pd
import torch
import re
import pkl_convert
import openmeteo_requests
import requests_cache
from retry_requests import retry

# 假設你已經有 .pkl 檔案，這裡模擬讀取過程

with open('./data/2019/13.pkl', 'rb') as f:
    real_data = pickle.load(f)
with open('./data/loc_map.pkl', 'rb') as f:
    map_loc = pickle.load(f)

user_traj_df = pkl_convert.process_llmob_data(real_data)
loc_map_df = pd.DataFrame([map_loc])

#loc_map_df.to_csv('loc_map.csv')

print(user_traj_df)
print(loc_map_df)

print(f"載入軌跡數據: {len(user_traj_df)} 筆")
print(f"載入地點映射: {loc_map_df.shape[1]} 個地點")

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

'''
# ---------------------------------------------------------
# STEP 2: PARSE DATAFRAME 2 (The cleanup)
# loc_map_df has coordinates in the columns, not the rows. We need to extract them.
# ---------------------------------------------------------

location_map = []

# Iterate over the column names of loc_map_df
for col in loc_map_df.columns:
    # Use Regex to find patterns like "Name (Lat, Lon)"
    # Pattern explanation:
    # (.*) captures the Name
    # \( captures the literal opening parenthesis
    # (\d+\.\d+) captures the float latitude
    # , captures the comma space
    # (\d+\.\d+) captures the float longitude
    match = re.search(r"(.*) \((\d+\.\d+), (\d+\.\d+)\)", col)
    
    if match:
        name = match.group(1).strip()
        lat = match.group(2)
        lon = match.group(3)
        
        # We also grab the ID from the first row of data corresponding to this column
        # Just in case you need to merge by that weird ID string later
        loc_id_from_loc_map_df = loc_map_df[col].iloc[0]
        
        location_map.append({
            'Location_Name': name,
            'latitude': float(lat),
            'longitude': float(lon),
            'loc_map_df_ID': loc_id_from_loc_map_df 
        })

# Create a clean reference dataframe
df_ref = pd.DataFrame(location_map)

print("--- Extracted Reference Data (Cleaned loc_map_df) ---")
print(df_ref.head())
print("-" * 30)
'''

# 2. Transpose (Flip) the DataFrame
# The columns (Location + Coords) become the index, and the row (IDs) becomes the data
df_transposed = loc_map_df.T

# Reset index to turn the 'Location (Lat, Lon)' string into a proper column we can work with
df_transposed.reset_index(inplace=True)
df_transposed.columns = ['raw_info', 'raw_id']

# 3. Define Extraction Functions

def extract_info(row):
    # Regex to capture Name, Lat, and Lon from format "Name (Lat, Lon)"
    # Pattern explanation:
    # ^(.*?)       -> Capture the name (non-greedy) at the start
    # \s*          -> Optional whitespace
    # \(           -> Literal opening parenthesis
    # ([\d\.]+)    -> Capture Latitude (digits and dots)
    # ,\s*         -> Comma and whitespace
    # ([\d\.]+)    -> Capture Longitude
    # \)           -> Literal closing parenthesis
    match = re.search(r'^(.*?)\s*\(([\d\.]+),\s*([\d\.]+)\)', row)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

def extract_id(raw_id_str):
    # Split by '#' and take the last part (the ID number)
    if isinstance(raw_id_str, str) and '#' in raw_id_str:
        return raw_id_str.split('#')[-1]
    return raw_id_str

# 4. Apply extractions
# Zip the results of the regex into new columns
extracted_data = df_transposed['raw_info'].apply(extract_info)
df_transposed['Location_Name'] = [x[0] for x in extracted_data]
df_transposed['Latitude'] = [x[1] for x in extracted_data]
df_transposed['Longitude'] = [x[2] for x in extracted_data]

# Extract the ID
df_transposed['Location_ID'] = df_transposed['raw_id'].apply(extract_id)

# 5. Final Formatting
# Select only the required columns in the specific order
final_df = df_transposed[['Location_Name', 'Latitude', 'Longitude', 'Location_ID']]

print(final_df)

# ---------------------------------------------------------
# STEP 3: MERGE
# ---------------------------------------------------------

# CRITICAL NOTE:
# In your sample, user_traj_df Location_ID is an int (e.g., 2582)
# loc_map_df ID is a string (e.g., Shrine#1). They DO NOT MATCH.
# However, 'Location_Name' (Shrine) DOES match.
# I will merge on Location_Name to ensure it works. 
# If your real data has matching IDs, change 'on' to 'Location_ID'.

df_merged = pd.merge(user_traj_df, final_df[['Location_Name', 'Latitude', 'Longitude', 'Location_ID']], on=['Location_Name', 'Location_ID'], how='left')

# ---------------------------------------------------------
# STEP 4: OUTPUT
# ---------------------------------------------------------

print("\n--- Final Merged DataFrame 1 ---")
print(df_merged)

# Optional: Save to CSV
# df_merged.to_csv("final_output.csv", index=False)

def get_historical_weather(lat, lon, date_str):
    """
    查詢特定經緯度與日期的天氣。
    注意：為了效能，實務上建議批次處理，這裡為了邏輯清晰演示單筆查詢邏輯。
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # 轉換日期格式
    date_obj = pd.to_datetime(date_str)
    start_date = date_obj.strftime('%Y-%m-%d')
    end_date = start_date # 我們只查當天
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "weather_code", "rain"], # 你可以增加濕度、風速等
        "timezone": "auto"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # 處理小時資料
        hourly = response.Hourly()
        hourly_temp = hourly.Variables(0).ValuesAsNumpy()
        hourly_code = hourly.Variables(1).ValuesAsNumpy()
        hourly_rain = hourly.Variables(2).ValuesAsNumpy()
        
        # 我們需要對應到具體小時 (e.g., 12:00)
        target_hour = date_obj.hour
        
        return {
            "temp_c": hourly_temp[target_hour],
            "weather_code": hourly_code[target_hour], # WMO Weather code
            "rain_mm": hourly_rain[target_hour]
        }
    except Exception as e:
        print(f"Error fetching weather for {lat}, {lon}: {e}")
        return {"temp_c": None, "weather_code": None, "rain_mm": None}
    
'''
# --- 2. 讀取與處理 loc_map.pkl (地點映射檔) ---
# 建立 Location ID 到 (Lat, Lon) 的映射字典
# 注意：你的 file2.csv 顯示 header 是地點名稱，row 0 是 ID。我們需要轉置或提取。
# 這裡假設我們要從 location string 中提取座標

def extract_coords(loc_string):
    # 使用正則表達式抓取括號內的數字: (36.310, 139.970)
    match = re.search(r'\(([\d\.]+),\s*([\d\.]+)\)', str(loc_string))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

# --- 3. 讀取主數據 data.pkl ---
# 假設 df 結構如 file.csv: Date, Time, Location_Name, Location_ID...
# 我們需要先從 loc_map 把座標合併進來。
# 如果 Location_Name 已經包含座標字串 (如 file2.csv 的 header)，我們直接解析它。

# 這裡演示直接對 Location_Name 解析 (如果主表也有這個格式)
# 如果主表只有 ID，你需要先跟 loc_map merge。這裡假設我們已經有了地點名稱字串。

print("正在解析經緯度...")
# 假設 Location_Name 欄位長得像 "Liquor Store (36.310, 139.970)"
# 如果你的 file.csv Location_Name 只有 "Movie Theater" 沒有座標，
# 你必須先用 Location_ID 去 join loc_map.pkl 取出完整的名稱或座標。

# 範例：擴充經緯度欄位
# 這裡為了範例，我們假設已經透過 ID mapping 拿到了帶座標的名稱字串
# 若需要 Mapping 邏輯：
# id_to_loc_str = { 123: "Movie Theater (35.123, 139.456)", ... } form loc_map.pkl
# df['Full_Loc_Str'] = df['Location_ID'].map(id_to_loc_str)

coordinates = user_traj_df['Location_Name'].apply(extract_coords) 
# 注意：如果你的 Location_Name 欄位只是純文字（如 "Movie Theater"），你需要先 Merge loc_map

print(coordinates)

loc_map_df['lat'] = [c[0] for c in coordinates]
loc_map_df['lon'] = [c[1] for c in coordinates]
'''

# --- 4. 加入天氣資料 ---
print("正在抓取歷史天氣資料 (這可能需要一點時間)...")

weather_data = []

# 為了避免 API Rate Limit，建議使用 apply 但要注意量大時的效能
# 這裡示範逐行處理 (Row-wise)
for index, row in df_merged.iterrows():
    if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
        # 呼叫 API
        w = get_historical_weather(row['Latitude'], row['Longitude'], row['Full_Datetime'])
        weather_data.append(w)
    else:
        weather_data.append({"temp_c": None, "weather_code": None, "rain_mm": None})

# 將天氣資料轉為 DataFrame 並合併
weather_df = pd.DataFrame(weather_data)
final_df2 = pd.concat([df_merged.reset_index(drop=True), weather_df], axis=1)

final_df2['Latitude'] = pd.to_numeric(final_df2['Latitude'], errors='coerce')
final_df2['Longitude'] = pd.to_numeric(final_df2['Longitude'], errors='coerce')

# --- 5. 輸出結果 ---
print(final_df2)
#final_df2.head(20).to_csv('data_with_weather.csv')
print("完成！")

import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# 確保按時間排序
final_df2['Full_Datetime'] = pd.to_datetime(final_df2['Full_Datetime'])
final_df2 = final_df2.sort_values(by=['Date', 'Time'])

# 2. 構建節點與邊
# 建立 Location_ID 到 0~N 的索引映射
loc_encoder = LabelEncoder()
final_df2['loc_idx'] = loc_encoder.fit_transform(final_df2['Location_ID'])
num_nodes = len(loc_encoder.classes_)

# 建立 Edge Index (來源節點 -> 目標節點)
# 這裡假設每一行是同一用戶的連續軌跡 (若有多用戶需先 GroupBy UserID)
src_nodes = final_df2['loc_idx'].values[:-1]
dst_nodes = final_df2['loc_idx'].values[1:]
edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

print(f"圖構建完成: {num_nodes} 個節點, {edge_index.shape[1]} 條邊")

import numpy as np

# 3. 提取節點特徵矩陣 (Node Features)
# 我們需要為每個唯一的 Location_ID 聚合特徵
unique_locs = final_df2.drop_duplicates(subset=['loc_idx']).sort_values('loc_idx')

# 類別 One-Hot Encoding
cat_encoder = LabelEncoder()
unique_locs['cat_idx'] = cat_encoder.fit_transform(unique_locs['Category'])
category_one_hot = torch.nn.functional.one_hot(
    torch.tensor(unique_locs['cat_idx'].values), 
    num_classes=len(cat_encoder.classes_)
).float()

# 座標歸一化 (Lat/Lon)
coords = torch.tensor(unique_locs[['Latitude', 'Longitude']].values, dtype=torch.float)
coords_norm = (coords - coords.mean(0)) / coords.std(0)

# 組合特徵: [Category_OneHot, Lat, Lon]
x = torch.cat([category_one_hot, coords_norm], dim=1)

# 建立 PyG Data 對象
data = Data(x=x, edge_index=edge_index)

import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MobilityGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MobilityGNN, self).__init__()
        # 使用多頭注意力機制增加模型的表達能力
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # 第一層 + 激活函數 + Dropout (增加混亂度，防止過擬合)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        
        # 第二層輸出最終 Embeddings
        x = self.conv2(x, edge_index)
        return x

# 初始化模型
in_dim = data.x.shape[1]
hidden_dim = 64
out_dim = 128  # 這是我們要給 LLM 的向量維度
model = MobilityGNN(in_dim, hidden_dim, out_dim)

# 獲取地點嵌入 (無需訓練循環的示例，實際需定義 Loss 進行預訓練)
location_embeddings = model(data.x, data.edge_index)

print(location_embeddings)