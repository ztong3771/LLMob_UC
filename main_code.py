import pickle
import numpy as np
import pandas as pd
import torch
import re
import pkl_convert
import openmeteo_requests
import requests_cache
from retry_requests import retry

with open('./data/2019/638.pkl','rb') as f: real_data=pickle.load(f)
with open('./data/loc_map.pkl','rb') as f: map_loc=pickle.load(f)

user_traj_df=pkl_convert.process_llmob_data(real_data)
loc_map_df=pd.DataFrame([map_loc])

#print(user_traj_df)
#print(loc_map_df)
#print(len(user_traj_df))
#print(loc_map_df.shape[1])

cache_session=requests_cache.CachedSession('.cache',expire_after=-1)
retry_session=retry(cache_session,retries=5,backoff_factor=0.2)
openmeteo=openmeteo_requests.Client(session=retry_session)

# Transpose the DataFrame
df_transposed=loc_map_df.T
df_transposed.reset_index(inplace=True)
df_transposed.columns=['raw_info','raw_id']

#Define Extraction Functions

def extract_info(row):
    match=re.search(r'^(.*?)\s*\(([\d\.]+),\s*([\d\.]+)\)',row)
    if match: return match.group(1),match.group(2),match.group(3)
    return None,None,None

def extract_id(raw_id_str):
    if isinstance(raw_id_str,str) and '#' in raw_id_str: return raw_id_str.split('#')[-1]
    return raw_id_str

# Apply extractions
extracted_data=df_transposed['raw_info'].apply(extract_info)
df_transposed['Location_Name']=[x[0] for x in extracted_data]
df_transposed['Latitude']=[x[1] for x in extracted_data]
df_transposed['Longitude']=[x[2] for x in extracted_data]

# Extract the ID
df_transposed['Location_ID']=df_transposed['raw_id'].apply(extract_id)

# Select only the required columns in the specific order
final_df=df_transposed[['Location_Name','Latitude','Longitude','Location_ID']]

#print(final_df)

df_merged=pd.merge(user_traj_df,final_df[['Location_Name','Latitude','Longitude','Location_ID']],on=['Location_Name','Location_ID'],how='left')

#print(df_merged)

def get_historical_weather(lat,lon,date_str):
    url="https://archive-api.open-meteo.com/v1/archive"
    
    # Convert date
    date_obj=pd.to_datetime(date_str)
    start_date=date_obj.strftime('%Y-%m-%d')
    end_date=start_date
    
    params={
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["weather_code"],
        "timezone": "auto"}
    
    try:
        responses=openmeteo.weather_api(url,params=params)
        response=responses[0]
        
        hourly=response.Hourly()
        hourly_code=hourly.Variables(0).ValuesAsNumpy()
        
        target_hour=date_obj.hour
        
        return{"weather_code": hourly_code[target_hour]}
    except Exception as e:
        print(f"Error fetching weather for {lat}, {lon}: {e}")
        return {"weather_code": None}

weather_data=[]

for index,row in df_merged.iterrows():
    if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
        w=get_historical_weather(row['Latitude'],row['Longitude'],row['Full_Datetime'])
        weather_data.append(w)
    else:
        weather_data.append({"weather_code": None})

# Convert weather data into DataFrame
weather_df=pd.DataFrame(weather_data)
final_df2=pd.concat([df_merged.reset_index(drop=True),weather_df],axis=1)

final_df2['Latitude']=pd.to_numeric(final_df2['Latitude'],errors='coerce')
final_df2['Longitude']=pd.to_numeric(final_df2['Longitude'],errors='coerce')

#print(final_df2)
#final_df2.head(20).to_csv('data_with_weather.csv')

from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

final_df2['Full_Datetime']=pd.to_datetime(final_df2['Full_Datetime'])
final_df2=final_df2.sort_values(by=['Date','Time'])

loc_encoder=LabelEncoder()
final_df2['loc_idx']=loc_encoder.fit_transform(final_df2['Location_ID'])
num_nodes=len(loc_encoder.classes_)

src_nodes=final_df2['loc_idx'].values[:-1]
dst_nodes=final_df2['loc_idx'].values[1:]
edge_index=torch.tensor(np.array([src_nodes,dst_nodes]),dtype=torch.long)

#print(num_nodes,edge_index.shape[1])

unique_locs=final_df2.drop_duplicates(subset=['loc_idx']).sort_values('loc_idx')

cat_encoder=LabelEncoder()
unique_locs['cat_idx']=cat_encoder.fit_transform(unique_locs['Category'])
category_one_hot=torch.nn.functional.one_hot(
    torch.tensor(unique_locs['cat_idx'].values),
    num_classes=len(cat_encoder.classes_)).float()

coords=torch.tensor(unique_locs[['Latitude','Longitude']].values,dtype=torch.float)
coords_norm=(coords - coords.mean(0)) / coords.std(0)

x=torch.cat([category_one_hot,coords_norm],dim=1)
data=Data(x=x,edge_index=edge_index)

import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MobilityGNN(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(MobilityGNN,self).__init__()
        self.conv1=GATConv(in_channels,hidden_channels,heads=4,concat=True)
        self.conv2=GATConv(hidden_channels*4,out_channels,heads=1,concat=False)

    def forward(self,x,edge_index):
        x=F.elu(self.conv1(x,edge_index))
        x=F.dropout(x,p=0.6,training=self.training)
        x=self.conv2(x,edge_index)
        return x

in_dim=data.x.shape[1]
hidden_dim=64
out_dim=128
model=MobilityGNN(in_dim,hidden_dim,out_dim)

location_embeddings=model(data.x,data.edge_index)
#print(location_embeddings)

embeddings_np=location_embeddings.detach().cpu().numpy()
original_ids=loc_encoder.classes_
embedding_df=pd.DataFrame(embeddings_np,index=original_ids)
embedding_df.columns=[f'emb_{i}' for i in range(embeddings_np.shape[1])]
embedding_df.index.name='Location_ID'
embedding_df.reset_index(inplace=True)

output_filename='location_embeddings_result.csv'
embedding_df.to_csv(output_filename,index=False)

#print(embedding_df)