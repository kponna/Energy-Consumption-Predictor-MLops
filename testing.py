import pandas as pd
import os
df = pd.read_csv("/home/karthikponna/kittu/Energy generation prediction project/Energy-Generation-Predictor-MLops/valid_data/batch_data.csv")
print(df)
df = df.drop(columns='value',axis = 1)
output_path = "valid_data/data.csv"
print(df)
df.sort_values(by="period",ascending=True,inplace=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True) 
df.to_csv(output_path, index=False)