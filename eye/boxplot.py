import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
# ファイルがあるディレクトリのパス
directory_path = '/home/sowa/prog/pointcloud/csv/potato_1118/result/'

# ディレクトリ内のすべてのCSVファイルを取得
csv_files_p1 = [file for file in os.listdir(directory_path) if file.startswith('peel_results_01_') and file.endswith('.csv')]
csv_files_p2 = [file for file in os.listdir(directory_path) if file.startswith('peel_results_02_') and file.endswith('.csv')]
csv_files_u1 = [file for file in os.listdir(directory_path) if file.startswith('unpeel_results_01_') and file.endswith('.csv')]
csv_files_u2 = [file for file in os.listdir(directory_path) if file.startswith('unpeel_results_02_') and file.endswith('.csv')]
# peel_results_01_ と .csv の間の文字列を格納する配列
extracted_strings_p1 = []
all_accuracy_data_p1 = []
average_accuracy_p1=[]

extracted_strings_p2 = []
all_accuracy_data_p2 = []
average_accuracy_p2=[]

extracted_strings_u1 = []
all_accuracy_data_u1 = []
average_accuracy_u1=[]

extracted_strings_u2 = []
all_accuracy_data_u2 = []
average_accuracy_u2=[]
    

for file in csv_files_p1:
    extracted_string = re.search(r'peel_results_01_(.*?)\.csv', file).group(1)
    extracted_strings_p1.append(extracted_string)
# 各CSVファイルからAccuracy列のデータを取得してリストに追加
for file in csv_files_p1:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)
    accuracy_data = data['Accuracy']
    all_accuracy_data_p1.append(accuracy_data)
    average = accuracy_data.mean()  # 平均値を計算
    average_accuracy_p1.append(average) 

for file in csv_files_p2:
    extracted_string = re.search(r'peel_results_02_(.*?)\.csv', file).group(1)
    extracted_strings_p2.append(extracted_string)
# 各CSVファイルからAccuracy列のデータを取得してリストに追加
for file in csv_files_p2:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)
    accuracy_data = data['Accuracy']
    all_accuracy_data_p2.append(accuracy_data)
    average = accuracy_data.mean()  # 平均値を計算
    average_accuracy_p2.append(average) 

for file in csv_files_u1:
    extracted_string = re.search(r'unpeel_results_01_(.*?)\.csv', file).group(1)
    extracted_strings_u1.append(extracted_string)
# 各CSVファイルからAccuracy列のデータを取得してリストに追加
for file in csv_files_u1:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)
    accuracy_data = data['Accuracy']
    all_accuracy_data_u1.append(accuracy_data)
    average = accuracy_data.mean()  # 平均値を計算
    average_accuracy_u1.append(average) 

for file in csv_files_u2:
    extracted_string = re.search(r'unpeel_results_02_(.*?)\.csv', file).group(1)
    extracted_strings_u2.append(extracted_string)
# 各CSVファイルからAccuracy列のデータを取得してリストに追加
for file in csv_files_u2:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)
    accuracy_data = data['Accuracy']
    all_accuracy_data_u2.append(accuracy_data)
    average = accuracy_data.mean()  # 平均値を計算
    average_accuracy_u2.append(average) 

desired_order = ['h', 's', 'v', 'hs', 'hv', 'sv', 'hsv']
extracted_strings_p1 = [x for x in desired_order if x in extracted_strings_p1]
extracted_strings_p2 = [x for x in desired_order if x in extracted_strings_p2]
extracted_strings_u1 = [x for x in desired_order if x in extracted_strings_u1]
extracted_strings_u2 = [x for x in desired_order if x in extracted_strings_u2]
# 箱ひげ図を作成
plt.figure(figsize=(8, 6))  # グラフサイズを設定（任意のサイズに変更可能）
print
# DataFrameを作成し、extracted_stringsとall_accuracy_dataを結合
data_p1 = pd.DataFrame({'Condition Setting': extracted_strings_p1, 'Accuracy_Average_p1': average_accuracy_p1})
print(data_p1)
data_p2 = pd.DataFrame({'Condition Setting': extracted_strings_p2, 'Accuracy_Average_p2': average_accuracy_p2})
print(data_p2)
data_u1 = pd.DataFrame({'Condition Setting': extracted_strings_u1, 'Accuracy_Average_u1': average_accuracy_u1})
print(data_u1)
data_u2 = pd.DataFrame({'Condition Setting': extracted_strings_u2, 'Accuracy_Average_u2': average_accuracy_u2})
print(data_u2)

all_data =pd.DataFrame({'Condition Setting': extracted_strings_u2,'Accuracy_Average_p1': average_accuracy_p1,'Accuracy_Average_p2': average_accuracy_p2, 'Accuracy_Average_u1': average_accuracy_u1,'Accuracy_Average_u2': average_accuracy_u2})
print(all_data)
all_data.to_csv('/home/sowa/prog/pointcloud/csv/potato_1118/result/Accuracy_Average_allresults.csv', index=False)

# サブプロットの作成
plt.figure(figsize=(10, 8))

# サブプロット1
plt.subplot(2, 2, 1)
plt.boxplot(all_accuracy_data_p1, labels=extracted_strings_p1)
plt.title('Peeling Potato A')  # サブプロットのタイトルを設定
plt.ylim(-5, 105) 
plt.grid(True,axis='y')
plt.tight_layout()  

# サブプロット2
plt.subplot(2, 2, 2)
plt.boxplot(all_accuracy_data_p2, labels=extracted_strings_p2)
plt.title('Peeling Potato B')  # サブプロットのタイトルを設定
plt.ylim(-5, 105) 
plt.grid(True,axis='y')
plt.tight_layout()  

# サブプロット3
plt.subplot(2, 2, 3)
plt.boxplot(all_accuracy_data_u1, labels=extracted_strings_u1)
plt.title('Unpeeling Potato A')  # サブプロットのタイトルを設定
plt.ylim(-5, 105) 
plt.grid(True,axis='y')
plt.tight_layout()  

# サブプロット4
plt.subplot(2, 2, 4)
plt.boxplot(all_accuracy_data_u2, labels=extracted_strings_u2)
plt.title('Unpeeling Potato B')  # サブプロットのタイトルを設定
plt.ylim(-5, 105) 
plt.grid(True,axis='y')    #
plt.tight_layout()  # レイアウトを調整
#plt.grid(True,axis='y')
# グラフを表示
#plt.show()

# データの作成

data = {
    'Condition Setting': all_data['Condition Setting'],
    'Accuracy_Average_p1': all_data['Accuracy_Average_p1'].round(2),
    'Accuracy_Average_p2': all_data['Accuracy_Average_p2'].round(2),
    'Accuracy_Average_u1': all_data['Accuracy_Average_u1'].round(2),
    'Accuracy_Average_u2': all_data['Accuracy_Average_u2'].round(2)
}


data['Average_All'] = all_data[['Accuracy_Average_p1', 'Accuracy_Average_p2', 'Accuracy_Average_u1', 'Accuracy_Average_u2']].mean(axis=1).round(2)

# 新しい DataFrame を作成
new_data = pd.DataFrame(data)
print(new_data)

# 新しい DataFrame の表示
fig, ax = plt.subplots(figsize=(8, 6))

# 表を作成
table = ax.table(cellText=new_data.values,
                 colLabels=new_data.columns,
                 loc='center')

# セルの書式を調整
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# グラフとして表示される領域を調整
ax.axis('off')

# 表を表示
plt.show()