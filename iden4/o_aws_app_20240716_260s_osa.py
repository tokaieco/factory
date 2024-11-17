from array import array
import pandas as pd
import numpy as np
import datetime
import random
import streamlit as st
#追加
import json

#aws
# import awswrangler as wr
# import boto3
# from st_aggrid import AgGrid
st.title("全体最適スケジュール")
# ここから
# %% [markdown]
# # スケジュール算出プログラム




# %%
# import sagemaker, boto3, os
#aws
# excel_path = "s3://sar-prod-schedule/f_iden_5moji.xlsm"
#vsc
#excel_path =  "C:/Users/tokai/OneDrive/ドキュメント/pv_file/vscode/iden3/f_iden_5moji.xlsm"
excel_path =  "C:/Users/tokai/OneDrive/ドキュメント/pv_file/vscode/iden3/f_iden_osa3.xlsm"



# %%

# S3のバケット名
# bucket_name = 'sar-prod-schedule'
# session = boto3.Session()


def read_excels():
    #vsc
    off_day_df  = pd.read_excel(excel_path, sheet_name="yukyu")                         # 休みの希望表
    skill_df    = pd.read_excel(excel_path, sheet_name="skill", header=[0, 1])          # スキル表
    due_date_df = pd.read_excel(excel_path, sheet_name="calendar")                      # 納期表
    start_date: datetime.datetime = pd.read_excel(excel_path, sheet_name="start", header=None).iloc[0,1]
    due_num = pd.read_excel(excel_path, sheet_name="start", header=None).iloc[2,1]
    jigu_date = pd.read_excel(excel_path, sheet_name="jigu")
    #追加
    sheet1 = pd.read_excel(excel_path, sheet_name="Sheet1")
     #治具追加
    due_date_df.insert(4, '治具', 0)
    #追加
    due_date_df['治具'] = due_date_df['治具'].astype(str)  # 文字列型に変換
    for u1 in range(0, len(jigu_date) ):
        jh=jigu_date.iloc[u1,0]
        
        jk=jigu_date.iloc[u1,1]
        
        for u2 in range(0,len(due_date_df)):
            if due_date_df.iloc[u2,1]==jh:
                due_date_df.iloc[u2,4]=jigu_date.iloc[u1,2]
    # 空データ(NaN)の置換(0)
    off_day_df.fillna(0, inplace=True)
   
    due_date_df.fillna(0, inplace=True)

    # データの置換
    off_day_df.replace("◎", 2, inplace=True)                                            # 希望休（◎）を 2 に置換
    due_date_df.replace("☆", 3, inplace=True)                                          # 納期（☆）を 3 に置換

    # 列ごとのデータ型(dtype)を、intに変更しておく
    off_day_df[off_day_df.columns[2:]] = off_day_df[off_day_df.columns[2:]].astype("int64")
    #治具追加による追加
    
    due_date_df[due_date_df.columns[6:]] = due_date_df[due_date_df.columns[6:]].astype("int64")   #スキルグループ→工程による列追加

    

    
    off_days = [date for date in due_date_df.columns[6:] if date.strftime("%a") in ["Sat","Sun"]]
    # 土日の日付を削除する
    off_day_df.drop(columns=off_days, inplace=True)
    due_date_df.drop(columns=off_days, inplace=True)
    #追加
    sheet1=sheet1.loc[:,["品番","品名","得意先","受注数量","工程  略称","工程順位"]]
    print('sheet1',sheet1)

    return off_day_df, skill_df, due_date_df, start_date, due_num, jigu_date,sheet1

def execAdjust(
    off_day_df: pd.DataFrame,
    skill_df: pd.DataFrame,
    due_date_df: pd.DataFrame,
    start_date: datetime.datetime,
    selected_start_date: datetime.datetime,
    due_num: pd.DataFrame,
    jigu_date: pd.DataFrame,
    ):

    st.session_state.adjusted_schedule = pd.DataFrame()

    off_day_df, skill_df, due_date_df, start_date , due_num, jigu_date ,sheet1= read_excels()
    
    # 調整後の納期表（納期表のコピー）
    adjusted_due_date_df = due_date_df.copy()
   
    copy_off_day_df = off_day_df.copy()
    copy_due_date_df = due_date_df.copy()
    copy_off_day_df.replace("", 0, inplace=True)
    #移動    
    product_num = copy_due_date_df.loc[:, "品番"]

    #親の保存
    parent=[]
    new_due_date=[]
    #for i in range(100):5は少ない、10で安定効果
    #3世代へ変更 停止用
    times=30
    for j in range(times):
        #第1世代
        adjusted_due_date_dfs,worker_list,process_num32 = first_gen(off_day_df, skill_df, due_date_df, product_num,start_date)

        df1=adjusted_due_date_dfs.iloc[:,:7]

        adjusted_due_date_df20= pre_evalution(adjusted_due_date_dfs)
        #左追加
        df2=adjusted_due_date_df20
       
        # DataFrameの各行に対して3を2移動
        df3 = df2.apply(move_three, axis=1)
        df_updated=pd.concat([df1,df3],axis=1,join='inner')
        
        df=df_updated
        # I列以降の部分を抽出
        data_cols = df.columns[7:]
        # 製造番号ごとに処理を行う
        for serial_num, group in df.groupby('製造番号'):
            sorted_group = group.sort_values(by='工程')
            indices = sorted_group.index
            # 工程番号1の位置を取得
            process1_pos = None
            for col in data_cols:
                if df.at[indices[0], col] == 3:
                    process1_pos = df.columns.get_loc(col)
                    break
            if process1_pos is None:
                continue  # 工程番号1に3がない場合は次の製造番号へ
        # 各工程番号について処理
            for idx in indices:
                row = df.loc[idx]
                process_num = row['工程']
                # 工程番号1の数値3は移動しない
                if process_num == 1:
                    continue
                # 数値3を右に移動
                for col in data_cols:
                    if row[col] == 3:
                        current_col_idx = df.columns.get_loc(col)
                        target_col_idx = process1_pos + (process_num - 1)
                        # 数値2がいる場合はさらに右に移動
                        while target_col_idx < len(df.columns):
                            target_col = df.columns[target_col_idx]
                            if df.at[idx, target_col] == 0:
                                df.at[idx, col] = 0
                                df.at[idx, target_col] = 3
                                break
                            elif df.at[idx, target_col] == 2:
                                target_col_idx += 1
                            else:
                                break
        # 調整後のデータを保存
        df25=df
 
        adjusted_due_date_df30=df25.iloc[:,7:]

        score20 ,count_df= evalution_function(adjusted_due_date_df30,jigu_date,start_date,df25,adjusted_due_date_dfs,due_num)
        #第1世代を格納

    parent.append([score20,j, adjusted_due_date_df30,adjusted_due_date_dfs.iloc[:,3],count_df])

    parent = sorted(parent, key=lambda x: -x[0])
    all_top=parent[0]
    # #最強個体の保存
    print('all_top',all_top)
    
    #最強個体の保存

    x=all_top[2].replace(3,'☆').replace(2,'◎').replace(0,'')
    x.columns=adjusted_due_date_df20.columns
    all_sagyousha=all_top[3]
    all_count_df=all_top[4]
    all_count_df.to_excel('all_count_df.xlsx')
    #追加
    path_a =  "C:/Users/tokai/OneDrive/ドキュメント/pv_file/vscode/iden3/all_count_df.json"
    json_a = all_count_df.to_json(orient="index", indent=4, double_precision=15, force_ascii=False)
    json_a = json_a.replace("\/", "/")
    with open(path_a, "w", encoding="utf-8") as f_a:
        f_a.write(json_a)
    
    #必要
    due_date_df_copy= due_date_df.copy()
    due_date_df_copy.loc[due_date_df.index, due_date_df.columns[6:]] = x.values
    due_date_df_copy= due_date_df_copy.set_index('製造番号')

    date_columns = pd.to_datetime(due_date_df_copy.columns[5:])
    formatted_dates = date_columns.strftime('%m/%d')
    due_date_df_copy.columns = list(due_date_df_copy.columns[:5]) + list(formatted_dates)
    date_column = pd.to_datetime(due_date_df_copy['納期'])
    formatted_dates2 = date_column.dt.strftime('%m/%d')
    due_date_df_copy['納期']= formatted_dates2
    y=due_date_df_copy
    df_l=pd.DataFrame(worker_list)

    combined_df0=adjusted_due_date_dfs.iloc[:,3:]

    combined_df=combined_df0.drop(['スキルグループ','治具','納期'],axis=1)
    #変更
    # 作業者をインデックスに、日付を列に設定
    df3 = combined_df.set_index('作業者')#x
    # 日付ごとに数値 '3' がいくつあるかカウント
    df_3_count = (df3 == 3).astype(int)
    # 作業者ごとに数値 '3' を集計
    count_df = df_3_count.groupby(df3.index).sum()
    #追加
    y.reset_index(drop=True, inplace=True)
    z=pd.concat([pd.Series(all_sagyousha, name='作業者'),y],axis=1)
    z.to_excel( 'aws716_260o.xlsx', index=False)
    st.session_state.adjusted_schedule = z
    #追加
    path =  "C:/Users/tokai/OneDrive/ドキュメント/pv_file/vscode/iden3/aws716_260o.json"
    json_z = z.to_json(orient="index", indent=4, double_precision=15, force_ascii=False)
    json_z = json_z.replace("\/", "/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_z)
    return

# # %%


def init():
    

    uploaded_file = st.file_uploader(
        "エクセルファイルを更新したい場合はファイルを選択して「エクセルを更新ボタン」を押してください", accept_multiple_files=False)
#aws
    # if st.button("エクセルを更新"):
    #     if uploaded_file:
    #         upload_to_s3(uploaded_file)

#vsc
    if st.button("エクセルを更新"):
        uploaded_files = st.file_uploader("ファイルを選択してください", accept_multiple_files=True)


    off_day_df, skill_df, due_date_df, start_date, due_num, jigu_date,sheet1 = read_excels()

    product_num = due_date_df.loc[:, "品番"]
    product_num=str(product_num)

    off_day_df.fillna('', inplace=True)

    day_list = off_day_df.columns[2:].tolist()
    day_config = {}
    for i, j in enumerate(day_list):
        label = j.strftime("%Y-%m-%d")
        day_config[j.strftime("%Y-%m-%d %H:%M:%S")] = st.column_config.SelectboxColumn(label, options=[
            "◎",
            ""
        ])

    copy_due_date_df = due_date_df.copy()
    due_day_list = copy_due_date_df.columns[6:].tolist()
    rename_due_columns = {}
    for i, j in enumerate(due_day_list):
        rename_due_columns[j] = j.strftime("%Y-%m-%d")
    copy_due_date_df = copy_due_date_df.rename(columns=rename_due_columns)
    #print('copy_due_date_df',copy_due_date_df )
    #copy_due_date_df.to_excel('copy_due_date_df.xlsx')

    kyuka_tab, start_date_tab, calendar = st.tabs(
        ["休暇", "開始日", "最適化されていないスケジュール"])
    

    with kyuka_tab:
        edited_off_day_df = st.data_editor(
            off_day_df,
            column_config=day_config,
            hide_index=True
        )

    with start_date_tab:
        selected_start_date = st.date_input("開始日を設定してください", start_date, min_value=datetime.date(
            2024, 6, 3))

    with calendar:
        st.write("※変更できません")
        copy_due_date_df.fillna('', inplace=True)
        copy_due_date_df["納期"] = copy_due_date_df["納期"].dt.strftime("%m-%d")
        st.table(copy_due_date_df)
#追加
    st.button("最適化スタート", on_click=execAdjust, args=(edited_off_day_df, skill_df, due_date_df,product_num,
              datetime.datetime.combine(selected_start_date, datetime.datetime.min.time()), due_num, jigu_date))

    if "adjusted_schedule" not in st.session_state:
        st.session_state.adjusted_schedule = pd.DataFrame()

    st.table(st.session_state.adjusted_schedule)

#aws
# def upload_to_s3(file):
#     s3 = boto3.client('s3')

#     try:
#         s3.upload_fileobj(file, bucket_name, "f_iden_5moji.xlsm")
#     except Exception as e:
#         st.error(f"Error uploading file: {e}")

#追加
    path_s =  "C:/Users/tokai/OneDrive/ドキュメント/pv_file/vscode/iden3/sheet1.json"


    json_s = sheet1.to_json(orient="index", indent=4, double_precision=15, force_ascii=False)
    json_s = json_s.replace("\/", "/")
    with open(path_s, "w", encoding="utf-8") as f_s:
        f_s.write(json_s)


# ## 3.第1世代関数

# %%
def move_three(row):
    # 現在の行の値をリストに変換
    values = row.values.tolist()
    
    # 3の位置を探す
    three_index = [i for i, x in enumerate(values) if x == 3]
    if not three_index:
        return row  # 3がない場合はそのまま返す

    # 1と2の位置を探す
    one_index = next((i for i, x in enumerate(values) if x == 1), None)
    two_index = next((i for i, x in enumerate(values) if x == 2), None)
    
    # 3が1より左にある場合、1の右側に移動させる
    if one_index is not None and three_index[0] < one_index:
        possible_positions = list(range(one_index + 1, len(values)))
        if two_index is not None:
            possible_positions = [pos for pos in possible_positions if pos < two_index]
        
        if possible_positions:
            new_position = random.choice(possible_positions)
            values[three_index[0]] = 0
            values[new_position] = 3
    
    # 3が1の右側にあり、1と3の間で移動する場合
    elif one_index is not None and three_index[0] > one_index:
        possible_positions = list(range(one_index + 1, three_index[0]))

        if two_index is not None:
            possible_positions = [pos for pos in possible_positions if pos < two_index]

        
        if possible_positions:
            new_position = random.choice(possible_positions)
            values[three_index[0]] = 0
            values[new_position] = 3

    # 元の行形式に戻す
    return pd.Series(values, index=row.index)


def first_gen(off_day_df: pd.DataFrame, skill_df: pd.DataFrame, due_date_df: pd.DataFrame, product_num: pd.Series,start_date: datetime.datetime):
    # 調整後の納期表（納期表のコピー）　→　以降、『調整納期表』　と呼ぶ
    adjusted_due_date_df: pd.DataFrame = due_date_df.copy()

    # 作業者の割り振り　と　作業者の希望休を調整納期表にコピー

    worker_list:list = []
    for index, row in due_date_df.iterrows():
        workers: pd.DataFrame = skill_df.loc[:, (["スキルグループ"], [row["スキルグループ"]])] #スキルグループ→工程関係ない
        
        worker = workers.dropna().sample().values[0, 0]

        worker_list.append(worker)
        worker_off_days: pd.DataFrame = off_day_df.loc[off_day_df["設備"].isin([worker]), off_day_df.columns[2:]]                   # 作業者の希望休を抽出

        due_date_df.loc[[index], due_date_df.columns[6:]] = worker_off_days.values[0]
        adjusted_due_date_df[start_date]=1

        # '納期' 列が存在することを前提としています
        due_date_column = '納期'
        # 各行の日付列に対応する場所に '3' を挿入
        for index, row in adjusted_due_date_df.iterrows():
            due_date = row[due_date_column]
            # 日付列がすべての列に存在するか確認
            if due_date in row.index:
                adjusted_due_date_df.at[index, due_date] = 3
    adjusted_due_date_df.insert(3, "作業者", worker_list)
    adjusted_due_date_dfs=adjusted_due_date_df

    process_num31=adjusted_due_date_df.iloc[:,7:]
    process_num32 = process_num31.apply(move_three, axis=1)
    adjusted_due_date_df.iloc[:,7:]=process_num32
    adjusted_due_date_df[start_date]=0
    return adjusted_due_date_dfs,worker_list,process_num32

   
#45変換
def pre_evalution(adjusted_due_date_df):

    adjusted_due_date_df2=adjusted_due_date_df.iloc[:,7:] #スキルグループ→工程による列追加   
    return adjusted_due_date_df2


#46ssd
def squared(adjusted_due_date_dfs,due_num: pd.DataFrame):#adjusted_due_date_df??

    combined_dfs0=adjusted_due_date_dfs.iloc[:,3:]

    combined_dfs=combined_dfs0.drop(['スキルグループ','治具','納期'],axis=1)

    #変更
    # 作業者をインデックスに、日付を列に設定
    df3 = combined_dfs.set_index('作業者')
    # 日付ごとに数値 '3' がいくつあるかカウント
    df_3_count = (df3 == 3).astype(int)
    # 作業者ごとに数値 '3' を集計
    count_df = df_3_count.groupby(df3.index).sum()
    count_df=count_df.reset_index() 
    df=count_df
    df = df.apply(pd.to_numeric, errors='coerce')
    # 条件に基づいた二乗和の計算
    squared_differences = np.where(df > due_num, (df - due_num) ** 2, 0)
    squared_differences_df = pd.DataFrame(squared_differences, index=df.index, columns=df.columns)
    ssd = squared_differences_df.sum().sum()
    squared_differences_df.drop(labels='作業者', axis=1,inplace=True)
    squared_differences_df.index = count_df.index

    return ssd,count_df

def evalution_function(adjusted_due_date_df2,jigu_date,start_date,due_date_df: datetime.datetime,adjusted_due_date_df,due_num):

    # (3)前倒し生産方式の場合、できるだけ開始日に近いこと、ジャストインタイム方式の場合、できるだけ納期に近いこと。
    # 秒換算
    # ’×－1
    score3=[]  
    fix_day3= []
    tdos_list3=[]
    adjusted_due_date_df_copy= adjusted_due_date_df2.copy()
    adjusted_due_date_df_copy.reset_index(inplace=True,drop=True)
    #行と列の数を取り出す
    sh3=adjusted_due_date_df_copy.shape
    adjusted_due_date_df_copy.columns=range(sh3[1])
    for k in range(len(adjusted_due_date_df_copy)):
        for v in range(len(adjusted_due_date_df_copy.columns)):
            if adjusted_due_date_df_copy.iloc[k,v]==3:

                fix_day3.append(adjusted_due_date_df.columns[v+7])

    tdos3= '0000-00-00 00:00:00'
    tm3 : datetime.datetime
    
    for tm3 in fix_day3:
        ttdelta3= tm3 - start_date
        tdos3= ttdelta3.total_seconds()
        tdos3=int(tdos3)
        tdos_list3.append(tdos3)
    #強化1→1000
    score3=sum(tdos_list3)*(-1000)


#追加
##(4)工程ができるだけ長くならないこと
#工程間1日ずれ（品番＝＝品番）len1=（日程（工程ｎ＋１）-（日程（工程ｎ）>0なら高得点
    # ’×－1 
    score4=[]  
    fix_day41= []
    fix_day42= []
    tdos_list4=[]
    serial_number = due_date_df.loc[:, "製造番号"].max()
    # 製造番号ごとに繰り返す
    for i in range(1, serial_number + 1):        
        during_process_df = adjusted_due_date_df[adjusted_due_date_df["製造番号"] == i]
        process_max = during_process_df.loc[:, "工程"].max()
        if process_max >1:
            for j in range(2, process_max):
                pds2:datetime= during_process_df[during_process_df["工程"]== j].iloc[0,7:]
                pds20=pds2[pds2.isin([3])]
                if "工程" in pds20:
                    pass
                else:
                    pds20=pds20.index.values[0]
                    fix_day42.append(pds20)
                    tm42 : datetime.datetime    
                    for tm42 in fix_day42:
                        pass
                    pds1:pd.Series= during_process_df[during_process_df["工程"]== j-1].iloc[0,7:]
                    pds10=pds1[pds1.isin([3])]
                    pds10=pds10.index.values[0]
                    fix_day41.append(pds10)
                    tm41 : datetime.datetime
                    for tm41 in fix_day41:
                        ttdelta4= tm42 - tm41
                        tdos4= ttdelta4.total_seconds()
                        tdos4=int(tdos4)
                        #追加
                        tdos4= tdos4- 86400
                        if tdos4== 0:
                            tdos4= tdos4- 100000#変更-100000->-1000000
                        elif tdos4< 0:
                            tdos4=tdos4+ 100000#変更+100000->+1000000
                        elif tdos4== 86400:
                            tdos4= tdos4-50000
                        tdos_list4.append(tdos4)                
        elif process_max ==1 :
            pass
           #強化1
        score4=sum(tdos_list4)*(-100)#変更+1000->

#追加
##(5)1日8工程まで
    score5=0
    tdos_list5=[]
    y=adjusted_due_date_df_copy[adjusted_due_date_df_copy==3].count()
    y_list = y.values.tolist()
    for i in range(len(y_list)):
        if y_list[i] >=due_num:
            tdos_list5.append(y_list[i])
    score5=sum(tdos_list5)*(-10000000)#変更-1000000->

#追加
##(6)同じ治具を使う工程が同日ならボーナス
    score6=0
    tdos_list6=[]
    jigu_date2= jigu_date.copy()
    jigu_date2['行番号']= 0
    jigu_date3= jigu_date.copy()  
    # 条件を満たす値のインデックスを取得
    rows, cols = np.where(adjusted_due_date_df_copy== 3)
    # 条件を満たすカラム名を取得
    selected_columns = adjusted_due_date_df_copy.columns[cols].tolist()
    # 特定の値を探す
    value_to_find = 3
# 条件を満たす行と列のインデックスを取得
    rows, cols = np.where(adjusted_due_date_df_copy == value_to_find)
# 条件を満たす行と列の名前を取得
    for row, col in zip(rows, cols):
        column_name = adjusted_due_date_df_copy.columns[col]
        row_data = adjusted_due_date_df_copy.iloc[row]
    variety_df=adjusted_due_date_df.iloc[:,:6]
    adjusted_due_date_df3=pd.concat([variety_df,adjusted_due_date_df2],axis=1)
    adjusted_due_date_df3.insert(0,'fix_day3',fix_day3)
    dup=adjusted_due_date_df3.duplicated(subset=['fix_day3','治具'])
    DUP= dup.sum()
    score6=DUP*(100000)
  
#追加
##(7)全てのマスが均等
    score7=0
    ssd ,count_df= squared(adjusted_due_date_df,due_num)
    score7=ssd*(-1000000000)#変更10000000->

    score=  score3 + score4 + score5 + score6 +score7
    print('total_score',score)
    return score,count_df

#@st.cache_data



init()
