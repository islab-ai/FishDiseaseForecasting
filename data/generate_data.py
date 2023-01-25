import json
import pandas as pd
import numpy as np
import os.path

current_path = os.getcwd()
data_path = current_path + "/dataset/"
json_daily_path = data_path + "Fish Disease Daily Report.json"
json_sensor_path = data_path + "sensor_data.json"

def create_save_directory():

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

def json_load(json_file_path):
    
    with open(json_file_path, 'r', encoding='utf-8') as j:
        json_file = json.loads(j.read())
        
    return json_file

def make_daily_dataFrame():
    
    daily = json_load(json_daily_path)
    df_daily = pd.DataFrame(daily['daily_report'])
    
    # df_daily에서 'disease'에 질병이 있는 데이터들만 사용, 질병이 없으면 버림
    mask = []
    for i in df_daily['disease']:
        if len(i) > 0:
            mask.append(True)
        else:
            mask.append(False)
    df_daily = df_daily[mask]
    
    # 'farmid'를 'tanknum'으로 name 변경
    df_daily = df_daily.rename(columns={'farmid' : 'tanknum'})
    
    return df_daily

def make_sensor_dataFrame():
    
    sensor = json_load(json_sensor_path)
    df_sensor = pd.DataFrame(sensor['sensor_data'])
    
    # 'datetime'에서 00:00:00(시간)을 제외한 날짜 정보만 가져오기
    split_date = []
    for i in df_sensor['datetime']:
        a = i.split(" ")[0]
        split_date.append(a)
    
    # 날짜 정보만 기록된 컬럼 추가
    df_sensor['report_date'] = split_date
    
    return df_sensor
    
def generate_preprocessed_data():
    same_date = []
    mask = []
    feed_zero = []
    feed_mp = []
    feed_ep = []
    no_medicine = []
    medicine_ceftiofur = []
    medicine_amoxicillin = []
    medicine_formaldehyde = []
    
    df_daily = make_daily_dataFrame()
    df_sensor = make_sensor_dataFrame()
    
    # df_sensor['report_date']와 df_daily['report_date']에 공통된 날짜 가져오기 
    sensor_date = df_sensor['report_date'].unique()
    daily_date = df_daily['report_date'].unique()
    
    for i in sensor_date:
        if i in daily_date:
            same_date.append(i)
    
    for j in df_daily['report_date']:
        if j in same_date:
            mask.append(True)
        else:
            mask.append(False)
    df_daily = df_daily[mask]
    
    df_sensor = df_sensor.drop(['datetime'], axis=1)
    
    # 같은 날, 같은 tanknum 기준으로 val_tp, val_do, val_ph, val_orp, val_sl의 평균 계산
    df_sensor_mean = df_sensor.groupby(['tanknum', 'report_date'], as_index=False).mean()
    
    # df_daily에서 시계열 학습에서 필요없는 col 제거
    df_daily = df_daily.drop(['tank_name', 'medicine_date', 'disease_fish'], axis=1)
    
    # 데이터 합치기
    all_df = pd.merge(left=df_sensor_mean, right=df_daily, how='left', on=['tanknum', 'report_date'])
    all_df = all_df.rename(columns={'report_date':'date'})
    all_df = all_df[['date', 'tanknum', 'val_tp', 'val_do', 'val_ph', 'val_orp', 'val_sl', 
                 'total_qty', 'dead_qty', 'feed_name', 'feed_qty', 
                 'medicine_name', 'medicine_qty', 'disease']]
    
    # 'disease'에 값이 있으면 1, NaN이면 0로 라벨링 + col추가
    all_df['disease_label'] = pd.notna(all_df['disease']).astype(int)
    all_df = all_df.drop(['disease'], axis=1)
    all_df = all_df.fillna(0)
    
    # 빈 값이 있는 col에 대해 빈 값을 0으로 대체 및 값 단위 맞추기
    all_df = all_df.replace({'feed_name' : ''}, 0)
    all_df = all_df.replace({'medicine_name' : ''}, 0)
    all_df = all_df.replace({'medicine_name' : '      '}, 0)
    all_df = all_df.replace({'medicine_qty' : ''}, 0)
    all_df = all_df.replace({'medicine_qty' : '200'}, '0.2')
    
    # 'feed_name' 및 'medicine_name'에서 종류별 col 나누기
    for i in all_df['feed_name']:
        if i == 0:
            feed_zero.append(1)
        else:
            feed_zero.append(0)
        if i == '생사료(MP)':
            feed_mp.append(1)
        else:
            feed_mp.append(0)
        if i == '배합사료(EP)':
            feed_ep.append(1)
        else:
            feed_ep.append(0)
    
    for i in all_df['medicine_name']:
        if i == 0:
            no_medicine.append(1)
        else:
            no_medicine.append(0)
        if i == '세프티오퍼':
            medicine_ceftiofur.append(1)
        else:
            medicine_ceftiofur.append(0)
        if i == '아목사실린':
            medicine_amoxicillin.append(1)
        else:
            medicine_amoxicillin.append(0)
        if i == '포르말린':
            medicine_formaldehyde.append(1)
        else:
            medicine_formaldehyde.append(0)
            
    all_df['no_feed'] = feed_zero
    all_df['feed_MP'] = feed_mp
    all_df['feed_EP'] = feed_ep
    all_df['no_medicine'] = no_medicine
    all_df['medi_ceftiofur'] = medicine_ceftiofur
    all_df['medi_amoxicillin'] = medicine_amoxicillin
    all_df['medi_formaldehyde'] = medicine_formaldehyde
    
    all_df = all_df.drop(['feed_name', 'medicine_name'], axis=1)
    all_df = all_df[['date', 'tanknum', 'val_tp', 'val_do', 'val_ph', 'val_orp', 'val_sl',
       'total_qty', 'dead_qty', 'no_feed', 'feed_MP', 'feed_EP', 'feed_qty','no_medicine', 'medi_ceftiofur',
       'medi_amoxicillin', 'medi_formaldehyde',
       'medicine_qty', 'disease_label']]

    # 삭제할 columns : val_tp, val_do, val_orp, feed_EP, medi_formaldehyde, medi_amoxicillin
    all_df = all_df.drop(['val_tp'], axis=1)
    all_df = all_df.drop(['val_do'], axis=1)
    all_df = all_df.drop(['val_orp'], axis=1)
    all_df = all_df.drop(['feed_EP'], axis=1)
    all_df = all_df.drop(['medi_formaldehyde'], axis=1)
    all_df = all_df.drop(['medi_amoxicillin'], axis=1)
    
    # 날짜순으로 정렬
    all_df = all_df.sort_values(by='date')
    
    # csv 파일로 저장 
    all_df.to_csv(data_path + 'TS_Flatfish.csv', index=False, encoding='utf-8')

if __name__ == '__main__':

    create_save_directory()
    file_list = os.listdir(data_path)
    try:
        file_list.remove(".ipynb_checkpoints")
        print("파일 리스트에서 .ipynb_checkpoints를 삭제합니다.")
    except:
        print("===================")
    generate_preprocessed_data()
    print("====================================")
    print("데이터 생성 완료")