import pandas as pd
import pandasql as ps
import joblib


def get_patient_infos(df, patient_id):
    query_ex1 = """ 
                SELECT 
                    patient_id, 
                    round('2021-04-01' -  max(donation_date), 0) AS months_since_last_donation, 
                    count(*) AS number_of_donations, 
                    sum(volume_donated_cc) AS total_volume_donated_cc, 
                    round('2021-04-01' - min(donation_date),0) AS months_since_first_donation
                FROM df
                WHERE patient_id  = {}
                GROUP BY patient_id
            """.format(patient_id)
    return ps.sqldf(query_ex1, locals())




if __name__ == '__main__':
    
    dict_filename = '../data/blood_donation_hist.csv'
    model_filename = '../data/blood_donation_model.joblib'
    
    df = pd.read_csv(dict_filename)
    df['donation_date'] =  pd.to_datetime(df['donation_date'], format='%Y-%m-%d')
    df = get_patient_infos(df, 0)
    df = df.astype(int)
    
    
    model = joblib.load(model_filename) 
    pred_cols = list(df.columns.values)[:-1]
    pred = pd.Series(model.predict(df[pred_cols]))
    df['prediction'] = pred
    df = df.to_dict('records')
    print(df)