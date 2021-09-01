import pandas as pd
import numpy as np
import os

OUTPUT_DIR = '/opt/ml/image-classification-level1-29/team/output'

def merge():
    data1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'output_mask.csv'))
    data2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'output_gender.csv'))
    data3 = pd.read_csv(os.path.join(OUTPUT_DIR, 'output_age.csv'))
    
    output2 = pd.merge(data1, data2,
                    on='ImageID', 
                    how='left',
                    suffixes=('_mask', '_gender'))

    output3 = pd.merge(output2, data3,
                    on='ImageID', 
                    how='left')

    df = pd.DataFrame(output3)
    conditions = [
        (df['ans_mask'] == 0) & (df['ans_gender'] == 0) & (df['ans'] == 0),
        (df['ans_mask'] == 0) & (df['ans_gender'] == 0) & (df['ans'] == 1),
        (df['ans_mask'] == 0) & (df['ans_gender'] == 0) & (df['ans'] == 2),
        (df['ans_mask'] == 0) & (df['ans_gender'] == 1) & (df['ans'] == 0),
        (df['ans_mask'] == 0) & (df['ans_gender'] == 1) & (df['ans'] == 1),
        (df['ans_mask'] == 0) & (df['ans_gender'] == 1) & (df['ans'] == 2),
        (df['ans_mask'] == 1) & (df['ans_gender'] == 0) & (df['ans'] == 0),
        (df['ans_mask'] == 1) & (df['ans_gender'] == 0) & (df['ans'] == 1),
        (df['ans_mask'] == 1) & (df['ans_gender'] == 0) & (df['ans'] == 2),
        (df['ans_mask'] == 1) & (df['ans_gender'] == 1) & (df['ans'] == 0),
        (df['ans_mask'] == 1) & (df['ans_gender'] == 1) & (df['ans'] == 1),
        (df['ans_mask'] == 1) & (df['ans_gender'] == 1) & (df['ans'] == 2),
        (df['ans_mask'] == 2) & (df['ans_gender'] == 0) & (df['ans'] == 0),
        (df['ans_mask'] == 2) & (df['ans_gender'] == 0) & (df['ans'] == 1),
        (df['ans_mask'] == 2) & (df['ans_gender'] == 0) & (df['ans'] == 2),
        (df['ans_mask'] == 2) & (df['ans_gender'] == 1) & (df['ans'] == 0),
        (df['ans_mask'] == 2) & (df['ans_gender'] == 1) & (df['ans'] == 1),
        (df['ans_mask'] == 2) & (df['ans_gender'] == 1) & (df['ans'] == 2)
    ]
    choices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    df['ans'] = np.select(conditions, choices)

    df = df.drop(columns=['ans_mask', 'ans_gender'], axis=1)
    df.set_index('ImageID', inplace=True)

    df.to_csv(os.path.join(OUTPUT_DIR, 'output_join.csv'))
