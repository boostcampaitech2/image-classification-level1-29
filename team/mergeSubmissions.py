import pandas as pd
import numpy as np
import os

OUTPUT_DIR = '/opt/ml/image-classification-level1-29/output'

def merge():
    data1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'output_gender.csv'))
    data2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'output_mask.csv'))
    data3 = pd.read_csv(os.path.join(OUTPUT_DIR, 'output_age.csv'))
    
    output2 = pd.merge(data1, data2,
                    on='ImageID', 
                    how='left')

    output3 = pd.merge(output2, data3,
                    on='ImageID', 
                    how='left')

    df = pd.DataFrame(output3)
    conditions = [
        (df['ans_x'] == 0) & (df['ans_y'] == 0) & (df['ans'] == 0),
        (df['ans_x'] == 0) & (df['ans_y'] == 0) & (df['ans'] == 1),
        (df['ans_x'] == 0) & (df['ans_y'] == 0) & (df['ans'] == 2),
        (df['ans_x'] == 1) & (df['ans_y'] == 0) & (df['ans'] == 0),
        (df['ans_x'] == 1) & (df['ans_y'] == 0) & (df['ans'] == 1),
        (df['ans_x'] == 1) & (df['ans_y'] == 0) & (df['ans'] == 2),
        (df['ans_x'] == 0) & (df['ans_y'] == 1) & (df['ans'] == 0),
        (df['ans_x'] == 0) & (df['ans_y'] == 1) & (df['ans'] == 1),
        (df['ans_x'] == 0) & (df['ans_y'] == 1) & (df['ans'] == 2),
        (df['ans_x'] == 1) & (df['ans_y'] == 1) & (df['ans'] == 0),
        (df['ans_x'] == 1) & (df['ans_y'] == 1) & (df['ans'] == 1),
        (df['ans_x'] == 1) & (df['ans_y'] == 1) & (df['ans'] == 2),
        (df['ans_x'] == 0) & (df['ans_y'] == 2) & (df['ans'] == 0),
        (df['ans_x'] == 0) & (df['ans_y'] == 2) & (df['ans'] == 1),
        (df['ans_x'] == 0) & (df['ans_y'] == 2) & (df['ans'] == 2),
        (df['ans_x'] == 1) & (df['ans_y'] == 2) & (df['ans'] == 0),
        (df['ans_x'] == 1) & (df['ans_y'] == 2) & (df['ans'] == 1),
        (df['ans_x'] == 1) & (df['ans_y'] == 2) & (df['ans'] == 2)
    ]
    choices = ['12', '13', '14', '15', '16', '17', '6', '7', '8', '9', '10', '11', '0', '1', '2', '3', '4', '5']
    df['ans'] = np.select(conditions, choices)
    del df['ans_x']
    del df['ans_y']
    df.to_csv(os.path.join(OUTPUT_DIR, 'output_join.csv'))