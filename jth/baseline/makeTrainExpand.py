import os
import csv
train_dir = '/opt/ml/input/data/train'

def makeTrainExpandGender(train_dir):
    with open(os.path.join(train_dir, 'train.csv'), 'r') as csvinput:
        with open(os.path.join(train_dir, 'train_expand_gender.csv'), 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append('file')
            row.append('target_gender')
            all.append(row)

            for row in reader:
                for filename in os.listdir(os.path.join(train_dir, 'images/' + row[-1])):
                    if '._' not in filename:
                        _row = row[:]
                        _row.append(filename)
                        if _row[1] == 'male':
                            _row.append(0)
                        else:
                            _row.append(1)

                        all.append(_row)    
            writer.writerows(all)

def makeTrainExpandMask(train_dir):
    with open(os.path.join(train_dir, 'train.csv'), 'r') as csvinput:
        with open(os.path.join(train_dir, 'train_expand_mask.csv'), 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append('file')
            row.append('target_mask')
            all.append(row)

            for row in reader:
                for filename in os.listdir(os.path.join(train_dir, 'images/' + row[-1])):
                    if '._' not in filename:
                        _row = row[:]
                        _row.append(filename)
                        if 'normal' in filename:
                            _row.append(0)
                        elif 'incorrect' in filename:
                            _row.append(1)
                        else:
                            _row.append(2)

                        all.append(_row)    
            writer.writerows(all)

def makeTrainExpandAge(train_dir):
    with open(os.path.join(train_dir, 'train.csv'), 'r') as csvinput:
        with open(os.path.join(train_dir, 'train_expand_age.csv'), 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append('file')
            row.append('target_age')
            all.append(row)

            for row in reader:
                for filename in os.listdir(os.path.join(train_dir, 'images/' + row[-1])):
                    if '._' not in filename:
                        _row = row[:]
                        _row.append(filename)
                        if int(row[3]) < 30:
                            _row.append(0)
                        elif int(row[3]) >= 30 and int(row[3]) < 60:
                            _row.append(1)
                        else:
                            _row.append(2)

                        all.append(_row)    
            writer.writerows(all)

def makeExpandFiles():
    makeTrainExpandGender(train_dir)
    makeTrainExpandMask(train_dir)
    makeTrainExpandAge(train_dir)