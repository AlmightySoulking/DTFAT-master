import pandas as pd
import csv
import json
import numpy as np
df = pd.read_csv('/home/almighty/Machine learning Codes/DTFAT-master/egs/audioset/data/train_metaData.csv')
#print(df)
# {"data": [{"video_id":,"wav":,"labels":}]}

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['file_name']] = int(row['index'])
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['class_name']
            line_count += 1
    return name_lookup

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def lookup_list(label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in list(table.keys()):
        label_list.append(table[item])
    return label_list

def create_json_file(label_csv):
    # {"data": [{"video_id":,"wav":,"labels":}]}
    AVSpoofDataset = {}
    AVSpoofDataset["data"] = []
    for idx,name in enumerate(lookup_list('/home/almighty/Machine learning Codes/DTFAT-master/egs/audioset/data/train_metaData.csv')):
        AVSpoofDataset["data"].append({"wav": df['file_path'][idx],
                                       "label":df['target'][idx]})
        
    return AVSpoofDataset

if __name__=='__main__':
    #print(make_index_dict('/home/almighty/Machine learning Codes/DTFAT-master/egs/audioset/data/train_metaData.csv'))
    #print(lookup_list('/home/almighty/Machine learning Codes/DTFAT-master/egs/audioset/data/train_metaData.csv'))
    with open('/home/almighty/Machine learning Codes/DTFAT-master/egs/audioset/data/AVSpoofDataset.json','w') as json_file:
        json.dump(create_json_file('/home/almighty/Machine learning Codes/DTFAT-master/egs/audioset/data/train_metaData.csv'),json_file, indent=4, cls = NpEncoder)