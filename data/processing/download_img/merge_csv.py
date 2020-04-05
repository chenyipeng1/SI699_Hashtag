import pandas as pd
import json
# should also change path in order to run.

with open('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/ImageList.txt', 'r') as filehandle:
    img_list = json.load(filehandle)
dictOfImg = dict.fromkeys(img_list , 1)

with open('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/FreqDict.txt', 'r') as filehandle:
    freq_dict = json.load(filehandle)

def filter_url(i):
    if i[-3:] == 'jpg':
        return i
    else:
        return None

def find_img(row):
    index = int(row.name)
    folder_num = str(100000 * (1 + (index // 100000))) + '/'
    if str(index)+'.jpg' in dictOfImg:
        return folder_num + str(index)+'.jpg'
    else:
        return None

def filter_tag(l):
    tmp = []
    for i in l:
        if i in freq_dict:
            tmp.append(i)
    if tmp == []:
        return None
    return tmp


if __name__ == "__main__":
    df = pd.read_json (r'/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/OneMonth10033.json', lines=True)

    df['image'] = df.image.apply(lambda x: filter_url(x))
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)

    df['path'] = df.image.apply(lambda x: None)
    df['path'] = df.apply(lambda x: find_img(x), axis=1)
    df = df.dropna()

    df['hashtags'] =  df.hashtags.apply(lambda x: filter_tag(x))
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.head())
    print(df.tail())
    print(df.shape)
    df.to_csv('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/OneMonthFilter002.csv', index=False, encoding = 'utf-8', mode = 'a')
