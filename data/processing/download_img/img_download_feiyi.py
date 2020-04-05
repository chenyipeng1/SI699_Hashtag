import pandas as pd
# import urllib
import requests

'''
###### change 'USER' in LINE 15 to your own name #####
###### change range in LINE 37 #####
There are total 1725884 images.
chenyp     range(0,50w)
feiyi      range(50w, 90w)
wowwh      range(90w, 130w)
yuyingli   range(130w, 1725884)
'''

USER = 'feiyi'


def filter_url(i):
    if i[-3:] == 'jpg':
        return i
    else:
        return None

# def filter_non_eng_tag(list):
#     tmp = []
#     for i in list:
#         try:
#             if detect(i) == 'en':
#                 tmp += [i]
#         except:
#             continue
#     return tmp

def download_img(df):
    count = 0
    urls = df.image.tolist()
    for i in range(800000,900000):
        if i % 1000 == 0:
            print(i, count) 
            # download_df = df.dropna()
            # download_df.to_csv('/nfs/locker/arcts-cavium-hadoop-stage/home/si699w20_cbudak_JI_team/data/dataset/OneMonthData/Image/' + '10033_' + USER + '.csv', index=False, encoding = 'utf-8', mode = 'a')

        folder_num = str(100000 * (1 + (i / 100000))) + '/'
        try:
            r = requests.get(urls[i], stream=True)
            if r.status_code == 200:
                open('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/Image/10033/'+ folder_num + str(i) + '.jpg', 'wb').write(r.content) 
                # urllib.urlretrieve(urls[i], '/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/img/small_batch/'+ str(i) + '.jpg')
                # df.iloc[i,3] = folder_num + str(i)+ '.jpg'
                count += 1
        except:
            continue
    # download_df = df.dropna()
    # download_df.to_csv('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/Image/' + '10033_' + USER + '.csv', index=False, encoding = 'utf-8', mode = 'a')
            
    pass

if __name__ == "__main__":
    #remove files not are not jpg or png, and also remain data with only one image per tweet. 
    df = pd.read_json (r'/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/OneMonth10033.json', lines=True)
    df['image'] = df.image.apply(lambda x: filter_url(x))
    df = df.dropna()
    df['path'] = df.image.apply(lambda x: None)
    print('df shape:', df.shape)
    download_img(df)
    

    

