import pandas as pd
# import urllib
import requests

def filter_url(list):
    if len(list) > 1:
        return None
    for i in list:
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
    urls = df.url.tolist()
    for i in range(len(urls)):
        if i % 1000 == 0:
            print(i, count) 
            download_df = df.dropna()
            download_df.to_csv('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/img/small_batch.csv', index=False, encoding = 'utf-8')
        try:
            r = requests.get(urls[i], stream=True)
            if r.status_code == 200:
                open('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/img/small_batch/'+ str(i) + '.jpg', 'wb').write(r.content) 
                # urllib.urlretrieve(urls[i], '/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/img/small_batch/'+ str(i) + '.jpg')
                df.iloc[i,3] = str(i)+ '.jpg'
                count += 1
        except:
            continue
    download_df = df.dropna()
    download_df.to_csv('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/img/small_batch.csv', index=False, encoding = 'utf-8')
            
    pass

if __name__ == "__main__":
    #remove files not are not jpg or png, and also remain data with only one image per tweet. 
    df = pd.read_json (r'/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/decahose/twitterEngImg.json', lines=True)
    df['url'] = df.url.apply(lambda x: filter_url(x))
    df = df.dropna()
    df['path'] = df.url.apply(lambda x: None)
    download_img(df)
    

    

