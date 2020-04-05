import json
import os

folder_list = [100000 * i for i in range(1,19)]

filePath = '/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/Image/10033/'

list = []

for i in folder_list:
	path = filePath + str(i)
	list += os.listdir(path)
	print(len(list))

with open('/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/ImageList.txt', 'w') as filehandle:
    json.dump(list, filehandle)
