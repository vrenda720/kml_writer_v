# Python program to read
# json file

import json

from dir_crawler import dir_crawler

files = dir_crawler(r'C:\msys64\home\didi\kml_writer','.json',None)
jsons = files.traverse()
# for json in jsons:
#     print(json)

# Opening JSON file
f = open(jsons[0])

# returns JSON object as 
# a dictionary
data = json.load(f)

# Iterating through the json
# list
i = data['extrinsics']['lat']
print(i)
# lats = []

# Closing file
f.close()

# for i in range(7):
#     f = open(r'ta2_metadata\image000000' + str(i) + r'1.json')
#     data = json.load(f)
#     i = data['extrinsics']['lat']
#     # print(i)
#     lats.append(i)
#     f.close()

# print(lats)
