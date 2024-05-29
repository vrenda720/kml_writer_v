from dir_crawler import dir_crawler

files = dir_crawler('C:\msys64\home\didi\kml_writer','.json',None)
jsons = files.traverse()
for json in jsons:
    print(json)