from newpoint import calculate_new_point
from kmlwriter_v import kml_writer
import json
from dir_crawler import dir_crawler

files = dir_crawler(r'C:\msys64\home\didi\kml_writer\stevens_challenge_dataset_4\stevens_challenge_dataset_4\reference\ta2_metadata','.json',None)
jsons = files.traverse()
# f = open(jsons[i])
# data = json.load(f)
# lon1 = data['extrinsics']['lon']
# lat1 = data['extrinsics']['lat']
# kap = data['extrinsics']['kappa']
# f.close()

# lat2, lon2 = calculate_new_point(lat1, lon1, 0.02, kap)

# # kml goes lon, lat, alt

# print(lon1)
# print(lat1)
# print(kap)
# print(lon2)
# print(lat2)

# out = kml_writer("kmltestcombo.kml")

out = kml_writer("kmltestcombo.kml")

for jsun in jsons:
    f = open(jsun)
    data = json.load(f)
    lon1 = data['extrinsics']['lon']
    lat1 = data['extrinsics']['lat']
    kap = data['extrinsics']['kappa']
    f.close()
    lat2, lon2 = calculate_new_point(lat1, lon1, 0.02, kap)
    out.addLine(lon1, lat1, lon2, lat2)

out.finish()