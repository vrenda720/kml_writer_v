from projections import opk_to_ypr, ypr_to_opk
import json

from dir_crawler import dir_crawler

# files = dir_crawler(r'C:\msys64\home\didi\kml_writer','.json',None)
# jsons = files.traverse()
# for json in jsons:
#     print(json)

# Opening JSON file
# f = open(jsons[0])
f = open(r'D:\240325-siteRTX0002-delivery\siteRTX002-SIT-sampus\siteRTX0002-SIT-campus\camA001-ZED2-2K-1\2024-01-31-13-14-31\siteRTX0002-camA001-2024-01-31-13-14-31-00000233.json')

# returns JSON object as 
# a dictionary
data = json.load(f)

# Iterating through the json
# list
lat = data['extrinsics']['lat']
lon = data['extrinsics']['lon']
alt = data['extrinsics']['alt']
omega = data['extrinsics']['omega']
phi = data['extrinsics']['phi']
kappa = data['extrinsics']['kappa']
# lats = []

# Closing file
f.close()


yaw, pitch, roll = opk_to_ypr(lat, lon, alt, omega, phi, kappa)
# print(pitch, roll, yaw)
print ('pitch = ', pitch)
print ('roll = ', roll)
print ('yaw = ', yaw, '\n')
omega, phi, kappa = ypr_to_opk(lat, lon, alt, yaw, pitch, roll)
# print(omega, phi, kappa)
print ('omega = ', omega)
print ('phi = ', phi)
print ('kappa = ', kappa)

from RPY2OPK import rph2opk
print("----------------")
omega, phi, kappa = rph2opk(roll, pitch, yaw)
# print(omega, phi, kappa)
print ('omega = ', omega)
print ('phi = ', phi)
print ('kappa = ', kappa)

