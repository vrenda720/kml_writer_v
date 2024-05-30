from newpoint import calculate_new_point
from kmlwriter_direction_paths import kml_writer
import json
from dir_crawler import dir_crawler
import os
import sys
# import kml_writer, view_edit
import argparse
# import pdb

parser = argparse.ArgumentParser(
    prog="Folder to KML",
    description="Turns folder of jsons into a KML file with a path that shows the direction each point is looking in"
)

parser.add_argument('-i',
                    '--input',
                    help="Path to metadata folder",
                    metavar="",
                    action="store")

parser.add_argument('-o',
                    '--outfile',
                    help="Name of output file (include .kml at the end. File will be in same folder as this script)",
                    metavar="",
                    action="store")

args = parser.parse_args()

# files = dir_crawler(r'C:\msys64\home\didi\kml_writer\stevens_challenge_dataset_4\stevens_challenge_dataset_4\reference\ta2_metadata','.json',None)
# files = dir_crawler(r'D:\240325-siteRTX0002-delivery\siteRTX002-SIT-sampus\siteRTX0002-SIT-campus\camA001-ZED2-2K-1\2024-01-31-13-14-31', '.json', None)
files = dir_crawler(args.input, '.json', None)
jsons = files.traverse()

out = kml_writer(args.outfile, os.path.basename(os.path.dirname(jsons[0])))

for jsun in jsons:
    f = open(jsun)
    data = json.load(f)
    lon1 = data['extrinsics']['lon']
    lat1 = data['extrinsics']['lat']
    kap = data['extrinsics']['kappa']
    f.close()
    lat2, lon2 = calculate_new_point(lat1, lon1, 0.02, kap)
    out.addLine(lon1, lat1, lon2, lat2, os.path.basename(jsun))

out.finish()