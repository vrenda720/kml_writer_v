from newpoint import calculate_new_point
from kmlwriter_direction_paths import kml_writer
import json
from dir_crawler import dir_crawler
import os
import argparse

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