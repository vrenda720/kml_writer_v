import csv

class kml_writer:

    def __init__(self, outpath):
        self.outfilehandle = open(outpath, 'w')
        self.outfilehandle.write(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        )
        self.outfilehandle.write(
            "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n"
        )
        self.outfilehandle.write(
            "\t<Document>\n"
            "\t\t<name></name>\n"
            "\t\t<description></description>\n"
            "\t\t<Style id=\"path_style\">\n"
            "\t\t\t<LineStyle>\n"
            "\t\t\t\t<color>ff00ffff</color>\n"
            "\t\t\t\t<width>5</width>\n"
            "\t\t\t</LineStyle>\n"
            "\t\t\t<PolyStyle>\n"
            "\t\t\t\t<color>ff00ff00</color>\n"
            "\t\t\t</PolyStyle>\n"
            "\t\t</Style>\n"
        )

    def addLine(self, lon1, lat1, lon2, lat2):
        next_line = ("\t\t<Placemark>\n"
            "\t\t\t<name></name>\n"
            "\t\t\t<description></description>\n"
            "\t\t\t<styleUrl>#path_style</styleUrl>\n"
            "\t\t\t<LineString>\n"
            "\t\t\t\t<extrude>0</extrude>\n"
            "\t\t\t\t<tesselate>1</tesselate>\n"
            "\t\t\t\t<altitudeMode>clampToGround</altitudeMode>\n"
            "\t\t\t\t<coordinates>\n")
        self.outfilehandle.write(next_line)

        coords = f"\t\t\t\t\t{lon1},{lat1},0\n\t\t\t\t\t{lon2},{lat2},0\n"
        self.outfilehandle.write(coords)

        ender = ("\t\t\t\t</coordinates>\n"
                  "\t\t\t</LineString>\n"
                  "\t\t</Placemark>\n")
        self.outfilehandle.write(ender)

    def finish(self):
        self.outfilehandle.write(
            "\t</Document>\n"
            "</kml>")
        
# test = kml_writer("testing123.kml")
# test.addLine(1,2,3,4)
# test.addLine(5,6,7,8)
# test.finish()