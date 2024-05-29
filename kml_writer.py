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

    def make_path(self, inpath, **kwargs):

        ### Get Data ###
        with open(inpath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            #time_table = [row["Measurement_DateTime"] for row in reader]
            coord_table = [[row["GPS_lon"], row["GPS_lat"]] for row in reader]

            #for row in reader:
            #    time_table.append(row["Measurement_DateTime"])
            #   coord_table.append([row["GPS_lon"], row["GPS_lat"]])

        ##### Header #####

        params = {
            "document_name":"",
            "document_desc":"",
            "line_color":"ff00ffff",
            "line_width":"5",
            "poly_color":"ff00ff00",
            "path_name":"",
            "path_desc":"",
            "extrude":"0",
            "tesselate":"1",
            "alt_mode":"clampToGround"
        }

        for key in kwargs.keys():
            if key in params.keys():
                params[key] = kwargs[key]

        header = ("\t<Document>\n"
            "\t\t<name>{}</name>\n"
            "\t\t<description>{}</description>\n"
            "\t\t<Style id=\"path_style\">\n"
            "\t\t\t<LineStyle>\n"
            "\t\t\t\t<color>{}</color>\n"
            "\t\t\t\t<width>{}</width>\n"
            "\t\t\t</LineStyle>\n"
            "\t\t\t<PolyStyle>\n"
            "\t\t\t\t<color>{}</color>\n"
            "\t\t\t</PolyStyle>\n"
            "\t\t</Style>\n"
            "\t\t<Placemark>\n"
            "\t\t\t<name>{}</name>\n"
            "\t\t\t<description>{}</description>\n"
            "\t\t\t<styleUrl>#path_style</styleUrl>\n"
            "\t\t\t<LineString>\n"
            "\t\t\t\t<extrude>{}</extrude>\n"
            "\t\t\t\t<tesselate>{}</tesselate>\n"
            "\t\t\t\t<altitudeMode>{}</altitudeMode>\n"
            "\t\t\t\t<coordinates>\n".format(
                params["document_name"],
                params["document_desc"],
                params["line_color"],
                params["line_width"],
                params["poly_color"],
                params["path_name"],
                params["path_desc"],
                params["extrude"],
                params["tesselate"],
                params["alt_mode"]

            ))

        self.outfilehandle.write(header)

        ##### Write Coords #####
        for lon, lat in coord_table:
            try:
                if float(lat) == 0.0 and float(lon) == 0.0:
                    continue
            except ValueError:
                pass

            try:
                self.outfilehandle.write(
                    "\t\t\t\t\t{},{},0\n".format(
                        lon,lat
                    )
                )
            except IndexError:
                break
            
        ##### Write Footer #####
        footer = ("\t\t\t\t</coordinates>\n"
                  "\t\t\t</LineString>\n"
                  "\t\t</Placemark>\n"
                  "\t</Document>\n"
                  "</kml>")


        self.outfilehandle.write(footer)

    def make_path_v(self, lon1, lat1, lon2, lat2):

        ### Get Data ###
        # with open(inpath, newline='') as csvfile:
        #     reader = csv.DictReader(csvfile)
            #time_table = [row["Measurement_DateTime"] for row in reader]
            # coord_table = [[row["GPS_lon"], row["GPS_lat"]] for row in reader]
        coord_table = [[lon1,lat1], [lon2,lat2]]
            #for row in reader:
            #    time_table.append(row["Measurement_DateTime"])
            #   coord_table.append([row["GPS_lon"], row["GPS_lat"]])

        ##### Header #####

        params = {
            "document_name":"",
            "document_desc":"",
            "line_color":"ff00ffff",
            "line_width":"5",
            "poly_color":"ff00ff00",
            "path_name":"",
            "path_desc":"",
            "extrude":"0",
            "tesselate":"1",
            "alt_mode":"clampToGround"
        }

        header = ("\t<Document>\n"
            "\t\t<name>{}</name>\n"
            "\t\t<description>{}</description>\n"
            "\t\t<Style id=\"path_style\">\n"
            "\t\t\t<LineStyle>\n"
            "\t\t\t\t<color>{}</color>\n"
            "\t\t\t\t<width>{}</width>\n"
            "\t\t\t</LineStyle>\n"
            "\t\t\t<PolyStyle>\n"
            "\t\t\t\t<color>{}</color>\n"
            "\t\t\t</PolyStyle>\n"
            "\t\t</Style>\n"
            "\t\t<Placemark>\n"
            "\t\t\t<name>{}</name>\n"
            "\t\t\t<description>{}</description>\n"
            "\t\t\t<styleUrl>#path_style</styleUrl>\n"
            "\t\t\t<LineString>\n"
            "\t\t\t\t<extrude>{}</extrude>\n"
            "\t\t\t\t<tesselate>{}</tesselate>\n"
            "\t\t\t\t<altitudeMode>{}</altitudeMode>\n"
            "\t\t\t\t<coordinates>\n".format(
                params["document_name"],
                params["document_desc"],
                params["line_color"],
                params["line_width"],
                params["poly_color"],
                params["path_name"],
                params["path_desc"],
                params["extrude"],
                params["tesselate"],
                params["alt_mode"]

            ))

        self.outfilehandle.write(header)

        ##### Write Coords #####
        for lon, lat in coord_table:
            try:
                if float(lat) == 0.0 and float(lon) == 0.0:
                    continue
            except ValueError:
                pass

            try:
                self.outfilehandle.write(
                    "\t\t\t\t\t{},{},0\n".format(
                        lon,lat
                    )
                )
            except IndexError:
                break
            
        ##### Write Footer #####
        footer = ("\t\t\t\t</coordinates>\n"
                  "\t\t\t</LineString>\n"
                  "\t\t</Placemark>\n"
                  "\t</Document>\n"
                  "</kml>")


        self.outfilehandle.write(footer)

    def make_points(self, inpath, **kwargs):
        ### Get Data ###
        with open(inpath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data_table = [[row["GPS_lon"], row["GPS_lat"], row["fname"]] for row in reader]


        ### Very bare bones, just write points ###
        self.outfilehandle.write("\t<Document>\n")

        for lat, lon, fname in data_table:
            try:
                if float(lat) == 0.0 and float(lon) == 0.0:
                    continue
            except ValueError:
                pass
            try:
                self.outfilehandle.write("\t\t<Placemark>\n")
                if fname != "None":
                    self.outfilehandle.write("\t\t<name>{}</name>\n".format(fname))
                this_point = ("\t\t\t<Point>\n"
                            "\t\t\t\t<coordinates>"
                            "{},{}</coordinates>\n"
                            "\t\t\t</Point>\n"
                            "\t\t</Placemark>\n".format(lon, lat))
                self.outfilehandle.write(this_point)
            except IndexError:
                break
            except ValueError:
                continue
        
        ## Write footer
        self.outfilehandle.write("\t</Document>\n</kml>")