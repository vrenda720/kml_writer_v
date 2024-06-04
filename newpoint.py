import math

def calculate_new_point(lat1, lon1, distance, kappa):
    # Convert input from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    kappa_rad = math.radians(kappa)
    # kappa_rad = math.radians(-135)
    # kappa_rad = kappa # - 0.698132
    
    # Earth radius in kilometers
    R = 6371.0
    
    # Calculate the new latitude
    lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(distance / R) +
                         math.cos(lat1_rad) * math.sin(distance / R) * math.cos(kappa_rad))
    
    # Calculate the new longitude
    lon2_rad = lon1_rad + math.atan2(math.sin(kappa_rad) * math.sin(distance / R) * math.cos(lat1_rad),
                                     math.cos(distance / R) - math.sin(lat1_rad) * math.sin(lat2_rad))
    
    # Convert the results from radians back to degrees
    lat2_deg = math.degrees(lat2_rad)
    lon2_deg = math.degrees(lon2_rad)
    
    return lat2_deg, lon2_deg

# Example usage:
# lat1 = 40.7447856  # Starting latitude in degrees (Los Angeles)
# lon1 = -74.0245255  # Starting longitude in degrees
# distance = 0.5  # Distance in kilometers
# kappa = 45  # Azimuth in degrees
# kappa = -2.79977038597

# new_lat, new_lon = calculate_new_point(lat1, lon1, distance, kappa)
# print("New latitude:", new_lat)
# print("New longitude:", new_lon)


# "extrinsics": {
#           "lat": 40.7447856,
#           "lon": -74.0245255,
#           "alt": 40.25947364807129,
#           "omega": 0.07505407058,
#           "phi": 0.910781310906,
#           "kappa": -2.79977038597
#      },