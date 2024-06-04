"""
Copyright © 2022-2023 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# YPR and OPK explained:
# https://support.pix4d.com/hc/en-us/articles/202558969-Yaw-Pitch-Roll-and-Omega-Phi-Kappa-angles
#
# Intrinsic and extrinsic parameters explained:
# https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
#
# The y axis from PIX4D is flipped:
# https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
#
# NAD83 Bursa Wolf parameters are provided here:
# https://gis.stackexchange.com/questions/112198/proj4-postgis-transformations-between-wgs84-and-nad83-transformations-in-alask/112202#112202
#
# Double check NAD83(2011) to WGS84 conversions using this online tool from NGS:
# https://www.ngs.noaa.gov/cgi-bin/HTDP/htdp.prl?f1=4&f2=1
#
# Code to convert to/from ENU was obtained here:
# https://stackoverflow.com/questions/53408780/convert-latitude-longitude-altitude-to-local-enu-coordinates-in-python

import json
import logging
import math
import typing
from glob import glob
from pathlib import Path

import numpy as np
import pyproj
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

_logger = logging.getLogger(__name__)

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    _logger.warning("scipy not installed: using non-scipy defaults")
    _USE_SCIPY = False
else:
    _logger.debug("By default using the scipy rotation functions")
    _USE_SCIPY = True


def _rotation_matrix(axis: str, angle_radians: float) -> np.ndarray:
    """Create a simple rotation matrix around a specified axis.

    Args:
        axis (str): the axis to rotate around (must be x, y, or z, case-sensitive)
        angle (float): the angle to rotate through, specified in radians

    Returns:
        np.ndarray: a 3x3 numpy array, suitable for matrix multiplication
    """
    sth = np.sin(angle_radians)
    cth = np.cos(angle_radians)
    _logger.debug(
        "returning single rotation for %s %0.3f: sin %0.3f, cos %0.3f",
        axis,
        angle_radians,
        sth,
        cth,
    )
    if axis == "x":
        return np.array(
            [
                [1, 0, 0],
                [0, cth, -sth],
                [0, sth, cth],
            ]
        )
    if axis == "y":
        return np.array(
            [
                [cth, 0, sth],
                [0, 1, 0],
                [-sth, 0, cth],
            ]
        )
    if axis == "z":
        return np.array(
            [
                [cth, -sth, 0],
                [sth, cth, 0],
                [0, 0, 1],
            ]
        )


def from_euler_esque(
    axes: str, angles: typing.Collection[float], degrees: bool = True
) -> np.ndarray:
    """Non-scipy substitution for Rotation.from_euler.as_matrix

    This does not take advantage of scipy's math boosts, and is likely to perform more
    slowly than its equivalent. However, it does not have a scipy dependency, and this
    might make it preferable for certain applications.

    Args:
        axes (str): order of axial rotations as Euler - case-insensitive
        angles (typing.Collection[float]): the angles to rotate each axis through
        degrees (bool, optional): indicates if the given angles are in degrees (not radians). Defaults to True.

    Returns:
        np.ndarray: a 3x3 numpy array, suitable for matrix multiplication
    """
    arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # to do the same thing as the from_euler function, we need to reverse the order of
    # arguments in matrix multiplication
    for axis, _angle in zip(axes.lower()[::-1], angles[::-1]):
        if degrees:
            angle = np.radians(_angle)
        else:
            angle = _angle
        _logger.debug("Processing axis %s, angle %0.3f radians", axis, angle)
        arr = arr.dot(_rotation_matrix(axis, angle))
    return arr


def geodetic_to_enu(
    lat, lon, alt, lat_org, lon_org, alt_org, use_scipy: bool = _USE_SCIPY
):
    """
    convert LLA to ENU
    :params lat, lon, alt: input LLA coordinates
    :params lat_org, lon_org, alt_org: LLA of the origin of the local ENU coordinate system
    :return: east, north, up coordinate
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    x_org, y_org, z_org = transformer.transform(
        lon_org, lat_org, alt_org, radians=False
    )
    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T
    if use_scipy:
        rot1 = Rotation.from_euler(
            "x", -(90 - lat_org), degrees=True
        ).as_matrix()  # angle*-1 : left handed *-1
        rot3 = Rotation.from_euler(
            "z", -(90 + lon_org), degrees=True
        ).as_matrix()  # angle*-1 : left handed *-1
    else:
        rot1 = from_euler_esque("x", -(90 - lat_org), degrees=True)
        rot3 = from_euler_esque("z", -(90 + lon_org), degrees=True)
    rotMatrix = rot1.dot(rot3)
    enu = rotMatrix.dot(vec).T.ravel()
    return enu.T


def enu_to_geodetic(x, y, z, lat_org, lon_org, alt_org, use_scipy: bool = _USE_SCIPY):
    """
    convert ENU to LLA
    :params x, y, z: input ENU coordinate
    :params lat_org, lon_org, alt_org: LLA of the origin of the local ENU coordinate system
    :return: lat, lon, alt coordinate
    """
    transformer1 = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    transformer2 = pyproj.Transformer.from_crs(
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x_org, y_org, z_org = transformer1.transform(
        lon_org, lat_org, alt_org, radians=False
    )
    ecef_org = np.array([[x_org, y_org, z_org]]).T
    if use_scipy:
        rot1 = Rotation.from_euler(
            "x", -(90 - lat_org), degrees=True
        ).as_matrix()  # angle*-1 : left handed *-1
        rot3 = Rotation.from_euler(
            "z", -(90 + lon_org), degrees=True
        ).as_matrix()  # angle*-1 : left handed *-1
    else:
        rot1 = from_euler_esque("x", -(90 - lat_org), degrees=True)
        rot3 = from_euler_esque("z", -(90 + lon_org), degrees=True)
    rotMatrix = rot1.dot(rot3)
    ecefDelta = rotMatrix.T.dot(np.array([[x, y, z]]).T)
    ecef = ecefDelta + ecef_org
    lon, lat, alt = transformer2.transform(
        ecef[0, 0], ecef[1, 0], ecef[2, 0], radians=False
    )
    return [lat, lon, alt]


def nad83_to_wgs84(lat, lon, hae):
    """
    convert coordinates from NAD83 ellipsoid to WGS84 ellipsoid
    :param lat: latitude in degrees
    :param lon: longitude in degrees
    :param hae: height above ellipsoid in meters
    :return: 3-tuple of WGS84 coordinates
    """
    # convert NAD83 to ECEF
    # using Bursa Wolf transformation parameters
    # and position vector method
    transformer = pyproj.Transformer.from_crs(
        {
            "proj": "latlong",
            "ellps": "GRS80",
            "datum": "NAD83",
            "towgs84": "-0.9956,1.9013,0.5215,0.025915,0.009426,0.0011599,-0.00062",
        },
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x, y, z = transformer.transform(lon, lat, hae, radians=False)
    # convert ECEF to WGS84
    transformer = pyproj.Transformer.from_crs(
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    longitude, latitude, altitude = transformer.transform(x, y, z, radians=False)
    return (latitude, longitude, altitude)


def utm_epsg_from_wgs84(latitude, longitude):
    """
    get UTM EPSG code for input WGS84 coordinate
    :param latitude, longitude: WGS84 coordinate in degrees
    :return: UTM EPSG code
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude,
            south_lat_degree=latitude,
            east_lon_degree=longitude,
            north_lat_degree=latitude,
        ),
    )
    utm_epsg = int(utm_crs_list[0].code)
    return utm_epsg


def get_utm_convergence_matrix(lat, lon, alt):
    """
    compute convergence matrix from true North for UTM projection to WGS84
    :params lat,lon,alt: input WGS84 coordinates
    :return: convergence matrix to convert UTM grid north to true north angle (degrees)
    """
    delta = 1e-6
    p1 = np.array(lla_to_utm(lat + delta, lon, alt))
    p2 = np.array(lla_to_utm(lat, lon, alt))
    xnp = p1 - p2
    angle = math.atan2(xnp[0], xnp[1])
    R_c = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return R_c


def lla_to_utm(lat, lon, alt, utm_epsg=None):
    """
    convert WGS84 coordinate to UTM
    :param lat,lon,alt: WGS84 coordinate
    :param utm_epsg: pre-calculated UTM EPSG
    :return: UTM x,y,z
    """
    if utm_epsg is None:
        utm_epsg = utm_epsg_from_wgs84(lat, lon)
    transformer = pyproj.Transformer.from_crs(4326, utm_epsg, always_xy=True)
    x, y, z = transformer.transform(lon, lat, alt)
    return x, y, z


def utm_to_lla(x, y, z, utm_epsg):
    """
    convert UTM coordinate to WGS84 LLA
    :param x,y,z: UTM coordinate
    :return lat,lon,alt: WGS84 coordinate
    """
    transformer = pyproj.Transformer.from_crs(utm_epsg, 4326, always_xy=True)
    lon, lat, alt = transformer.transform(x, y, z)
    return lon, lat, alt


def utm_to_ll(x, y, utm_epsg):
    """
    convert UTM coordinate to WGS84 LL
    :param x,y: UTM coordinate
    :return lat,lon: WGS84 coordinate in degrees
    """
    transformer = pyproj.Transformer.from_crs(utm_epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


"""
New change July 2023: We need to define the coordinate system ypr refers to.
First let's define the world coordinate: Z refers to NDIR, X is North, Y is East and we have a right hand system
The next question is: what is the camera pose at ypr = [0,0,0]?
We know Pix4D uses the coordinate system that [0,0,0] means the camera is looking ndir (Z), camera up points at North (X)
However, for most of the camera systems, we have a different [0,0,0] camera pose, such as the PTZ.
PTZ camera uses the coordinate system that [0,0,0] means the camera is facing North (X), with camera up pointing at -ndir (-Z)
This indicates 1. The rotation matrix for PTZ different since we have different initial position. 2. the meaning of ypr is different.
We define our ypr order as follow: First we rotate the yaw (Z), the pitch (Y) the last roll (X).
"""


def ypr_to_opk(lat, lon, alt, y, p, r, use_scipy: bool = _USE_SCIPY):
    """
    convert yaw, pitch, and roll to omega, phi, and kappa angles
    :param lat, lon, alt: WGS84 coordinate used to determine axis normals
    :param y, p, r: yaw, pitch, and roll angles in degrees
    :return: omega, phi, kappa angles in degrees
    """

    # fpr PTZ and other cameras we follow ZYX
    # should use intrinsic rotation
    if use_scipy:
        rot_ptz = Rotation.from_euler("ZYX", [y, p, r], degrees=True).as_matrix()
    else:
        rot_ptz = from_euler_esque("zyx", [y, p, r], degrees=True)
    # rot_ptz rotate the camera initially from [1,0,0], for a camera initially pointing at [0 0 1], we need to rotate
    # along Y axis for 90 degrees.  Note this is right hand system.
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    cnb = rot_ptz.dot(Ry)
    # swap x and y and flip z
    cbb = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    # compute x, y, and z normals; x is North in UTM grid
    delta = 1e-6
    p1 = np.array(lla_to_utm(lat + delta, lon, alt))
    p2 = np.array(lla_to_utm(lat - delta, lon, alt))
    xnp = p1 - p2
    m = np.linalg.norm(xnp)
    xnp /= m
    znp = np.array([0, 0, -1])
    ynp = np.cross(znp, xnp)
    # convert to OPK
    cen = np.array([xnp, ynp, znp]).T
    ceb = cen.dot(cnb).dot(cbb)
    omega = np.degrees(np.arctan2(-ceb[1][2], ceb[2][2]))
    phi = np.degrees(np.arcsin(ceb[0][2]))
    kappa = np.degrees(np.arctan2(-ceb[0][1], ceb[0][0]))
    return omega, phi, kappa


def opk_to_ypr(lat, lon, alt, o, p, k, use_scipy=_USE_SCIPY):
    """
    convert omega, phi, and kappa angles to yaw, pitch, and roll angles
    :param lat, lon, alt: nad83 coordinate used to determine axis normals
    :param o, p, k: omega, phi, kappa  angles in degrees
    :return: yaw, pitch, and roll angles in degrees
    """

    opk = [o, p, k]
    ceb = opk_to_rotation(opk)
    cbb = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    # compute x, y, and z normals; x is North in UTM grid
    delta = 1e-6
    p1 = np.array(lla_to_utm(lat + delta, lon, alt))
    p2 = np.array(lla_to_utm(lat - delta, lon, alt))
    xnp = p1 - p2
    m = np.linalg.norm(xnp)
    xnp /= m
    znp = np.array([0, 0, -1])
    ynp = np.cross(znp, xnp)
    # convert to OPK
    cen = np.array([xnp, ynp, znp]).T
    cnb = np.linalg.inv(cen).dot(ceb).dot(np.linalg.inv(cbb))

    R = cnb
    # additional Ry rotation -90
    Ry = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    R = R.dot(Ry)
    if use_scipy:
        r = Rotation.from_matrix(R)
        [yaw, pitch, roll] = r.as_euler("ZYX", degrees=True)

        # if yaw<0:
        #     yaw +=360
        # if roll<0:
        #     roll+=360
    else:
        yaw = math.atan2(R[1, 0], R[0, 0]) * 180 / math.pi
        # if yaw < 0:
        #     yaw += 360
        pitch = (
            math.atan(-R[2, 0] / (R[2, 1] ** 2 + R[2, 2] ** 2) ** 0.5) * 180 / math.pi
        )
        roll = math.atan2(R[2, 1], R[2, 2]) * 180 / math.pi

        if roll < -90:  # roll out of range
            roll += 180
            pitch = 180 - pitch
            yaw -= 180

        if roll > 90:
            roll -= 180
            pitch = 180 - pitch
            yaw -= 180
        if yaw < 0:
            yaw += 360

    return yaw, pitch, roll


def opk_to_rotation(opk_degrees):
    """
    calculate a rotation matrix given euler angles
    :params opk: list of [omega, phi, kappa] angles (degrees)
    :return: rotation matrix
    """
    opk = np.radians(opk_degrees)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(opk[0]), -math.sin(opk[0])],
            [0, math.sin(opk[0]), math.cos(opk[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(opk[1]), 0, math.sin(opk[1])],
            [0, 1, 0],
            [-math.sin(opk[1]), 0, math.cos(opk[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(opk[2]), -math.sin(opk[2]), 0],
            [math.sin(opk[2]), math.cos(opk[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R


def opk_to_rotation_portrait(opk_degrees):
    """
    calculate a rotation matrix given euler angles
    :params opk: list of [omega, phi, kappa] angles (degrees)
    :return: rotation matrix
    """
    opk = np.radians(opk_degrees)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(opk[0]), -math.sin(opk[0])],
            [0, math.sin(opk[0]), math.cos(opk[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(opk[1]), 0, math.sin(opk[1])],
            [0, 1, 0],
            [-math.sin(opk[1]), 0, math.cos(opk[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(opk[2]), -math.sin(opk[2]), 0],
            [math.sin(opk[2]), math.cos(opk[2]), 0],
            [0, 0, 1],
        ]
    )
    R_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T
    R = np.dot(R_x, np.dot(R_y, np.dot(R_z, R_90)))
    return R


def rotation_to_opk(R):
    """
    calculate euler angles from a rotation matrix
    :param R: input rotation matrix
    :return opk: list of [omega, phi, kappa] angles (degrees)
    """
    omega = np.degrees(np.arctan2(-R[1][2], R[2][2]))
    phi = np.degrees(np.arcsin(R[0][2]))
    kappa = np.degrees(np.arctan2(-R[0][1], R[0][0]))
    return [omega, phi, kappa]


def lla_to_image_utm_opk(latitude, longitude, altitude, params):
    """
    project WGS84 coordinate into an image with WRIVA JSON metadata using UTM coordinates and OPK angles (used only for debugging)
    :params latitude, longitude, altitude: input WGS84 coordinate
    :param params: dictionary with image projection metadata
    :return: image coordinates
    """
    # convert LLA to UTM
    tlat = params["extrinsics"]["lat"]
    tlon = params["extrinsics"]["lon"]
    talt = params["extrinsics"]["alt"]
    # UTM should be fine for close ranges; ENU is preferred because it will work for anything
    x, y, z = lla_to_utm(latitude, longitude, altitude)
    tx, ty, tz = lla_to_utm(tlat, tlon, talt)
    # project world coordinates to camera coordinates
    # convert north angle from true north to UTM grid north
    omega = params["extrinsics"]["omega"]
    phi = params["extrinsics"]["phi"]
    kappa = params["extrinsics"]["kappa"]
    R = opk_to_rotation([omega, phi, kappa])
    T = np.array([tx, ty, tz])
    X_world = np.array([x, y, z])
    T_trans = X_world - T
    X_camera = np.matmul(R.T, T_trans)
    # Negate the camera y axis since image rows are flipped
    X_camera[1] = -X_camera[1]
    # project camera coordinates to pinhole pixel coordinates
    fx = params["intrinsics"]["fx"]
    fy = params["intrinsics"]["fy"]
    cx = params["intrinsics"]["cx"]
    cy = params["intrinsics"]["cy"]
    column = -fx * X_camera[0] / X_camera[2] + cx
    row = -fy * X_camera[1] / X_camera[2] + cy
    # project camera coordinates to distorted pixel coordinates
    k1 = params["intrinsics"]["k1"]
    k2 = params["intrinsics"]["k2"]
    k3 = params["intrinsics"]["k3"]
    p1 = params["intrinsics"]["p1"]
    p2 = params["intrinsics"]["p2"]
    xh = X_camera[0] / X_camera[2]
    yh = X_camera[1] / X_camera[2]
    r2 = xh * xh + yh * yh
    gamma = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xhd = xh * gamma + 2.0 * p1 * xh * yh + p2 * (r2 + 2 * xh * xh)
    yhd = yh * gamma + p1 * (r2 + 2 * yh * yh) + 2.0 * p2 * xh * yh
    column_distorted = -fx * xhd + cx
    row_distorted = -fy * yhd + cy
    return column_distorted, row_distorted


def lla_to_image_enu_opk(latitude, longitude, altitude, params, return_front=False):
    """
    project WGS84 coordinate into an image with WRIVA JSON metadata using ENU coordinates and OPK angles
    :params latitude, longitude, altitude: input WGS84 coordinate
    :param params: dictionary with image projection metadata
    :param return_front: bool whether to return output indicating whether LLA is in front of camera
    :return: image coordinates(, front)
    """
    # convert LLA to UTM or ENU
    tlat = params["extrinsics"]["lat"]
    tlon = params["extrinsics"]["lon"]
    talt = params["extrinsics"]["alt"]
    # Any ENU coordinate system referenced to true north should produce the same result
    x, y, z = geodetic_to_enu(latitude, longitude, altitude, tlat, tlon, talt)
    tx, ty, tz = geodetic_to_enu(tlat, tlon, talt, tlat, tlon, talt)
    # project world coordinates to camera coordinates
    # convert north angle from true north to UTM grid north
    omega = params["extrinsics"]["omega"]
    phi = params["extrinsics"]["phi"]
    kappa = params["extrinsics"]["kappa"]
    R = opk_to_rotation([omega, phi, kappa])
    T = np.array([tx, ty, tz])
    X_world = np.array([x, y, z])
    T_trans = X_world - T
    X_camera = np.matmul(R.T, T_trans)
    # Negate the camera y axis since image rows are flipped
    X_camera[1] = -X_camera[1]
    # project camera coordinates to distorted pixel coordinates
    if "projection" in params and params["projection"] == "fisheye_pix4d":
        cx = params["intrinsics"]["cx"]
        cy = params["intrinsics"]["cy"]
        p2 = params["intrinsics"]["p2"]
        p3 = params["intrinsics"]["p3"]
        p4 = params["intrinsics"]["p4"]
        C = params["intrinsics"]["C"]
        D = params["intrinsics"]["D"]
        E = params["intrinsics"]["E"]
        F = params["intrinsics"]["F"]
        theta = abs(
            (2.0 / math.pi)
            * math.atan(
                math.sqrt(X_camera[0] * X_camera[0] + X_camera[1] * X_camera[1])
                / X_camera[2]
            )
        )
        rho = (
            theta
            + p2 * theta * theta
            + p3 * theta * theta * theta
            + p4 * theta * theta * theta * theta
        )
        xh = (
            rho
            * X_camera[0]
            / math.sqrt(X_camera[0] * X_camera[0] + X_camera[1] * X_camera[1])
        )
        yh = (
            rho
            * X_camera[1]
            / math.sqrt(X_camera[0] * X_camera[0] + X_camera[1] * X_camera[1])
        )
        column = C * xh + D * yh + cx
        row = E * xh + F * yh + cy
    else:
        fx = params["intrinsics"]["fx"]
        fy = params["intrinsics"]["fy"]
        cx = params["intrinsics"]["cx"]
        cy = params["intrinsics"]["cy"]
        k1 = params["intrinsics"]["k1"]
        k2 = params["intrinsics"]["k2"]
        k3 = params["intrinsics"]["k3"]
        p1 = params["intrinsics"]["p1"]
        p2 = params["intrinsics"]["p2"]
        xh = X_camera[0] / X_camera[2]
        yh = X_camera[1] / X_camera[2]
        r2 = xh * xh + yh * yh
        gamma = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xhd = xh * gamma + 2.0 * p1 * xh * yh + p2 * (r2 + 2 * xh * xh)
        yhd = yh * gamma + p1 * (r2 + 2 * yh * yh) + 2.0 * p2 * xh * yh
        K = [[-fx, 0, cx], [0, -fy, cy], [0, 0, 1]]
        ijk = np.matmul(K, [xhd, yhd, 1.0])
        column = ijk[0] / ijk[2]
        row = ijk[1] / ijk[2]
    if return_front:
        return column, row, X_camera[2] < 0
    else:
        return column, row


def utm_opk_to_enu_opk(lat, lon, alt, omega_utm, phi_utm, kappa_utm):
    """
    convert UTM OPK angles to ENU OPK angles
    :params lat, lon, alt: camera location in WGS84 coordinates
    :params omega_utm, phi_utm, kappa_utm: camera OPK angles in UTM projection
    :return omega_enu, phi_enu, kappa_enu: camera OPK angles in ENU
    """
    # build rotation matrix from OPK angles
    R = opk_to_rotation([omega_utm, phi_utm, kappa_utm])
    # get UTM convergence angle rotation matrix and map UTM OPK to ENU
    R_c = get_utm_convergence_matrix(lat, lon, alt)
    R = np.dot(R_c, R)
    omega_enu, phi_enu, kappa_enu = rotation_to_opk(R)
    return omega_enu, phi_enu, kappa_enu


def enu_opk_to_utm_opk(lat, lon, alt, omega_enu, phi_enu, kappa_enu):
    """
    convert ENU OPK angles to UTM OPK angles
    :params lat, lon, alt: camera location in WGS84 coordinates
    :params omega_enu, phi_enu, kappa_enu: camera OPK angles in ENU
    :return omega_utm, phi_utm, kappa_utm: camera OPK angles in UTM projection
    """
    # build rotation matrix from OPK angles
    R = opk_to_rotation([omega_enu, phi_enu, kappa_enu])
    # get UTM convergence angle rotation matrix and map UTM OPK to ENU
    R_c = get_utm_convergence_matrix(lat, lon, alt)
    R = np.dot(np.linalg.inv(R_c), R)
    omega_utm, phi_utm, kappa_utm = rotation_to_opk(R)
    return omega_utm, phi_utm, kappa_utm


def lla_in_image_fov(latitude, longitude, altitude, params):
    """
    check if world coordinate is inside image field of view
    :params latitude, longitude, altitude: world coordinate
    :param params: camera parameters
    :return inside: bool is True if LLA is inside image FOV
    :return front: bool is True if LLA is in front of camera
    :return norm_dist: normalized distance from image center
    """
    # compute normalized distance of xyz projection to image center
    # reject if projected image coordinates are out of bounds
    col, row, front = lla_to_image_enu_opk(
        latitude, longitude, altitude, params, return_front=True
    )
    # reject if xyz is behind the camera
    if not front:
        inside = False
        norm_dist = 1e6
    else:
        cx = params["intrinsics"]["columns"] / 2.0
        cy = params["intrinsics"]["rows"] / 2.0
        dx = abs((cx - col) / cx)
        dy = abs((cy - row) / cy)
        norm_dist = math.sqrt(dx * dx + dy * dy)
        inside = dx < 1.0 and dy < 1.0
    # if xyz is close to perpendicular to the camera line of sight, computed image coordinates will be unreliable
    # and xyz will not be visible in the image, except maybe for the most extreme fisheye lens, so reject
    if inside:
        cam_lat = params["extrinsics"]["lat"]
        cam_lon = params["extrinsics"]["lon"]
        cam_alt = params["extrinsics"]["alt"]
        wx, wy, wz = geodetic_to_enu(
            latitude, longitude, altitude, cam_lat, cam_lon, cam_alt
        )
        cx, cy, cz = geodetic_to_enu(
            cam_lat, cam_lon, cam_alt, cam_lat, cam_lon, cam_alt
        )
        yaw, pitch, roll = opk_to_ypr(
            cam_lat,
            cam_lon,
            cam_alt,
            params["extrinsics"]["omega"],
            params["extrinsics"]["phi"],
            params["extrinsics"]["kappa"],
        )
        pitch_point = math.degrees(
            math.atan2(
                wz - cz, math.sqrt((cx - wx) * (cx - wx) + (cy - wy) * (cy - wy))
            )
        )
        yaw_point = math.degrees(math.atan2(wx - cx, wy - cy))
        pitch_diff = abs((pitch - pitch_point) % 360)
        if pitch_diff > 180:
            pitch_diff = 360 - pitch_diff
        yaw_diff = abs((yaw - yaw_point) % 360)
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        if yaw_diff > 40 or pitch_diff > 40:
            inside = False
            norm_dist = 1e6
    return inside, front, norm_dist


""" The function below could replace lla_to_image_enu_opk above now with no other further changes...
"""


def lla_to_image_enu_opk_arrays(
    latitude, longitude, altitude, params, return_front=False
):
    """
    project WGS84 coordinate into an image with WRIVA JSON metadata using ENU coordinates and OPK angles
    :params latitude, longitude, altitude: input WGS84 coordinate
    :param params: dictionary with image projection metadata
    :param return_front: bool whether to return output indicating whether LLA is in front of camera
    :return: image coordinates(, front)
    """
    use_arrays = isinstance(latitude, np.ndarray)
    print("use_arrays = ", use_arrays)
    # convert LLA to UTM or ENU
    tlat = params["extrinsics"]["lat"]
    tlon = params["extrinsics"]["lon"]
    talt = params["extrinsics"]["alt"]
    # Any ENU coordinate system referenced to true north should produce the same result
    x, y, z = geodetic_to_enu(latitude, longitude, altitude, tlat, tlon, talt)
    tx, ty, tz = geodetic_to_enu(tlat, tlon, talt, tlat, tlon, talt)
    # project world coordinates to camera coordinates
    # convert north angle from true north to UTM grid north
    omega = params["extrinsics"]["omega"]
    phi = params["extrinsics"]["phi"]
    kappa = params["extrinsics"]["kappa"]
    R = opk_to_rotation([omega, phi, kappa])
    T = np.array([tx, ty, tz])
    X_world = np.array([x, y, z])
    #    T_trans = X_world - T
    T_trans = (X_world.T - T).T
    X_camera = np.matmul(R.T, T_trans)
    # Negate the camera y axis since image rows are flipped
    X_camera[1] = -X_camera[1]
    # project camera coordinates to distorted pixel coordinates
    if "projection" in params and params["projection"] == "fisheye_pix4d":
        cx = params["intrinsics"]["cx"]
        cy = params["intrinsics"]["cy"]
        p2 = params["intrinsics"]["p2"]
        p3 = params["intrinsics"]["p3"]
        p4 = params["intrinsics"]["p4"]
        C = params["intrinsics"]["C"]
        D = params["intrinsics"]["D"]
        E = params["intrinsics"]["E"]
        F = params["intrinsics"]["F"]
        theta = abs(
            (2.0 / math.pi)
            * math.atan(
                math.sqrt(X_camera[0] * X_camera[0] + X_camera[1] * X_camera[1])
                / X_camera[2]
            )
        )
        rho = (
            theta
            + p2 * theta * theta
            + p3 * theta * theta * theta
            + p4 * theta * theta * theta * theta
        )
        xh = (
            rho
            * X_camera[0]
            / math.sqrt(X_camera[0] * X_camera[0] + X_camera[1] * X_camera[1])
        )
        yh = (
            rho
            * X_camera[1]
            / math.sqrt(X_camera[0] * X_camera[0] + X_camera[1] * X_camera[1])
        )
        column = C * xh + D * yh + cx
        row = E * xh + F * yh + cy
    else:
        fx = params["intrinsics"]["fx"]
        fy = params["intrinsics"]["fy"]
        cx = params["intrinsics"]["cx"]
        cy = params["intrinsics"]["cy"]
        k1 = params["intrinsics"]["k1"]
        k2 = params["intrinsics"]["k2"]
        k3 = params["intrinsics"]["k3"]
        p1 = params["intrinsics"]["p1"]
        p2 = params["intrinsics"]["p2"]
        xh = X_camera[0] / X_camera[2]
        yh = X_camera[1] / X_camera[2]
        r2 = xh * xh + yh * yh
        gamma = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xhd = xh * gamma + 2.0 * p1 * xh * yh + p2 * (r2 + 2 * xh * xh)
        yhd = yh * gamma + p1 * (r2 + 2 * yh * yh) + 2.0 * p2 * xh * yh
        K = [[-fx, 0, cx], [0, -fy, cy], [0, 0, 1]]
        if use_arrays:
            ijk = np.matmul(K, [xhd, yhd, np.squeeze(np.ones(len(xhd)))])
        else:
            ijk = np.matmul(K, [xhd, yhd, 1.0])
        column = ijk[0] / ijk[2]
        row = ijk[1] / ijk[2]
    if return_front:
        return column, row, X_camera[2] < 0
    else:
        return column, row

# opk_to_ypr(lat, lon, alt, o, p, k, use_scipy=_USE_SCIPY)
# yaw, pitch, roll = opk_to_ypr(40.745268, -74.025488, 32.2, 87.63394218002315, 0.6402805446239054, 0.1403051924373668)
# print(pitch, roll, yaw)
# omega, phi, kappa = ypr_to_opk(40.745268, -74.025488, 32.2, yaw, pitch, roll)
# print(omega, phi, kappa)

# from RPY2OPK import rph2opk
# print("----------------")
# omega, phi, kappa = rph2opk(roll, pitch, yaw)
# print(omega, phi, kappa)
