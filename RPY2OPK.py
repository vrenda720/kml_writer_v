# from scipy.spatial.transform import Rotation as R
# import numpy as np

# # Define the RPY angles
# roll = np.deg2rad(45)   # Roll in radians
# pitch = np.deg2rad(30)  # Pitch in radians
# yaw = np.deg2rad(60)    # Yaw in radians

# # Create the rotation object from RPY angles
# rpy_rotation = R.from_euler('xyz', [roll, pitch, yaw])

# # Extract the rotation matrix
# rotation_matrix = rpy_rotation.as_matrix()

# # Create the rotation object from the rotation matrix
# opk_rotation = R.from_matrix(rotation_matrix)

# # Extract the OPK angles (assuming the order is 'xyz' for omega, phi, kappa)
# opk_angles = opk_rotation.as_euler('xyz', degrees=True)

# omega, phi, kappa = opk_angles
# print(f'Omega: {omega} degrees, Phi: {phi} degrees, Kappa: {kappa} degrees')


# from scipy.spatial.transform import Rotation as R
# import numpy as np

# # Define the RPY angles in degrees
# roll = 45   # Roll in degrees
# pitch = 30  # Pitch in degrees
# yaw = 60    # Yaw in degrees

# # Convert RPY angles to radians
# roll_rad = np.deg2rad(roll)
# pitch_rad = np.deg2rad(pitch)
# yaw_rad = np.deg2rad(yaw)

# # Create the rotation object from RPY angles (in radians)
# rpy_rotation = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])

# # Extract the rotation matrix
# rotation_matrix = rpy_rotation.as_matrix()

# # Create the rotation object from the rotation matrix
# opk_rotation = R.from_matrix(rotation_matrix)

# # Extract the OPK angles (assuming the order is 'xyz' for omega, phi, kappa)
# opk_angles = opk_rotation.as_euler('xyz', degrees=True)

# # Assign the OPK angles
# omega, phi, kappa = opk_angles

# # Print the results
# print(f"Roll (ϕ): {roll} degrees")
# print(f"Pitch (θ): {pitch} degrees")
# print(f"Yaw (ψ): {yaw} degrees")
# print()
# print(f"Omega (ω): {omega:.2f} degrees")
# print(f"Phi (φ): {phi:.2f} degrees")
# print(f"Kappa (κ): {kappa:.2f} degrees")


# from scipy.spatial.transform import Rotation as R
# import numpy as np

# # Define the RPY angles in degrees
# roll = 45   # Roll in degrees
# pitch = 30  # Pitch in degrees
# yaw = 60    # Yaw in degrees

# # Convert RPY angles to radians
# roll_rad = np.deg2rad(roll)
# pitch_rad = np.deg2rad(pitch)
# yaw_rad = np.deg2rad(yaw)

# # Create the rotation object from RPY angles (in radians)
# rpy_rotation = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])

# # Extract the rotation matrix
# rotation_matrix = rpy_rotation.as_matrix()

# # Convert the rotation matrix back to OPK angles
# opk_rotation = R.from_matrix(rotation_matrix)

# # Extract the OPK angles (assuming the order is 'zyx' for omega, phi, kappa)
# opk_angles = opk_rotation.as_euler('zyx', degrees=True)

# # Assign the OPK angles
# omega, phi, kappa = opk_angles

# # Print the results
# print(f"Roll (ϕ): {roll} degrees")
# print(f"Pitch (θ): {pitch} degrees")
# print(f"Yaw (ψ): {yaw} degrees")
# print()
# print(f"Omega (ω): {omega:.2f} degrees")
# print(f"Phi (φ): {phi:.2f} degrees")
# print(f"Kappa (κ): {kappa:.2f} degrees")


import numpy as np


def getSignOf(chifre):
    if chifre >= 0:
        return 1
    else:
        return -1

def rph2opk(Roll, Pitch, heading):
    Roll = np.deg2rad(Roll)
    Pitch   = np.deg2rad(Pitch)
    heading = np.deg2rad(heading)

    A_SINH = np.sin(heading)
    A_SINR = np.sin(Roll)
    A_SINP = np.sin(Pitch)

    A_COSH = np.cos(heading)
    A_COSR = np.cos(Roll)
    A_COSP = np.cos(Pitch)

    MX = np.zeros((3, 3))
    MX[0][0] =  (A_COSH *A_COSR) + (A_SINH*A_SINP*A_SINR)
    MX[0][1] =  (-A_SINH*A_COSR)+(A_COSH*A_SINP*A_SINR)
    MX[0][2] =   -A_COSP*A_SINR

    MX[1][0] = A_SINH*A_COSP
    MX[1][1] = A_COSH*A_COSP
    MX[1][2] = A_SINP


    MX[2][0] = (A_COSH*A_SINR)-(A_SINH*A_SINP*A_COSR)
    MX[2][1] = (-A_SINH*A_SINR)-(A_COSH*A_SINP*A_COSR)
    MX[2][2] =  A_COSP*A_COSR

    P = np.zeros((3, 3))
    P[0][0] = MX[0][0]
    P[0][1] = MX[1][0]
    P[0][2] = MX[2][0]
    
    P[1][0] = MX[0][1]
    P[1][1] = MX[1][1]
    P[1][2] = MX[2][1]
    
    P[2][0] = MX[2][0]
    P[2][1] = MX[1][2]
    P[2][2] = MX[2][2]

    Omega = 0
    Phi   = 0
    Kappa = 0

    Omega = np.arctan(-P[2][1]/P[2][2])
    Phi = np.arcsin(P[2][2])
    Kappa = np.arctan(-P[1][0]/P[0][0])

    Phi   = abs(np.arcsin(P[2][0]))
    Phi = Phi * getSignOf(P[2][0])
    Omega = abs(np.arccos((P[2][2] / np.cos(Phi))))
    Omega = Omega * (getSignOf(P[2][1] / P[2][2]*-1))
    Kappa = np.arccos(P[0][0] / np.cos(Phi))

    if getSignOf(P[0][0]) == getSignOf((P[1][0] / P[0][0])):
        Kappa = Kappa * -1

    Omega = np.rad2deg(Omega)
    Phi   = np.rad2deg(Phi)
    Kappa = np.rad2deg(Kappa)

    return(Omega,Phi,Kappa)

# roll = 10
# pitch = 10
# yaw = 90
# omega, phi, kappa = rph2opk(roll, pitch, yaw)


# print ('omega = ', omega)
# print ('phi = ', phi)
# print ('kappa = ', kappa)

# https://github.com/davdmaccartney/rpy_opk/blob/master/rpy_opk.py