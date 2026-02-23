"""
generate talweg from x and y coordinates
"""

import numpy as np
import logging

import avaframe.in3Utils.geoTrans as gT
from avaframe.ana5Utils import DFAPathGeneration

log = logging.getLogger(__name__)
logName = "runRegionalThalweg2DPlot"


def xyToProfile(x, y, dem):
    """
    for given coordinates (of the talweg) read z values
    from DEM and compute distance between coordinates

    Parameters
    ------------
    x: np.array
        x coordinates
    y: np.array
        y coordinates
    dem: dict
        contains dem data
    """
    demHeader = dem["header"]

    z, _ = gT.projectOnGrid(
        x,
        y,
        dem["rasterData"],
        csz=demHeader["cellsize"],
        xllc=demHeader["xllcenter"],
        yllc=demHeader["yllcenter"],
    )
    s = np.append(np.array([0]), gT.computeLengthOfLine2D(x, y))
    profile = {"x": x, "y": y, "z": z, "s": s}

    return profile


def preparePathGeneralMain(profile, cfgDFAPath, dem):
    """
     prepare thalweg from x and y coordinates:
     1. read z coordinates from DEM and compute horizontally projected distance
     2. extend path to bottom and top
     3. resample path points

     Parameters
     -------------
     profile: dict
        contains x and y coordinates of thalweg location
    cfgDFAPath: configparser object
        configuration for DFA pat generation
    dem: dict
        elevation model

    Returns
    -------------
    profileAveraged: dict
        s and z coordinates are added (x and y original)
    profileExtended: dict
        x, y, s, z of extended and resampled path
    """
    x = profile["x"]
    y = profile["y"]
    # get profile with normalized x and y coordinates and z and s values
    profileAveraged = xyToProfile(x, y, dem)
    profileExtended = profileAveraged.copy()
    # if extTopOption == 2, particlesIni are not used!!
    profileExtended = pathExtension(profileExtended, dem, cfgDFAPath)
    # resample profile/ path and save in an extra dictionary
    profileExtended = DFAPathGeneration.resamplePath(cfgDFAPath["PATH"], dem, profileExtended)
    # profileExtended = replaceResampledProfileCore(profileExtended, profileResample)
    for inputPara in ["alpha", "exponent", "zDeltaMax"]:
        profileExtended[inputPara] = profile[inputPara]

    return profileAveraged, profileExtended


def replaceResampledProfileCore(profile, profileResample):
    """
    for all variables (x, y, s, z, zdelta, fluxSum, flowEnergy)
    use the resampled top and bottom (extended) values and the original vlaues inbetween.

    Parameters
    -----------
    profile: dict
        contains the original thalweg data, that is kept in the core
    profileResample: dict
        contains extended and resampled thalweg data that is kept at start and end

    Returns
    ------------
    profile: dict
        contains original data in core and extended and resampled data at start and end

    """
    indStart = profile["indStartMassAverage"]
    indEnd = profile["indEndMassAverage"] + 1
    indStartRes = profileResample["indStartMassAverage"]
    indEndRes = profileResample["indEndMassAverage"] + 1

    lenExtTop = len(profileResample["x"][0:indStartRes])
    lenExtBot = len(profileResample["x"][indEndRes:])

    for key in profile.keys():
        if key in ["indStartMassAverage", "indEndMassAverage"]:
            continue
        else:
            resampledTop = profileResample[key][0:indStartRes]
            resampledBottom = profileResample[key][indEndRes:]
            keepCore = profile[key][indStart:indEnd]
        profile[key] = np.concatenate((resampledTop, keepCore, resampledBottom))

    return profile


def pathExtension(profile, demDict, cfgPathGen):
    """
    extend thalweg to top and bottom of path
    (now both options depend on the direction of the path)

    Parameters
    ------------
    profile: dict
        thalweg data
    demDict: dict
        DEM data
    cfgPathGen: confiparser object
        configuration setup for DFA Path generation

    Returns
    -------------
    profile: dict
        thalweg data that are extended to top and bottom
    """

    cfgPathGen["PATH"]["extTopOption"] = "2"
    profile["indStartMassAverage"] = 1
    profile["indEndMassAverage"] = np.size(profile["x"]) - 1

    # TODO: also allow extTopOption 0 and 1 ?

    profile = DFAPathGeneration.extendProfileTop(
        cfgPathGen["PATH"].getint("extTopOption"),
        {},
        profile,
        dem=demDict,
        cfg=cfgPathGen["PATH"],
        considerLLC=True,
    )

    # extend the bottom quite far
    profile = DFAPathGeneration.extendProfileBottom(cfgPathGen["PATH"], demDict, profile, considerLLC=True)

    return profile
