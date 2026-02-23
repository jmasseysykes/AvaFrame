import numpy as np
import pathlib
import matplotlib.pyplot as plt
import logging
import pickle
import copy

import avaframe.ana5Utils.regionalThalwegTools as tools
from avaframe.in3Utils import cfgHandling
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.out3Plot.outAIMEC as outAIMEC
from avaframe.in3Utils import cfgUtils
from avaframe.ana3AIMEC import ana3AIMEC
import avaframe.ana5Utils.preparePathGeneral as pathGen
import avaframe.in1Data.getInput as gI
from avaframe.ana5Utils import DFAPathGeneration
import avaframe.in2Trans.rasterUtils as rasterUtils
from avaframe.out3Plot import outCom3Plots

log = logging.getLogger(__name__)


def regionalThalweg2DPlotMain(avalanchedir, cfg):
    """
    read in Input data and general function for 2D thalweg plot

    Parameters
    -----------
    avalanchedir: str
        Path to the th avalanche directory
    cfg: configparser Object
        contains configuration settings
    """
    avalanchedir = pathlib.Path(avalanchedir)

    cfgDFAPath = cfgUtils.getModuleConfig(
        DFAPathGeneration,
        onlyDefault=cfg["ana5Utils_DFAPathGeneration_override"].getboolean("defaultConfig"),
    )
    # and override with settings from config
    cfgDFAPath, cfg = cfgHandling.applyCfgOverride(cfgDFAPath, cfg, DFAPathGeneration, addModValues=False)

    simhash = cfg["GENERAL"].get("simHash")
    module = cfg["GENERAL"].get("modName")
    startRow = cfg["GENERAL"].get("startRow")
    startCol = cfg["GENERAL"].get("startCol")
    relId = cfg["GENERAL"].get("relId")

    pathToOutput = avalanchedir / "Outputs" / module / "peakFiles" / f"res_{simhash}"
    savePath = pathToOutput / "ThalwegPlots"
    fU.makeADir(savePath)
    pathDict = {"avalancheDir": avalanchedir, "pathToOutput": pathToOutput, "savePath": savePath}

    demDict = gI.readDEM(avalanchedir)
    # TODO: Check if flipping DEM is needed!(gI.readDem flips the raster.)
    # demDict["rasterData"] = np.flipud(demDict["rasterData"])

    # check which thalweg is plotted
    if startRow != "" and startCol != "" and relId != "":
        message = "When choosing one thalweg that is plotted, only select with startcell coordinates or release Id!"
        log.error(message)
        raise ValueError(message)
    plotAllThalwegs = False
    if startCol != "" or startRow != "":
        startCol = np.int16(startCol)
        startRow = np.int16(startRow)
    elif relId != "":
        relId = np.int16(relId)
    else:
        plotAllThalwegs = True

    centerOf = cfg["GENERAL"].get("centerOfVariable")
    if centerOf == "":
        plotAllCenterOf = True
    else:
        plotAllCenterOf = False

    pathDict["titleVariables"] = {
        "startRow": startRow,
        "startCol": startCol,
        "relId": relId,
        "centerOf": centerOf,
        "simHash": simhash,
    }

    # read in thalweg data
    if plotAllCenterOf:
        log.info(f"Plot all thalweg data that can be found in {pathToOutput}/ThalwegData.")
        files = sorted(list((pathToOutput / "thalwegData").glob(f"thalwegData_*.pickle")))
    elif plotAllThalwegs:
        log.info(
            f"Plot all thalwegs averaged with {centerOf} that can be found in {pathToOutput}/ThalwegData."
        )
        files = sorted(list((pathToOutput / "thalwegData").glob(f"thalwegData_{centerOf}_*.pickle")))
        if len(files) == 0:
            message = f"There is no thalweg data computed with {centerOf} in{pathToOutput}/ThalwegData."
            log.error(message)
            raise FileNotFoundError(message)
    fileDict = {}
    if plotAllThalwegs or plotAllCenterOf:
        for thalwegDataFile in files:
            stem = thalwegDataFile.stem
            nameParts = stem.split("_")
            if len(nameParts) == 4:
                _, centerOf, startRow, startCol = stem.split("_")
            elif len(nameParts) == 3:
                _, centerOf, relId = stem.split("_")
            pathDictLoop = copy.deepcopy(pathDict)

            pathDictLoop["titleVariables"]["startRow"] = startRow
            pathDictLoop["titleVariables"]["startCol"] = startCol
            pathDictLoop["titleVariables"]["centerOf"] = centerOf
            pathDictLoop["titleVariables"]["relId"] = relId

            dataThalweg = np.load(thalwegDataFile, allow_pickle="TRUE")

            _, profileExtended = pathGen.preparePathGeneralMain(dataThalweg, cfgDFAPath, demDict)
            fileDict[thalwegDataFile] = {"pathDict": pathDictLoop, "thalwegData": profileExtended}
            savePickle(profileExtended, thalwegDataFile)
    else:
        thalwegDataFile = pathToOutput / "thalwegData"
        dataThalweg = tools.readThalwegData(thalwegDataFile, pathDict["titleVariables"])
        _, profileExtended = pathGen.preparePathGeneralMain(dataThalweg, cfgDFAPath, demDict)
        savePickle(profileExtended, thalwegDataFile)
        fileDict[thalwegDataFile] = {"pathDict": pathDict, "thalwegData": profileExtended}

    # make plots
    for fileName in fileDict.keys():
        profileExtended = fileDict[fileName]["thalwegData"]
        pathDict = fileDict[fileName]["pathDict"]
        zDeltaRasterFile = tools.getRasterFile(pathDict["pathToOutput"], variable="zdelta")
        profileExtended["zdelta"] = tools.getThalwegValuesFromRaster(
            zDeltaRasterFile, profileExtended["x"], profileExtended["y"]
        )
        plotThalweg2D(pathDict, cfg, profileExtended)
        plotThalwegAltitude(pathDict, profileExtended)
        plotDFAGenerationLocation(pathDict, profileExtended, rasterVariable="fpTravelAngleMax")


def savePickle(profileExtended, inFileName):
    dir = inFileName.parent
    fileName = inFileName.stem
    outFileName = dir / f"extended_{fileName}.pickle"

    with open(outFileName, "wb") as handle:
        pickle.dump(profileExtended, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plotThalweg2D(pathDict, cfg, dataThalweg):
    """
    saves 2D thalweg plot:
    top panel: position of the thalweg in the field
    bottom panel: 2 dimensional representation

    Parameters
    ------------
    pathDict: dict
        contains the simulation paths
    cfg: configparser Object
        contains configuration settings
    dataThalweg: numpy array
        thalweg data that are saved in the simulation (averaged x-, y-coordinates, zdelta, ..)

    """
    variable = cfg["GENERAL"].get("plotVariable")
    thalwegPra = cfg["GENERAL"].getboolean("thalwegPra")
    size = cfg["GENERAL"].get("avalancheSize")
    savePath = pathDict["savePath"]
    centerOf = pathDict["titleVariables"]["centerOf"]

    if thalwegPra:
        folder = pathlib.Path(pathDict["pathToOutput"] / "thalwegData")
        files = list(folder.glob(f"extended_thalwegData_{centerOf}*"))
        x = []
        y = []

        for thalwegFile in files:
            data = np.load(thalwegFile, allow_pickle="TRUE")
            newX = np.array(data["x"])
            newY = np.array(data["y"])

            y.append(newY)
            x.append(newX)
    else:
        y = np.array(dataThalweg[f"y"])
        x = np.array(dataThalweg[f"x"])

    # PLOT
    fig, axs = plt.subplots(2, 1)

    fig.set_figheight(10)
    fig.tight_layout(pad=3.0)
    fig.set_figwidth(8)

    fig, axs[0] = tools.makeFieldPlot(
        axs[0], fig, pathDict, variable, x, y, dataThalweg, thalwegPra=thalwegPra
    )
    axs[1] = tools.makeThalwegPlot(axs[1], dataThalweg, pathDict, centerOf=centerOf)

    if size != "":
        axs[0].set_title(f"Avalanche size: {size}")

    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    outFileName = f"Thalweg2D_{outFileNamePart}.png"
    fig.savefig(savePath / outFileName)
    log.info(f"saved plot: {(savePath / outFileName)}")


def plotDFAGenerationLocation(pathDict, profile, rasterVariable="fpTravelAngleMax"):
    savePath = pathDict["savePath"]

    file = tools.getRasterFile(pathDict["pathToOutput"], variable=rasterVariable)
    rasterDict = rasterUtils.readRaster(file)
    raster = rasterDict["rasterData"]
    raster = np.where(raster > 0, raster, 0)

    dem = gI.readDEM(pathDict["avalancheDir"])

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=150)
    ax1 = outCom3Plots.avalancheThalwegPlot(ax1, raster, dem, profile)
    ax1.legend()
    # set plot limits depending on thalweg
    print(np.min(profile["x"]), np.max(profile["x"]), np.min(profile["y"]), np.max(profile["y"]))
    plt.xlim((np.min(profile["x"]) - 100, np.max(profile["x"]) + 100))
    plt.ylim((np.min(profile["y"]) - 100, np.max(profile["y"]) + 100))
    # ax1 = tools.DFAThalwegPlot(ax1, avaProfile, pathDict, rasterVariable)
    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    outFileName = f"DFA_thalwegLocation_{outFileNamePart}.png"
    print(ax1.get_xlim(), ax1.get_ylim())

    fig.savefig(savePath / outFileName)
    log.info(f"saved plot: {(savePath / outFileName)}")


def plotThalwegAltitude(pathDict, dataThalweg):
    """
    plot the AIMEC thalweg-altitude plot

    Parameters
    """
    dataThalweg["indStartOfRunout"] = 0
    dataThalweg["startOfRunoutAreaAngle"] = False

    velocityThalweg = tools.zDelta2velocity(dataThalweg["zdelta"])

    file = tools.getRasterFile(pathDict["pathToOutput"], variable="flux")
    flux = tools.getThalwegValuesFromRaster(file, dataThalweg["x"], dataThalweg["y"])
    pftCrossMax = flux * 10
    # pftCrossMax = np.ones_like(velocityThalweg) * 10

    cfg = cfgUtils.getModuleConfig(ana3AIMEC)
    cfgPlots = cfg["PLOTS"]

    simName = str(pathDict["avalancheDir"]).split("/")[-1]

    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    pathDict["projectName"] = outFileNamePart
    pathDict["pathResult"] = str(pathDict["savePath"])
    # TODO: we could divide the function outAIMEC.plotVelThAlongThalweg to enable modifications, e.g. the pft representation
    outAIMEC.plotVelThAlongThalweg(pathDict, dataThalweg, pftCrossMax, velocityThalweg, cfgPlots, simName)
