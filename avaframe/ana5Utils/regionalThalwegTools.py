"""
Tools/ help functions for regional thalweg plots
TODO: (further functions that could be helpful are in the bitbucket repo FlowPy postprocessing)
"""

import numpy as np
import pathlib
import rasterio
import logging
import os
from cmcrameri import cm as cmapCrameri
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pickle

import avaframe.in2Trans.rasterUtils as rasterUtils
import avaframe.in1Data.getInput as gI
import avaframe.out3Plot.plotUtils as pU
import avaframe.in3Utils.geoTrans as gT

# create local logger
log = logging.getLogger(__name__)


def getRasterFile(path, variable=""):
    """
    read in raster (*.asc or *.tif)

    Parameters:
    -----------
    path: pathlib.Path
        path to raster file or folder containing raster
    variable: str
        test part that is searched for (name is in file name)

    Returns:
    -----------
    filePath: pathlib Path
        path to raster file in the folder
    """
    path = pathlib.Path(path)

    try:
        raster = rasterio.open(path)
        filePath = path
    except:
        files = sorted(list(path.glob(f"*{variable}.asc")))
        if len(files) == 0:
            files = sorted(list(path.glob(f"*{variable}.tif")))
        raster = rasterio.open(files[0])
        # filePath = pathlib.Path(files[0])
        filePath = files[0]
    return filePath


def zDelta2velocity(zDelta):
    """compute velocity from energy line hight
    Parameters
    -----------
    zDelta: numpy float or array
        energy line height

    Returns
    -----------
    velocity: numpy float or array
        velocity comuted frm zDelta
    """
    velocity = (zDelta * 2 * 9.81) ** 0.5
    return velocity


def readThalwegData(path, titleDict):
    """
    load thalweg data

    Parameters:
    -----------
    path: pathlib.Path
        OutputPath of the FlowPy simulation
    titleDict: dict
        contains

    Returns:
    -----------
    data: dict
        thalweg data of one thalweg
    """
    centerOf = titleDict["centerOf"]
    startRow = titleDict["startRow"]
    startCol = titleDict["startCol"]
    relId = titleDict["relId"]

    if startRow != "":
        filePath = pathlib.Path(f"{path}/thalwegData_{centerOf}_{startRow}_{startCol}.pickle")
    else:
        filePath = pathlib.Path(f"{path}/thalwegData_{centerOf}_{relId}.pickle")
    if filePath.is_file():
        data = np.load(filePath, allow_pickle="TRUE")
    else:
        message = f"No thalwegdata exist averaged with {centerOf} for starcell with row {startRow} and column {startCol} in {path}"
        log.error(message)
        raise FileNotFoundError(message)
    return data


def getOutFileNamePartly(titleDict):
    """
    make name for outputfile

    Paramaters
    -------------
    titleDict: dict
    contains parameters of avalanche path

    Returns
    -------------
    outFileNamePart: str
        name for outputfile
    """
    centerOf = titleDict["centerOf"]
    startRow = titleDict["startRow"]
    startCol = titleDict["startCol"]
    relId = titleDict["relId"]
    simhash = titleDict["simHash"]

    if relId != "":
        outFileNamePart = f"{simhash}_{centerOf}_{relId}"
    else:
        outFileNamePart = f"{simhash}_{centerOf}_{startRow}_{startCol}"

    return outFileNamePart


def plotField(ax, fig, pathDict, variable, thalwegPra=False):
    """plots hillshade of the DEM and the output raster of the simulation zoomed in to the simulation extent

    Parameters:
    -----------
    ax: matplotlib axis
        axis in which the hillshade and output raster is plotted
    fig: matplotlib figure
        figure to that the plot belongs to
    pathDict: dict
        contains simulation paths
    variable: str
        output variable that is plotted (of whole simulation)

    Returns:
    -----------
    ax: matplotlib axis
        axis containing hillshade and output raster of simulation
    """
    pathInput = pathDict["avalancheDir"] / "Inputs"
    praPath = getRasterFile(pathInput / "REL")

    demDict = gI.readDEM(pathDict["avalancheDir"])
    dem = demDict["rasterData"]
    header = demDict["header"]
    cellSize = header["cellsize"]
    clabel = {
        "zdelta": "zDelta [m]",
        "fpTravelAngle": "travel angle [°]",
        "travelLength": "travel length [m]",
        "velocityMax": "velocity [m/s]",
    }

    file = getRasterFile(pathDict["pathToOutput"], variable=variable)
    rasterDict = rasterUtils.readRaster(file)
    raster = rasterDict["rasterData"]
    rasterPraDict = rasterUtils.readRaster(praPath)
    rasterPra = rasterPraDict["rasterData"]

    rowsMin, rowsMax, colsMin, colsMax = pU.constrainPlotsToData(raster, header["cellsize"], buffer=150)
    rowsMin = int(rowsMin)
    rowsMax = int(rowsMax)
    colsMin = int(colsMin)
    colsMax = int(colsMax)
    dataConstrained = raster[rowsMin : rowsMax + 1, colsMin : colsMax + 1]
    demConstrained = dem[rowsMin : rowsMax + 1, colsMin : colsMax + 1]
    praConstrained = rasterPra[rowsMin : rowsMax + 1, colsMin : colsMax + 1]

    data = np.ma.masked_where(dataConstrained == 0.0, dataConstrained)
    dataConstrained = np.ma.masked_where(dataConstrained == 0.0, dataConstrained)

    # set 0 and smaller to np.nan
    praConstrained = np.where(praConstrained > 0, 1.0, np.nan)

    # Set extent of peak file
    ny = data.shape[0]
    nx = data.shape[1]
    Ly = ny * cellSize
    Lx = nx * cellSize

    (extentCellCenters, extentCellCorners, rowsMinPlot, rowsMaxPlot, colsMinPlot, colsMaxPlot) = (
        pU.createExtent(rowsMin, rowsMax, colsMin, colsMax, header)
    )

    _, _ = pU.addHillShadeContours(ax, demConstrained, cellSize, extentCellCenters)

    extent = extentCellCenters
    extentPlot = [
        extent[0] - 0.5 * cellSize,
        extent[1] + 0.5 * cellSize,
        extent[2] - 0.5 * cellSize,
        extent[3] + 0.5 * cellSize,
    ]

    CS = ax.contour(
        demConstrained, levels=np.arange(0, 3500, 100), extent=extentPlot, colors="dimgrey", linewidths=0.5
    )
    ax.clabel(CS, CS.levels[::2], inline=True, fontsize=9)
    # dataOneColor = np.where(dataConstrained > 0.0, np.amax(data)*0.25, np.nan)
    colorsS = ["#FFCEF4", "#FFA7A8", "#C19A1B", "#578B21", "#007054", "#004960", "#201158"]
    cmapS = cmapCrameri.batlow.reversed()
    levels = 7
    bounds = np.linspace(
        np.nanmin(dataConstrained), np.nanmax(dataConstrained), levels + 1
    )  # Define boundaries
    norm = BoundaryNorm(bounds, ncolors=cmapS.N, clip=True)  # Create a norm based on the boundaries

    f = ax.imshow(
        dataConstrained,
        cmap=cmapS,
        norm=norm,
        extent=extentCellCorners,
        origin="lower",
        aspect="equal",
        zorder=4,
        alpha=0.7,
    )
    fig.colorbar(f, ax=ax, label=clabel[variable])
    if thalwegPra:
        cmapPra = ListedColormap(["magenta"])
        cmapPra.set_bad(color="none")
        # normPra = BoundaryNorm([0.5, 1.5], cmapPra.N)
        ax.imshow(
            praConstrained,
            cmap=cmapPra,
            extent=extentCellCorners,
            origin="lower",
            aspect="equal",
            zorder=3,
            alpha=0.5,
            interpolation="none",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return ax


def makeFieldPlot(ax, fig, pathDict, variable, xThalweg, yThalweg, dataThalweg, thalwegPra=False):
    """make a raster plot for FlowPy output

    Parameters
    -----------
    ax: matplotlib axis
        Axis for the plot
    fig: matplotlib figure
        Figure for the plot
    pathDict: dict
        contains simulation paths
    variable: str
        output variable that is plotted (of whole simulation)
    xThalweg: numpy array
        x coordinates of all thalwegs
    yThalweg: numpy array
        y coordinates of all thalwegs
    dataThalweg: dict
        profile of thalweg that is highlighted here
    thalwegPra: bool
        if True, PRA is coloured

    Returns
    -----------
    fig: matplotlib figure
        Figure containg the plot
    ax: matplotlib axis
        Axis containg the plot
    """
    centerOf = pathDict["titleVariables"]["centerOf"]
    ax = plotField(ax, fig, pathDict, variable, thalwegPra=thalwegPra)
    # ax.scatter(xThalweg, yThalweg, c="r", s=0.3, zorder=5, label=f"thalweg {centerOf}")
    # ax.scatter(xThalweg[0], yThalweg[0], c="b", s=2.0, zorder=6, label="startcell")
    for i, (x, y) in enumerate(zip(xThalweg, yThalweg)):
        ax.plot(x, y, "-", c="k", lw=0.5, zorder=7, label=f"thalweg {centerOf}" if i == 0 else None)
        ax.plot(dataThalweg["x"], dataThalweg["y"], "-", c="m", zorder=8)
    ax.legend()
    if thalwegPra:
        # add PRA path to existing legend
        handles, labels = ax.get_legend_handles_labels()
        praPath = Patch(facecolor="magenta", edgecolor="magenta", label="PRA", alpha=0.7)
        handles.append(praPath)
        labels.append(praPath.get_label())
        ax.legend(handles=handles, labels=labels)
    return fig, ax


def makeThalwegPlot(ax, dataThalweg, pathDict, centerOf=""):
    """make a 2D thalweg plot for FlowPy output

    Parameters
    -----------
    ax: matplotlib axis
        Axis for the plot
    dataThalweg: dict
        contains thalweg data:
        s or travelLength (np.array): travel length along thalweg
        z or altitude (np.array): altitude along thalweg
        zDelta (np.array): velocity altitude along thalweg
        alpha (float or int): input parameter of the simulation: alpha angle
        exp (float or int): input parameter of the simulation: exponent
        zDeltaMax (float or int): input parameter of the simulation: zDelta Maximum threshold
    pathDict: dict
        contains paths
    centerOf: str
        which center of is used (possible: '' (default),'CoE', 'CoZd', 'CoF')

    Returns
    -----------
    ax: matplotlib axis
        Axis containing the thalweg plot
    """
    # demDict = gI.readDEM(pathDict["avalancheDir"])
    # TODO: Check if flipping DEM is needed!(gI.readDem flips the raster.)
    # dem = demDict["rasterData"]
    # header = demDict["header"]
    # cellSize = header["cellsize"]

    x = np.array(dataThalweg["x"])
    y = np.array(dataThalweg["y"])
    z = np.array(dataThalweg["z"])
    s = np.array(dataThalweg["s"])

    file = getRasterFile(pathDict["pathToOutput"], variable="zdelta")
    zdelta = getThalwegValuesFromRaster(file, x, y)

    # only use these values that are within the avalanche path
    # and add those values from the corner
    indInPath = np.where(zdelta > 0)[0]
    if indInPath[0] > 0:
        indInPath = np.append(indInPath[0] - 1, indInPath)
    if (indInPath[-1] + 1) < len(zdelta):
        indInPath = np.append(indInPath, indInPath[-1] + 1)

    s = s[indInPath]
    s = s - s[0]
    zdelta = zdelta[indInPath]
    z = z[indInPath]

    # get FlowPy input parameter
    if "alpha" in dataThalweg.keys():
        alpha = dataThalweg["alpha"]
        exp = dataThalweg["exponent"]
        zDeltaMax = dataThalweg["zDeltaMax"]
    else:
        alpha = None
        exp = None
        zDeltaMax = None

    s_max = s[zdelta == max(zdelta)]
    z_max = z[zdelta == max(zdelta)]
    zdelta_max = zdelta[zdelta == max(zdelta)]

    # calculate tatsaechlicher runout angle
    angle_rad = np.arctan((max(z) - min(z)) / (max(s) - min(s)))
    angle_degrees = np.rad2deg(angle_rad)

    ds = max(s) - min(s)
    dh = ds * np.tan(np.deg2rad(alpha))

    ax.hlines(max(z) - dh, ds * 0.85, ds, colors="k", linestyles="dotted", linewidths=0.7)

    ax.plot(s, z, c="gray", linestyle="-", label="z")
    ax.plot(s, [d + z for d, z in zip(z, zdelta)], "r", label="$z^{vel}$")

    ax.vlines(
        s_max[0],
        z_max[0],
        z_max[0] + zdelta_max[0],
        label="$v_{max}$ = " + str(np.round(np.sqrt(zdelta_max[0] * 2 * 9.81), 1)) + " m/s",
    )
    ax.plot(
        [s[0], s[-1]],
        [z[0], z[-1]],
        color="lightgrey",
        linestyle="--",
        linewidth=1,
        label=rf"""$\alpha_{{eff}}$ = {np.round(angle_degrees, 1)}°""",
    )
    ax.plot(
        [0, ds],
        [max(z), max(z) - dh],
        "k--",
        linewidth=0.7,
        label=rf"""$\alpha_{{input}}$ = {np.round(alpha, 1)}°""" if alpha is not None else "",
    )
    ax.plot(
        [s[0], s[-1]],
        [min(z)] * 2,
        color="grey",
        linewidth=1,
        linestyle="--",
        label=rf"""$\Delta$s = {np.round(s[-1] - s[0], 1)} m""",
    )
    ax.vlines(
        x=0,
        ymin=z[-1],
        ymax=z[0],
        color="silver",
        linestyle="--",
        linewidth=1,
        label=(rf"$\Delta z = {np.round(z[0] - z[-1], 1)}$$m$"),
    )

    # ax.text(s_max[0] + 1, z_max[0] + zdelta_max[0]/2, '$v_{max}$ = ' + str(np.round(np.sqrt(zdelta_max[0] * 2 * 9.81),1)) + ' m/s', va = 'center')
    # ax.text((max(s)/5*4), min(z) + (max(z) - min(z)) / 22, fr'{angle_degrees:.1f}°', fontsize=11, ha='center')
    # ax.text((ds*0.88), (max(z)-dh) * 1.05, fr'{alpha:.1f}°', fontsize=11, ha='center')
    ax.set(xlabel="$s_{xy}$ [m]")
    ax.set(ylabel="elevation [m]")
    ax.legend()

    ax.text(
        max(s) * 0.5,
        max(z) * 0.95,
        (
            f"""model parameters: \n alpha: {alpha}° \n exp: {np.round(exp, 1)} \n $Z^{{vel}}_{{max}}$: {np.round(zDeltaMax, 1)} m \n $v_{{max}}$: {round(np.sqrt(zDeltaMax * 2 * 9.81), 1)} m/s"""
            if alpha is not None
            else ""
        ),
        va="top",
        ha="left",
    )

    return ax


def getThalwegValuesFromRaster(rasterFile, x, y):
    """
    project thalweg coordinates to raster and extract values

    Parameters
    ----------
    rasterFile : str
        path to raster file
    x: np array
        x coordinates of thalweg
    y: np array
        y coordinates of thalweg

    Returns
    ------------
    thalwegValues: np array
        values along thalweg read from raster
    """
    rasterDict = rasterUtils.readRaster(rasterFile)
    header = rasterDict["header"]
    rasterValues = rasterDict["rasterData"]
    rasterValues = np.where(rasterValues > 0, rasterValues, 0)

    thalwegValues, _ = gT.projectOnGrid(
        x,
        y,
        rasterValues,
        csz=header["cellsize"],
        xllc=header["xllcenter"],
        yllc=header["yllcenter"],
    )
    return thalwegValues


def savePickle(profileExtended, inFileName):
    """
    save dictionary as pickle file, the filename is modified with an "extended"

    Parameters
    ----------
    profileExtended : dict
        dictionary that is saved
    inFileName : pathlib Path
        file name that is modified with an "extended"
    """
    dir = inFileName.parent
    fileName = inFileName.stem
    outFileName = dir / f"extended_{fileName}.pickle"

    with open(outFileName, "wb") as handle:
        pickle.dump(profileExtended, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plotBoxplot(pathDict, cfg, title=""):
    """
    shows and potentially saves Violinplot and Boxplot

    Parameters:
    -----------
    path: str
        OutputPath of the FlowPy simulation
    dataNan: np.array
        data that is analysed and plotted (can contain nans)
    title: str
        title for the plot
    """
    path = pathDict["pathToOutput"]
    cfgGen = cfg["GENERAL"]
    cfgSize = cfg["SIZECLASS"]
    varName = cfgGen.get("statisticVariable")
    centerOf = cfgGen.get("centerOfVariable")
    ylabel = getYlabelBoxplot(varName)

    dataNan = getDataBoxplots(path, varName, centerOf)

    data = np.delete(dataNan, np.where(np.isnan(dataNan)))
    fig, ax2 = plt.subplots()  # figsize = [4,5])
    # fig.tight_layout()
    labels = [f" (n = {len(data)})"]
    if cfgGen.getboolean("plotLogScale"):
        ax2.set_yscale("log")
    ax2.violinplot([data])
    ax2.boxplot([data], whis=0, widths=0.07, showfliers=False, medianprops={"color": "blue"})
    ax2.set_xticks(np.arange(1, len(labels) + 1), labels=labels, fontsize=13)
    ax2.set_xlim(0.25, len(labels) + 0.75)

    # Color background
    if varName in ["travelLengthMax", "impressure"]:

        ysize1Max = cfgSize.getint(f"{varName}Size1Max")
        ysize2Max = cfgSize.getint(f"{varName}Size2Max")
        ysize3Max = cfgSize.getint(f"{varName}Size3Max")
        ysize4Max = cfgSize.getint(f"{varName}Size4Max")
        y_min, y_max = ax2.get_ylim()
        y_max = np.max([y_max, 1.1 * ysize4Max])
        ax2.axhspan(0, ysize1Max, facecolor="#" + cfgSize["colorSize1"], alpha=0.2)  # Avalanche size 1
        ax2.axhspan(
            ysize1Max,
            ysize2Max,
            facecolor="#" + cfgSize["colorSize2"],
            alpha=0.2,
        )  # size 2
        ax2.axhspan(
            ysize2Max,
            ysize3Max,
            facecolor="#" + cfgSize["colorSize3"],
            alpha=0.2,
        )  # size 3
        ax2.axhspan(
            ysize3Max,
            ysize4Max,
            facecolor="#" + cfgSize["colorSize4"],
            alpha=0.2,
        )  # size 4
        ax2.axhspan(ysize4Max, y_max, facecolor="#" + cfgSize["colorSize5"], alpha=0.2)  # size 5

        if varName == "impressure":
            class_lab = "$C_{ip}$"
        # elif varName == "path_area":
        #   class_lab = "$B_{aa}$"
        elif varName == "travelLengthMax":
            class_lab = "$E_{rl}$"
        else:
            class_lab = ""

        ax2.text(
            1.5,
            0 + (ysize1Max * 0.75),
            f"{class_lab} 1",
            ha="center",
            va="center",
            color="#008B8B",
            fontsize=13,
        )
        ax2.text(
            1.5,
            ysize1Max + (ysize2Max - ysize1Max) / 2,
            f"{class_lab} 2",
            ha="center",
            va="center",
            color="#4682B4",
            fontsize=13,
        )
        ax2.text(
            1.5,
            ysize2Max + (ysize3Max - ysize2Max) / 2,
            f"{class_lab} 3",
            ha="center",
            va="center",
            color="#6495ED",
            fontsize=13,
        )
        ax2.text(
            1.5,
            ysize3Max + (ysize4Max - ysize3Max) / 2,
            f"{class_lab} 4",
            ha="center",
            va="center",
            color="#CD5C5C",
            fontsize=13,
        )
        ax2.text(
            1.5,
            ysize4Max + (y_max - ysize4Max) / 2,
            f"{class_lab} 5",
            ha="center",
            va="center",
            color="#B22222",
            fontsize=13,
        )
        ax2.set_yticks([ysize1Max, ysize2Max, ysize3Max, ysize4Max])
        ax2.set_yticklabels([ysize1Max, ysize2Max, ysize3Max, ysize4Max], fontsize=13)

    plt.ylabel(ylabel, fontsize=13)

    if title == "":
        title = f"thalwege {centerOf}"
    plt.title(title)
    plt.grid(True)
    savePath = pathDict["savePath"]
    simhash = pathDict["titleVariables"]["simHash"]
    fig.savefig(f"{savePath}/ThalwegStatistic_{simhash}_{varName}{centerOf}.png")


def getDataBoxplots(path, variable, centerOf):
    """
    get the thalweg data

    Parameters:
    -----------
    path: pathlib Path
        OutputPath of the FlowPy simulation
    variable: str
        name of output variable that is analysed and plotted (e.g, impressure, travelLengthMax)

    Returns:
    -----------
    data: numpy array
        maximum value of the parameter varName of all thalwegs
    """

    data = ""

    if variable == "velocity":
        varName = "velocity"
        variable = "zdelta"

    elif variable == "impressure":
        varName = "impressure"
        variable = f"zdelta"
    else:
        varName = f"{variable}"

    dataDict = maxParameterOfAllThalwegs(path, variable, centerOf)
    data = np.array(dataDict[variable])
    if varName == "velocity":
        data = zDelta2velocity(data)

    if varName == "impressure":
        velo = zDelta2velocity(data)

        rho = 200  # km m-3
        data = rho * velo**2 * 1e-3

    return data


def maxParameterOfAllThalwegs(path, variableList, centerOf):
    """
    get thalweg data (maximum per thalweg)

    Parameters:
    -----------
    path: pathlib Path
        Thalweg-Output Path of the FlowPy simulation
    variable: str
        name of thalweg parameter

    Returns:
    -----------
    variableValues: list
        maximum values of the parameter variable of all thalwegs
    """
    if type(variableList) == str:
        variableList = [variableList]
    variableValues = {}
    for variable in variableList:
        variableValues[variable] = []
        for filename in os.listdir(path / "thalwegData"):
            # Check if the filename starts with 'thalweg'
            if filename.startswith(f"thalwegData_{centerOf}"):
                # Construct full file path
                filePath = path / "thalwegData" / filename
                data = np.load(filePath, allow_pickle="TRUE")
                x = data["x"]
                y = data["y"]

                outputRasterFile = getRasterFile(path, variable=variable)
                valuesThalweg = getThalwegValuesFromRaster(outputRasterFile, x, y)
                valueMax = np.nanmax(valuesThalweg)
                variableValues[variable].append(valueMax)

                # for plotting averaged values:
                # variableValues[variable].append(np.nanmax(data[variable]))
    return variableValues


def getYlabelBoxplot(variable):
    """
    return ylabel

    Parameters:
    --------------
    variable: str
        name of thalweg parameter that is plotted

    Returns:
    --------------
    ylabel: str
        ylabel for plot
    """

    if variable == "velocity":
        ylabel = "max. velocity [m/s]"
    elif variable == "impressure":
        ylabel = "max. impact pressure [kPa]"
    elif variable == "travelLengthMax":
        ylabel = "runout length [m]"
    elif variable == "zDelta":
        ylabel = "max. zDelta [m]"
    elif variable == "flux":
        ylabel = "flux"
    else:
        message = f"{variable} is not a valid thalweg variable for the statistic boxplot"
        log.error(message)
        raise ValueError(message)
    return ylabel
