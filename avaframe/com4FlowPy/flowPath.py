import numpy as np
import pickle
import logging

import avaframe.ana5Utils.DFAPathGeneration as DFAPathGeneration
from avaframe.in3Utils import cfgUtils
from avaframe.ana5Utils import DFAPathGeneration
import avaframe.in3Utils.geoTrans as gT

log = logging.getLogger(__name__)


class Path:
    """Class contains a path, containing one startcell and corresponding child cells"""

    def __init__(self, dem, startcellRow, startcellCol, genList, rasterAttributes, countArray, relId=None):
        """initializes a GMF path, that belongs to a startcell

        Parameters
        ----------
        dem: numpy array
            Digital elevation model
        startcellRow: int
            Row index of startcell
        startcellCol: int
            Column index of startcell
        genList: list
            contains all cells that belong to the path (per generation an extra list)
        rasterAttributes: dict
            contains information about the input rasters
        """
        self.dem = dem
        self.cellsize = rasterAttributes["cellsize"]
        self.xllcorner = rasterAttributes["xllcenter"] - self.cellsize / 2
        self.yllcorner = rasterAttributes["yllcenter"] - self.cellsize / 2
        self.xllcenter = rasterAttributes["xllcenter"]
        self.yllcenter = rasterAttributes["yllcenter"]
        self.nrows = rasterAttributes["nrows"]
        # self.crs = rasterAttributes["crs"]

        self.alpha = genList[0][0].alpha
        self.exp = genList[0][0].exp
        self.maxZDelta = genList[0][0].max_z_delta
        self.genList = genList
        self.startcellRow = startcellRow
        self.startcellCol = startcellCol
        self.numberGen = len(genList)
        self.relId = int(relId)
        self.pathRaster = np.where(countArray > 0, countArray, np.nan)

        self.dropHeight = 0
        self.travelLength = 0

        self.zDeltaGeneration = []
        self.fluxGeneration = []
        self.depFluxGeneration = []
        self.travelLengthGeneration = []
        self.flowEnergyGeneration = []
        self.rowGeneration = []
        self.colGeneration = []
        self.altitudeGeneration = []
        self.gammaGeneration = []
        # self.pathArea = 0
        self.flux_gen = []

        self.zDeltaArray = np.zeros_like(self.dem, dtype=np.float32)
        self.flowEnergyArray = np.zeros_like(self.dem, dtype=np.float32)
        self.fluxArray = np.zeros_like(self.dem, dtype=np.float32)
        self.routFluxSumArray = np.zeros_like(self.dem, dtype=np.float32)
        self.depFluxSumArray = np.zeros_like(self.dem, dtype=np.float32)
        self.cfgPathGen = cfgUtils.getModuleConfig(DFAPathGeneration)

        """
        self.travel_length_array = np.zeros_like(self.dem, dtype=np.float32)
        self.generation_array = np.full_like(self.dem, np.nan, dtype=np.float32)
        """

    def indizesToDFACoords(self, cols, rows):
        """calculates the row and column indices to the x and y coordinates

        Parameters
        ----------
        cols: numpy array
            column indices of cells belonging to path
        rows: numpy array
            row indices of cells belonging to path

        Returns
        ----------
        x: numpy array
            x coordinates (in m) of cells belonging to path
        y: numpy array
            y coordinates (in m) of cells belonging to path
        """
        x = cols * self.cellsize + self.xllcorner
        y = self.yllcorner + rows * self.cellsize
        return x, y

    def updateYCoord(self, yDFA):
        """
        for original avaframe com1DFA calculations the y coordinates are flipped
        (rasters are read with flipud -> upside down), so we need to flip the y coordines
        after the DFA computations

        Parameters
        ------------
        yDFA: numpy array
            y coodinates that need to be flipped

        Returns
        -----------
        y: numpy array
            flipped y coordinates
        """
        rows = (yDFA - self.yllcorner) / self.cellsize
        y = self.yllcorner + (self.nrows - rows) * self.cellsize
        return y

    def getVariablesGeneration(self):
        """write lists with size and format of genList containing specific parameters
        (the main list contains lists for every generation)
        TODO: only calculate 'important'/output arrays
        """
        for cellList in self.genList:
            cellListZDelta = []
            cellListFlux = []
            cellListMinDistance = []
            cellListFlowEnergy = []
            cellListRow = []
            cellListCol = []
            cellListAlt = []
            cellListGamma = []
            cellListDepFlux = []

            for cell in cellList:
                cellListZDelta.append(cell.z_delta)
                cellListFlux.append(cell.flux)
                cellListDepFlux.append(cell.fluxDep)
                cellListMinDistance.append(cell.min_distance)
                cellListFlowEnergy.append(cell.flowEnergy)
                cellListRow.append(cell.rowindex)
                cellListCol.append(cell.colindex)
                cellListAlt.append(cell.altitude)
                cellListGamma.append(cell.max_gamma)
                # cellListFlux_gen.append(cell.flux_generation)

            self.zDeltaGeneration.append(cellListZDelta)
            self.fluxGeneration.append(cellListFlux)
            self.depFluxGeneration.append(cellListDepFlux)
            self.travelLengthGeneration.append(cellListMinDistance)
            self.flowEnergyGeneration.append(cellListFlowEnergy)
            self.rowGeneration.append(cellListRow)
            self.colGeneration.append(cellListCol)
            self.altitudeGeneration.append(cellListAlt)
            self.gammaGeneration.append(cellListGamma)
            # self.flux_gen.append(cellListFlux_gen)

    def getGenerationList(self, variable, generation=None):
        """write lists with size and format of genList containing specific parameters
        (the main list contains lists for every generation)

        Parameters
        -----------
        variable: string
            for the variable is the generation list created
        generation: int
            generation that is extracted (if None, all generations are added)
        """

        variableGeneration = []
        if generation is None:
            for cellList in self.genList:
                listVariable = self.getListFromCellList(cellList, variable)
                variableGeneration.append(listVariable)
        else:
            cellList = self.genList[generation]
            variableGeneration = self.getListFromCellList(cellList, variable)
        return variableGeneration

    def getListFromCellList(self, cellList, variable):
        listVariable = []

        for cell in cellList:
            if variable == "zDelta":
                listVariable.append(cell.z_delta)
            elif variable == "flux":
                listVariable.append(cell.flux)
            elif variable in ["travelLength", "s"]:
                listVariable.append(cell.min_distance)
            elif variable in ["altitude", "z"]:
                listVariable.append(cell.altitude)
            elif variable == "row":
                listVariable.append(cell.rowindex)
            elif variable == "col":
                listVariable.append(cell.colindex)
            elif variable == "gamma":
                listVariable.append(cell.max_gamma)
            elif variable == "flowEnergy":
                listVariable.append(cell.flowEnergy)
            else:
                log.error(f"variable {variable} can not be computed to a generation list")
        return listVariable


    def getPathArrays(self):
        """write arrays with size of dem containing the maximum of the variable values of every path
        value 0 means, the path does not hit the cell
        TODO: only calculate 'important'/output arrays
        """
        for gen, cellList in enumerate(self.genList):
            for cell in cellList:
                self.zDeltaArray[cell.rowindex, cell.colindex] = max(
                    self.zDeltaArray[cell.rowindex, cell.colindex], cell.z_delta
                )
                self.flowEnergyArray[cell.rowindex, cell.colindex] = max(
                    self.flowEnergyArray[cell.rowindex, cell.colindex], cell.flowEnergy
                )
                self.fluxArray[cell.rowindex, cell.colindex] = max(
                    self.fluxArray[cell.rowindex, cell.colindex], cell.flux
                )
                self.routFluxSumArray[cell.rowindex, cell.colindex] += cell.flux
                self.depFluxSumArray[cell.rowindex, cell.colindex] += cell.fluxDep

                """
                self.travel_length_array[cell.rowindex, cell.colindex] = max(self.travel_length_array[cell.rowindex, cell.colindex], cell.min_distance)
                self.generation_array[cell.rowindex, cell.colindex] = gen
                """

    def calcThalwegCenterof(self, variable, variableCo):
        """calculates for a specific variable the center of a specific variable (thalweg)

        Parameters
        ----------
        variable: list
            variable, which is centered (in format genList)
        variableCo: list
            center of variableCo is calculated (variable is weighted) (in format genList)

        Returns
        ----------
        variableSum: numpy array
            sum of variable per generation
        coVar: numpy array
            centered variable (per generation)
        """
        coVar = np.zeros(len(self.genList))
        variableSum = np.zeros(len(self.genList))
        for gen in range(0, len(self.genList)):
            var = np.array(variable[gen])
            co = np.array(variableCo[gen])
            variableSum[gen] = np.sum(var)
            variableCoSum = np.sum(co)
            if variableCoSum > 0:  # flow_energy and zdelta are 0 in generation 0
                # coVar[gen] = 1 / variableCoSum * np.sum(var * co)
                coVar[gen] = np.average(var, weights=co)
            else:
                # TODO: does this makes sense??
                coVar[gen] = np.average(var)
        return variableSum, coVar

    def getCenterofs(self, variables, centerOfs):
        """
        calculate sum of variable for every iteration step/ generation and
        center of energy, flux and zDelta for the following variables:

        Parameters
        ----------
        variables: list
            List of variables that should be weighted (with center of energy and flux)
        """

        #self.getVariablesGeneration()

        for varName in variables:
            if varName in [
                "s",
                "z",
                "x",
                "y",
                "flowEnergyArray",
                "zDeltaArray",
                "fluxArray",
                "routFluxSumArray",
                "depFluxSumArray",
            ]:
                continue
            if varName == "depFluxSum":
                variables.append("depFlux")
                continue
            if varName == "fluxSum":
                variables.append("flux")
                continue

            values = self.getGenerationList(varName)

            if "CoE" in centerOfs:
                self.energyGenList = self.getGenerationList("flowEnergy")
                sumE, coE = self.calcThalwegCenterof(
                    values, self.energyGenList)
                # TODO: zdelta is 0 in generation 1, so the first value does not make sense
                setattr(self, f"{varName}CoE", coE[1:])
            if "CoF" in centerOfs:
                self.fluxGenList = self.getGenerationList("flux")
                sumF, coF = self.calcThalwegCenterof(
                    values, self.fluxGenList)
                setattr(self, f"{varName}CoF", coF)
            if "CoZd" in centerOfs:
                self.zDeltaGenList = self.getGenerationList("zDelta")
                sumZd, coZd = self.calcThalwegCenterof(
                    values, self.zDeltaGenList)
                setattr(self, f"{varName}CoZd", coZd)
        # for saving RAM, empty the lists
        self.energyGenList = []
        self.fluxGenList = []
        self.zDeltaGenList = []

        #values = getattr(self, f"{varName}Generation")
        '''
        sumF, coF = self.calcThalwegCenterof(
            values, self.fluxGeneration
        )  # center of flux of every variable
        sumE, coE = self.calcThalwegCenterof(
            values, self.flowEnergyGeneration
        )  # center of energy of every variable
        sumZd, coZd = self.calcThalwegCenterof(
            values, self.zDeltaGeneration
        )  # center of energy of every variable

        setattr(self, f"{varName}SumCoE", sumE)
        setattr(self, f"{varName}CoF", coF)
        # TODO: zdelta is 0 in generation 1, so the first value does not make sense
        setattr(self, f"{varName}CoE", coE[1:])
        setattr(self, f"{varName}CoZd", coZd[1:])
        '''

    def calcAlphaEff(self, s, z):
        """Compute the effective alpha angle of the thalweg"""
        dz = z[0] - z[-1]
        ds = s[-1] - s[0]
        if ds > 0:
            alphaEff = np.rad2deg(np.arctan(dz / ds))
        else:
            alphaEff = np.nan
        return alphaEff

    def DFAextendTop(self, co):
        """
        extend the thalweg to top of the release area using the function of
        ana5Utils.DFAPathGeneration
        and update the thalweg values
        """
        demDict = {
            "rasterData": self.dem,
            "header": {
                "cellsize": self.cellsize,
                "xllcorner": self.xllcorner,
                "yllcorner": self.yllcorner,
                "xllcenter": self.xllcenter,
                "yllcenter": self.yllcenter,
            },
        }
        extTopOption = self.cfgPathGen["PATH"].getint("extTopOption")
        colGen0 = self.getGenerationList("col", generation=0)
        rowGen0 = self.getGenerationList("row", generation=0)
        zGen0 = self.getGenerationList("altitude", generation=0)

        xIni, yIni = self.indizesToDFACoords(
            np.asarray(colGen0), np.asarray(rowGen0)
        )
        particlesIni = {"x": xIni, "y": yIni, "z": np.asarray(zGen0)}
        profile = {
            "x": getattr(self, f"x{co}"),
            "y": getattr(self, f"y{co}"),
            "z": getattr(self, f"altitude{co}"),
            "s": getattr(self, f"travelLength{co}"),
            "zDelta": getattr(self, f"zDelta{co}"),
            "flux": getattr(self, f"flux{co}"),
            "flowEnergy": getattr(self, f"flowEnergy{co}"),
            "indStartMassAverage": 1, # after the top extension!
        }
        # do not use the last, because at the end there are some weird direction
        profile["indEndMassAverage"] = np.size(profile["x"])
        # remember the coordinates of the averaged thalweg without extensions
        self.startThalweg = {"x": profile["x"][0], "y": self.updateYCoord(profile["y"][0]), "z": profile["z"][0], "s": profile["s"][0]}
        self.endThalweg = {"x": profile["x"][-1], "y": self.updateYCoord(profile["y"][-1]), "z": profile["z"][-1], "s": profile["s"][-1]}

        profile = DFAPathGeneration.extendProfileTop(extTopOption, particlesIni, profile)

        self.setThalwegDataFromDict(profile, co)
        return profile

    def DFAextendTopWILD(self, co):
        demUDDict = {
            "rasterData": self.dem * (-1),
            "header": {
                "cellsize": self.cellsize,
                "xllcorner": self.xllcorner,
                "yllcorner": self.yllcorner,
                "xllcenter": self.xllcenter,
                "yllcenter": self.yllcenter,
            },
        }
        profileUD = {
            "x": getattr(self, f"x{co}")[::-1],
            "y": getattr(self, f"y{co}")[::-1],
            "z": (getattr(self, f"altitude{co}")[::-1]) * (-1),
            "s": getattr(self, f"travelLength{co}")[::-1],
            "indStartMassAverage": 1,
        }
        profileUD["indEndMassAverage"] = len(profileUD["x"]) + 1

        profileUD = DFAPathGeneration.extendProfileBottom(
            self.cfgPathGen["PATH"], demUDDict, profileUD, considerLLC=True
        )
        profileUD["s"][-1] = 0
        #profileUD = self.findLastPointInRaster(profileUD)
        #profileUD = self.findBottomPointInPath(profileUD)

        profile = {"indStartMassAverage": 0, "indEndMassAverage": profileUD["indEndMassAverage"]}
        for variable in ["x", "y", "z", "s"]:
            profile[variable] = profileUD[variable][::-1]
        profile["z"] = profile["z"] * (-1)

        self.setThalwegDataFromDict(profile, co)

        changedLen = len(profile["x"]) - len(getattr(self, f"zDelta{co}"))
        if changedLen > 0:
            # TODO: only compute when they are in the variable list?
            zDelta = np.append(np.zeros(changedLen), getattr(self, f"zDelta{co}"))
            setattr(self, f"zDelta{co}", zDelta)
            flux = np.append(np.ones(changedLen), getattr(self, f"flux{co}"))
            setattr(self, f"flux{co}", flux)
            flowEnergy = np.append(np.zeros(changedLen), getattr(self, f"flowEnergy{co}"))
            setattr(self, f"flowEnergy{co}", flowEnergy)
            # fluxSum = np.append(np.ones(changedLen), getattr(self, f"fluxSum{co}"))
            # setattr(self, f"fluxSum{co}", fluxSum)

        return profile


    def DFAextendBottom(self, co, profile):

        # TODO: use DFAPatheneratin.extendDFAPath()?
        demDict = {
            "rasterData": self.dem,
            "header": {
                "cellsize": self.cellsize,
                "xllcorner": self.xllcorner,
                "yllcorner": self.yllcorner,
                "xllcenter": self.xllcenter,
                "yllcenter": self.yllcenter,
            },
        }
        # extend the bottom quite far
        profile = DFAPathGeneration.extendProfileBottom(
            self.cfgPathGen["PATH"], demDict, profile, considerLLC=True
        )

        profileResample = profile.copy()
        profileResample = DFAPathGeneration.resamplePath(self.cfgPathGen["PATH"], demDict, profileResample)
        profile = self.replaceResampledProfileCore(profile, profileResample)

        profile = self.findLastPointInRaster(profile)

        profile = self.findBottomPointInPath(profile)

        self.setThalwegDataFromDict(profile, co)

        return profile

    def replaceResampledProfileCore(self, profile, profileResample):
        """
        for all variables (x, y, s, z, zdelta, fluxSum, flowEnergy)
        use the resampled top and bottom (extended) values and the original vlaues inbetween.
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
            elif key in ["zDelta", "flowEnergy"]:
                keepCore = profile[key][1:] # dont use the first value because it is 0
                resampledTop = np.linspace(0,keepCore[0], lenExtTop + 1, endpoint=False)

                resampledBottom = np.linspace(keepCore[-1],0, lenExtBot, endpoint=False)
            elif key in ["fluxSum", "flux"]:
                keepCore = profile[key]
                resampledTop = np.linspace(1, keepCore[0], lenExtTop, endpoint=False)
                resampledBottom = np.linspace(keepCore[-1], 0, lenExtBot, endpoint=False)
            else:
                resampledTop = profileResample[key][0:indStartRes]
                resampledBottom = profileResample[key][indEndRes:]
                keepCore = profile[key][indStart:indEnd]
            profile[key] = np.concatenate((resampledTop, keepCore, resampledBottom))

        return profile

    def pathExtension(self, co):
        """
        thalweg extension to top and bottom of path
        """

        profile = self.DFAextendTop(co)
        profile = self.DFAextendBottom(co, profile)
        self.indexStartThalweg = profile["indStartMassAverage"]
        self.indexEndThalweg = profile["indEndMassAverage"]

        # update y coordinate from upside down to right direction
        yUpdate = self.updateYCoord(getattr(self, f"y{co}"))
        setattr(self, f"y{co}", yUpdate)


    def setThalwegDataFromDict(self, profile, co):

        for variable in profile.keys():
            if variable in ["indStartMassAverage", "indEndMassAverage"]:
                continue
            if variable == "s":
                setattr(self, f"travelLength{co}", profile["s"])
            if variable == "z":
                setattr(self, f"altitude{co}", profile["z"])
            setattr(self, f"{variable}{co}", profile[variable])


    def findLastPointInRaster(self, profile):
        values, _ = gT.projectOnGrid(
            profile["x"],
            profile["y"],
            self.pathRaster,
            csz=self.cellsize,
            xllc=self.xllcenter,
            yllc=self.yllcenter,
        )

        indexInRaster = np.where(values > 0)[0]
        # keep also the last point
        indexInRaster = np.append(indexInRaster, len(profile["x"]) - 1)
        for variable in profile.keys():
            if variable in ["indStartMassAverage", "indEndMassAverage"]:
                continue
            profile[variable] = profile[variable][indexInRaster]
        # correct s because the upper part can be outside the raster
        profile["s"] = profile["s"] - profile["s"][0]
        return profile

    def findBottomPointInPath(self, profile, tol=1, max_iter=100):
        """
        A, B : (x, y)
        A must be raster > 0
        B must be raster <= 0 or nan

        returns (x, y)
        """

        ax = np.asarray([profile["x"][-2]])
        ay = np.asarray([profile["y"][-2]])
        bx = np.asarray([profile["x"][-1]])
        by = np.asarray([profile["y"][-1]])

        valueA, _ = gT.projectOnGrid(
            ax,
            ay,
            self.pathRaster,
            csz=self.cellsize,
            xllc=self.xllcenter,
            yllc=self.yllcenter,
        )

        valueB, _ = gT.projectOnGrid(
            bx,
            by,
            self.pathRaster,
            csz=self.cellsize,
            xllc=self.xllcenter,
            yllc=self.yllcenter,
        )

        for _ in range(max_iter):

            mx = 0.5 * (ax + bx)
            my = 0.5 * (ay + by)

            valueM, _ = gT.projectOnGrid(
                mx,
                my,
                self.pathRaster,
                csz=self.cellsize,
                xllc=self.xllcenter,
                yllc=self.yllcenter,
            )

            # treat nan as outside
            if np.isfinite(valueM) and valueM > 0:
                ax, ay = mx, my
            else:
                bx, by = mx, my

            if np.sqrt((ax - bx)**2 + (ay - by)**2) < tol:
                break

        profile["x"][-1] = ax
        profile["y"][-1] = ay
        profile["z"][-1], _ = gT.projectOnGrid(
                ax,
                ay,
                self.dem,
                csz=self.cellsize,
                xllc=self.xllcenter,
                yllc=self.yllcenter,
            )
        ds = np.sqrt((profile["x"][-2] - ax) ** 2 + (profile["y"][-2] - ay) ** 2)
        profile["s"][-1] = profile["s"][-2] + ds

        return profile

    def updateTravelLengthTopExtension(self, rowThalweg, colThalweg, sThalweg):
        """
        update travel length thalweg with the top-extension

        Parameters
        ------------
        rowThalweg: numpy array
            row values of thalweg
        colThalweg: numpy array
            col values of thalweg
        sThalweg: numpy array
            travel length (s values) projected into the horizontal of thalweg

        Returns
        ------------
        sThalweg: numpy array
            updated travel length (second index) considering that the thalweg was extended to the rop of the release area
        """
        deltaRow = rowThalweg[1] - rowThalweg[0]
        deltaCol = colThalweg[1] - colThalweg[0]
        # compute deltaS in meters
        deltaS = np.sqrt((deltaRow * self.cellsize) ** 2 + (deltaCol * self.cellsize) ** 2)
        sThalweg[1:] += deltaS
        return sThalweg

    def saveDict(self, saveDir, centerOfs, variables):
        """
        save thalweg data. (One file per thalweg)

        Parameters
        ------------
        saveDir: pathlib.PosixPath
            directory, in which the thalweg data is saved
        centerOfs: list
            contains the center-of-variable names that are saved
        variables: list
            contains the variable names that are saved
        """
        thalwegData = {
            "alpha": round(self.alpha, 1),
            "exponent": self.exp,
            "zDeltaMax": round(self.maxZDelta, 1),
            # 'crs': self.crs,
            "numberGen": self.numberGen,
            "startAverageData": self.startThalweg,
            "endAverageData": self.endThalweg,
            "indexStartAverageData": self.indexStartThalweg,
            "indexEndAverageData": self.indexEndThalweg,
        }
        variables = variables
        centerOfs = centerOfs

        for co in centerOfs:
            for varName in variables:
                if varName in [
                    "flowEnergyArray",
                    "zDeltaArray",
                    "fluxArray",
                    "routFluxSumArray",
                    "depFluxSumArray",
                ]:
                    if np.any(getattr(self, f"{varName}")) == False:
                        self.getPathArrays()
                    value = getattr(self, f"{varName}")
                elif varName == "z":
                    value = getattr(self, f"altitude{co}")
                elif varName == "s":
                    value = getattr(self, f"travelLength{co}")
                else:
                    value = getattr(self, f"{varName}{co}")
                thalwegData[f"{varName}"] = value

            if "travelLength" in variables and "altitude" in variables:
                # compute and save effective alpha angle
                alpha = self.calcAlphaEff(getattr(self, f"travelLength{co}"), getattr(self, f"altitude{co}"))
                thalwegData[f"alphaEff"] = alpha

            # output file name and save teh pickle file
            if self.relId is None:
                outName = f"thalwegData_{co}_{self.startcellRow}_{self.startcellCol}.pickle"
            else:
                outName = f"thalwegData_{co}_{self.relId}.pickle"
            with open(saveDir / (outName), "wb") as handle:
                pickle.dump(thalwegData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def calcAndSaveThalwegData(self, thalwegParameters):
        """main function for paths & thalwegs: calculates the thalweg and saves the data

        Parameters:
        ------------
        thalwegParameters: dict
            contains information to calculate and save the thalweg data (from .ini file)
        """
        saveDir = thalwegParameters["thalwegDir"]
        cos = eval(thalwegParameters["thalwegCenterOf"])
        variables = eval(thalwegParameters["thalwegVariables"])
        centerOfs = []
        for co in cos:
            co.lower()
            if co in ["energy", "coe"]:
                centerOf = "CoE"
            elif co in ["flux", "cof"]:
                centerOf = "CoF"
            elif co in ["zdelta", "cozd"]:
                centerOf = "CoZd"
            centerOfs.append(centerOf)

        if "s" in variables:
            variables.append("travelLength")
        if "z" in variables:
            variables.append("altitude")
        if "x" in variables or "y" in variables:
            variables.append("col")
            variables.append("row")

        self.getCenterofs(variables, centerOfs)
        for co in centerOfs:
            # convert column and row to coordinates s, y
            x, y = self.indizesToDFACoords(getattr(self, f"col{co}"), getattr(self, f"row{co}"))
            setattr(self, f"x{co}", x)
            setattr(self, f"y{co}", y)
            self.pathExtension(co)
            # sUpdate = self.updateTravelLengthTopExtension(getattr(self, f"row{co}"), getattr(self, f"col{co}"), getattr(self, f"travelLength{co}"))
            # setattr(self, f"s{co}", sUpdate)
            # setattr(self, f"travelLength{co}", sUpdate)
        self.saveDict(saveDir, centerOfs, variables)
