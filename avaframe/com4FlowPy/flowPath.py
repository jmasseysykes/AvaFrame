import numpy as np
import pickle
import logging

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
        self.nrows = rasterAttributes["nrows"]
        self.rasterAttributes = rasterAttributes

        self.alpha = genList[0][0].alpha
        self.exp = genList[0][0].exp
        self.maxZDelta = genList[0][0].max_z_delta
        self.genList = genList
        self.startcellRow = startcellRow
        self.startcellCol = startcellCol
        self.numberGen = len(genList)
        self.relId = int(relId)
        self.pathRaster = np.where(countArray > 0, countArray, np.nan)

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

        """
        self.travel_length_array = np.zeros_like(self.dem, dtype=np.float32)
        self.generation_array = np.full_like(self.dem, np.nan, dtype=np.float32)
        """

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

        # self.getVariablesGeneration()

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
                sumE, coE = self.calcThalwegCenterof(values, self.energyGenList)
                # TODO: zdelta is 0 in generation 1, so the first value does not make sense
                setattr(self, f"{varName}CoE", coE[1:])
            if "CoF" in centerOfs:
                self.fluxGenList = self.getGenerationList("flux")
                sumF, coF = self.calcThalwegCenterof(values, self.fluxGenList)
                setattr(self, f"{varName}CoF", coF)
            if "CoZd" in centerOfs:
                self.zDeltaGenList = self.getGenerationList("zDelta")
                sumZd, coZd = self.calcThalwegCenterof(values, self.zDeltaGenList)
                setattr(self, f"{varName}CoZd", coZd)
        # for saving RAM, empty the lists
        self.energyGenList = []
        self.fluxGenList = []
        self.zDeltaGenList = []

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
            else:
                message = f"{co} is a not valid thalweg parameter"
                log.error(message)
                raise ValueError(message)
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
            x, y = gT.indicesToCoords(
                getattr(self, f"col{co}"), getattr(self, f"row{co}"), self.rasterAttributes
            )
            setattr(self, f"x{co}", x)
            setattr(self, f"y{co}", y)
            # update y coordinate
        self.saveDict(saveDir, centerOfs, variables)
