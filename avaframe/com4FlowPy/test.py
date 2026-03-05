import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

import avaframe.ana5Utils.regionalThalwegTools as tools


z = 0
density = 200


alphaG = [25, 30, 35, 40]
alphaLStyle = {40: "--", 25: "-", 30: "-.", 35: ":"}

z_gamma = np.linspace(100, 1500, 3)

cmap = plt.cm.Greens
norm = plt.Normalize(vmin=z_gamma.min() - 200, vmax=z_gamma.max())


def computeVelocity(alpha, gamma, z_gamma):
    tanAlpha = np.tan(np.deg2rad(alpha))
    tanGamma = np.tan(np.deg2rad(gamma))
    zDelta = z_gamma * (1 - tanAlpha / tanGamma)
    velocity = tools.zDelta2velocity(zDelta)
    return velocity


def computeVelocityFromPressure(pressure, density):
    velocity = np.sqrt(pressure / density)
    return velocity


def computeDS(alpha, gamma, z_gamma, z):
    tanAlpha = np.tan(np.deg2rad(alpha))
    tanGamma = np.tan(np.deg2rad(gamma))
    ds = z_gamma * (1 / tanAlpha - 1 / tanGamma) + z / tanAlpha
    return ds


vel1kPa = computeVelocityFromPressure(1000, density)
vel10kPa = computeVelocityFromPressure(10000, density)

pressureClasses = np.array([5, 50, 250.0, 750]) * 1000
velocityClass = computeVelocityFromPressure(pressureClasses, density)

figG1, axsG1 = plt.subplots(2, figsize=(12, 12))
figG2, axsG2 = plt.subplots(figsize=(12, 12))
figS1, axsS1 = plt.subplots(2, figsize=(12, 12))
figGL1, axsGL1 = plt.subplots(2, figsize=(12, 12))
figGL2, axsGL2 = plt.subplots(figsize=(12, 12))

# customLines = []
for alpha in alphaG:
    gamma = np.arange(alpha, 41, 0.1)
    gammaGL = np.arange(alpha, 41, 2)
    normGL = plt.Normalize(vmin=gammaGL.min() - 4, vmax=gammaGL.max())

    lStyle = alphaLStyle[alpha]

    for ax in [axsG1[0], axsG2, axsS1[0], axsGL1[0], axsGL2]:
        (dummy,) = ax.plot([alphaG[0]], [0], color="black", linestyle=lStyle, label=f"alpha = {alpha}°")

    # customLines.append(dummy)
    for zG in z_gamma:
        color = cmap(norm(zG))

        tanGamma = np.tan(np.deg2rad(gamma))
        sGamma = tanGamma * zG

        velocity = computeVelocity(alpha, gamma, zG)
        # pressure = computePressure(velocity, density)
        # pressure = pressure * 1 / 1000
        ds = computeDS(alpha, gamma, zG, z)

        axsS1[0].plot(sGamma, velocity, linestyle=lStyle, color=color, label=f"$z_\gamma$ = {zG} m")
        axsS1[0].axhline(vel1kPa, c="y")
        axsS1[0].axhline(vel10kPa, c="r")
        axsG1[0].plot(gamma, velocity, linestyle=lStyle, color=color, label=f"$z_\gamma$ = {zG} m")
        # axsG1[0].axhline(vel1kPa, c="y")
        # axsG1[0].axhline(vel10kPa, c="r")
        for velC in velocityClass:
            axsG1[0].axhline(velC)
        axsS1[1].plot(sGamma, ds, linestyle=lStyle, c=color)
        axsG1[1].plot(gamma, ds, linestyle=lStyle, c=color)
        # customLines.append(c)

        axsG2.plot(velocity, ds, linestyle=lStyle, color=color, label=f"$z_\gamma$ = {zG} m")
        for velC in velocityClass:
            axsG2.axvline(velC)

    for g in gammaGL:
        color = cmap(normGL(g))
        tanGamma = np.tan(np.deg2rad(g))
        sGamma = tanGamma * z_gamma

        velocity = computeVelocity(alpha, g, z_gamma)
        # pressure = computePressure(velocity, density)
        # pressure = pressure / 1000

        ds = computeDS(alpha, g, z_gamma, z)

        axsGL1[0].plot(sGamma, velocity, linestyle=lStyle, color=color, label=f"$\gamma$ = {g}°")

        axsGL1[1].plot(sGamma, ds, linestyle=lStyle, c=color)
        # customLines.append(c)

        axsGL2.plot(velocity, ds, linestyle=lStyle, color=color, label=f"$\gamma$ = {g}°")

for ax in [axsS1[0], axsS1[1], axsGL1[0], axsGL1[1]]:
    # ax.invert_xaxis()
    ax.set_xlabel("local horizontal distance ($s_\gamma$) [m]")

for ax in [axsG1[0], axsG1[1]]:
    ax.invert_xaxis()
    ax.set_xlabel("local travel angle ($\gamma$) [°]")

for ax in [axsS1[1], axsG1[1], axsGL1[1]]:
    ax.legend(fontsize="x-small")
    ax.set_ylabel("Ueberlauflaenge [m]")

for ax in [axsS1[0], axsG1[0], axsGL1[0]]:
    ax.legend(fontsize="x-small")
    ax.set_ylabel("velocity in m/s")
    # ax.set_ylabel("pressure [kPa]")
    # ax.set_title(f"with denisty {density} kg/m³")


for ax in [axsG2, axsGL2]:
    ax.legend(fontsize="x-small")
    ax.set_xlabel("velocity in m/s")
    ax.set_ylabel("Ueberlauflaenge [m]")
    ax.set_title(f"at beta$_{z}$")
    # ax.set_xlim([-5, 400])


# axs.legend(handles=customLines, fontsize="small")


figG1.savefig("/home/paula/Downloads/gamma.png")
figG2.savefig("/home/paula/Downloads/gammaCrazy.png")

figS1.savefig("/home/paula/Downloads/s_gamma.png")

figGL1.savefig("/home/paula/Downloads/gammaL.png")
figGL2.savefig("/home/paula/Downloads/gammaLCrazy.png")


# ---------------------------------


z = 0

alphaG = [20, 25]
alphaLStyle = {20: "--", 25: "-"}

z_gamma = np.linspace(100, 1000, 10)
s_gamma = np.linspace(10, 1010, 11)

cmap = plt.cm.Greens
norm = plt.Normalize(vmin=z_gamma.min() - 200, vmax=z_gamma.max())


def computeDS(alpha, gamma, z_gamma, z):
    tanAlpha = np.tan(np.deg2rad(alpha))
    tanGamma = np.tan(np.deg2rad(gamma))
    ds = z_gamma * (1 / tanAlpha - 1 / tanGamma) + z / tanAlpha
    return ds


figSN1, axsSN1 = plt.subplots(2, figsize=(12, 12))
figSN2, axsSN2 = plt.subplots(figsize=(12, 12))


# customLines = []
for alpha in alphaG:
    gammaGL = np.arange(alpha, 30, 2)
    normGL = plt.Normalize(vmin=gammaGL.min() - 4, vmax=gammaGL.max())

    lStyle = alphaLStyle[alpha]

    (dummy,) = axsSN1[0].plot([alphaG[0]], [0], color="black", linestyle=lStyle, label=f"alpha = {alpha}°")

    # customLines.append(dummy)
    for zG in z_gamma:
        color = cmap(norm(zG))

        tanAlpha = np.tan(np.deg2rad(alpha))
        zDelta = zG * (1 - tanAlpha * zG * s_gamma)
        velocity = tools.zDelta2velocity(zDelta)

        gamma = np.rad2deg(np.arctan(zG / s_gamma))

        ds = zG * (1 / tanAlpha - s_gamma / zG) + z / tanAlpha

        axsSN1[0].plot(s_gamma, gamma, linestyle=lStyle, color=color, label=f"$z_\gamma$ = {zG} m")
        axsSN1[1].plot(s_gamma, ds, linestyle=lStyle, c=color)
        # customLines.append(c)

        axsSN2.plot(velocity, ds, linestyle=lStyle, color=color, label=f"$z_\gamma$ = {zG} m")

axsSN1[0].legend(fontsize="x-small")
axsSN1[0].set_xlabel("local horizontal distance ($s_\gamma$) [m]")
axsSN1[1].set_xlabel("local horizontal distance ($s_\gamma$) [m]")

axsSN1[0].set_ylabel("velocity in m/s")
axsSN1[1].set_ylabel("Ueberlauflaenge [m]")

figSN1.savefig("/home/paula/Downloads/sNew.png")
figSN2.savefig("/home/paula/Downloads/s2New.png")
