import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

import avaframe.ana5Utils.regionalThalwegTools as tools

z = 0

alphaG = [20, 25]
alphaLStyle = {20: "--", 25: "-"}

z_gamma = np.linspace(100, 1000, 10)

cmap = plt.cm.Greens
norm = plt.Normalize(vmin=z_gamma.min(), vmax=z_gamma.max())

fig, (axs, axs2) = plt.subplots(2, figsize=(12, 12))
fig2, ax2 = plt.subplots(figsize=(12, 12))

# customLines = []
for alpha in alphaG:
    gamma = np.arange(alpha, 30, 0.1)

    lStyle = alphaLStyle[alpha]
    (dummy,) = axs.plot([alphaG[0]], [0], color="black", linestyle=lStyle, label=f"alpha = {alpha}°")
    (dummy,) = ax2.plot([alphaG[0]], [0], color="black", linestyle=lStyle, label=f"alpha = {alpha}°")

    # customLines.append(dummy)
    for zG in z_gamma:
        color = cmap(norm(zG))

        tanAlpha = np.tan(np.deg2rad(alpha))
        tanGamma = np.tan(np.deg2rad(gamma))
        sGamma = tanGamma * zG

        zDelta = zG * (1 - tanAlpha / tanGamma)
        velocity = tools.zDelta2velocity(zDelta)

        axs.plot(sGamma, velocity, linestyle=lStyle, color=color, label=f"$z_\gamma$ = {zG} m")

        ds = zG * (1 / tanAlpha - 1 / tanGamma) + z / tanAlpha
        axs2.plot(sGamma, ds, linestyle=lStyle, c=color)
        # customLines.append(c)

        ax2.plot(velocity, ds, linestyle=lStyle, color=color, label=f"$z_\gamma$ = {zG} m")

for ax in [axs, axs2]:
    # ax.invert_xaxis()
    ax.set_xlabel("local horizontal distance ($s_\gamma$) [m]")


# axs.legend(handles=customLines, fontsize="small")

axs.legend(fontsize="x-small")
axs.set_ylabel("velocity in m/s")
axs2.set_ylabel("Ueberlauflaenge [m]")
axs2.set_title(f"at beta$_{z}$")
fig.savefig("/home/paula/Downloads/z_gammaS.png")


ax2.legend(fontsize="x-small")
ax2.set_xlabel("velocity in m/s")
ax2.set_ylabel("Ueberlauflaenge [m]")
# ax2.set_xlim([-5, 400])
fig2.savefig("/home/paula/Downloads/newS.png")
