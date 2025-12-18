import os
import glob
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from pathlib import Path
from matplotlib.widgets import RectangleSelector, Button, TextBox
from matplotlib.ticker import NullLocator  # kill minor y ticks (non-distorting)

plt.rcParams['keymap.pan'] = []
plt.rcParams['keymap.zoom'] = []

#quick and dirty solution to make the plot start at maximum
startMaxDispRows = 500
startMaxDispCols = 500

minCapRows = 250
minCapCols = 250

#observing location below
lat_deg = 38 + 23/60 + 50.0/3600
lon_deg = -(115 + 58/60 + 41.6/3600)

location = EarthLocation(lat=lat_deg * u.deg,
						  lon=lon_deg * u.deg,
						  height=0 * u.m)

folderPath = Path(r"G:\Shared drives\jbp Research\high-z (CMU)\Data\Field_Data\LunarDryLake\2025November\digital_spec_condensed\20251105")

print("Resolved folder path:", folderPath)
print("Exists?", os.path.isdir(folderPath))

timeFolders  = sorted(glob.glob(os.path.join(folderPath, "*")))

allSpectra = list()
calibSpectra = list()

for folder in timeFolders:
	allFiles = sorted(glob.glob(os.path.join(folder, "*")))

	skyFiles = [f for f in allFiles if f.endswith("state1.npy")]
	calibFiles = [f for f in allFiles if f.endswith("state2.npy")]
	#calibFiles = [f for f in allFiles if not f.endswith("state1.npy")]
	calibSpectra.extend(calibFiles)
	allSpectra.extend(skyFiles)


#WORK IN PROGRESS HERE
def toSidereal(hhmmss):
	hh = hhmmss[0:2]
	mm = hhmmss[2:4]
	ss = hhmmss[4:6]
	
	utc_string = f"2025-11-02 {hh}:{mm}:{ss}"   #Nov 2nd date; adjust for different day
	
	t = Time(utc_string, scale="utc")
	
	lst = t.sidereal_time('apparent', longitude=location.lon)
	
	return lst.hour   #Return as float sidereal hours


def spec_to_dbm(spectrum, offset=-135):
	"""Convert recorded spectrum from digital spectrometer to dBm with an offset obtained from calibration."""
	spectrum = np.array(spectrum)
	finalSpectrum = 10 * np.log10(spectrum)+offset
	return finalSpectrum

def compileHeatData(filePaths, state = "collection"):
	twoDListOfValues = list()
	
	collectedUTCTime = []
	collectedSpectra = []
	#singleSpectrum = []
	for path in filePaths:
		print(path)
		print()
		data = np.load(path, allow_pickle = True)
		data = data.tolist()
		for key in data:
			if key.isdigit():
				if type(data[key]) == dict:
					collectedUTCTime.append(key)
					collectedSpectra.append(spec_to_dbm(data[key]["spectrum"].tolist(), -128).tolist())
					#if len(collectedSpectra) == 22578//2: 
						#singleSpectrum = spec_to_dbm(data[key]["spectrum"].tolist(), -128).tolist()
				elif type(data[key]) == list:
					print("CORRECTED FOR LIST!!!!!!!!!!!!!!!!!!!1!")
					print
					correctDict = data[key][0]
					collectedSpectra.append(spec_to_dbm(correctDict["spectrum"].tolist(), -128).tolist())
					collectedUTCTime.append(key)
				print(len(collectedSpectra))
			elif key == "spectrum":
				print("still missed data!")
				#twoDListOfValues.append(spec_to_dbm(data["spectrum"].tolist(), -128).tolist())
				
	zippedData = zip(collectedUTCTime, collectedSpectra)

	sortedZippedData = sorted(zippedData, key=lambda item: item[0])

	organizedUTCTime, twoDListOfValues = zip(*sortedZippedData)
	print("Collected data")

	print(len(twoDListOfValues))
	#print(singleSpectrum)
	
	twoDListOfValues = np.array(twoDListOfValues, dtype=float)

	print("Min:", np.min(twoDListOfValues))
	print("Max:", np.max(twoDListOfValues))

	fig, ax = plt.subplots(figsize=(12, 8))
	#leave room at bottom for widgets
	plt.subplots_adjust(bottom=0.20)

	nRows, nCols = twoDListOfValues.shape

	#Real UTC time range (kept for labeling)
	maxTimeUTC = max(organizedUTCTime)
	minTimeUTC = min(organizedUTCTime)

	printedMaxTime = maxTimeUTC[0:2] + ":" + maxTimeUTC[2:4] + ":" + maxTimeUTC[4:6]
	printedMinTime = minTimeUTC[0:2] + ":" + minTimeUTC[2:4] + ":" + minTimeUTC[4:6]

	maxTimeSidereal = toSidereal(maxTimeUTC)
	minTimeSidereal = toSidereal(minTimeUTC)

	print(maxTimeSidereal, minTimeSidereal)

	#Gamma/power norm with gamma < 1 boosts small values, compresses large ones
	if state == "collection":
		norm = PowerNorm(gamma=4, vmin=-70, vmax=-50)
	if state == "calib":
		norm = PowerNorm(gamma=1, vmin=-70, vmax=-35)

	#caps MAX equals full data size
	maxCapRows = nRows
	maxCapCols = nCols

	#start caps
	maxDispRows = min(startMaxDispRows, nRows)
	maxDispCols = min(startMaxDispCols, nCols)

	#numeric extents for image coords
	xExtent = (0.0, 409.6)
	yExtent = (0.0, float(nRows))

	def toUTC(hhmmss):
		return hhmmss[0:2] + ":" + hhmmss[2:4] + ":" + hhmmss[4:6]

	#"clamping" functions
	def clamp(val, lo, hi):
		return max(lo, min(hi, val))

	def clampRows(val):
		return clamp(val, minCapRows, maxCapRows)

	def clampCols(val):
		return clamp(val, minCapCols, maxCapCols)

    #Text details of helper message
	hudText = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=12, 
		color="white", bbox=dict(boxstyle="round", facecolor="black", alpha=0.4))

	def updateHud():
		hudText.set_text(f"Disp cap: rows={maxDispRows}/{maxCapRows}, cols={maxDispCols}/{maxCapCols}\n")

	#UTC-only ticks
	def updateTimeTicks(row0, row1):
		nticks = 10
		if row1 <= row0 + 1: return
		tickRows = np.linspace(row0, row1 - 1, nticks).astype(int)
		tickLabels = [toUTC(organizedUTCTime[r]) for r in tickRows]
		ax.set_yticks(tickRows)
		ax.set_yticklabels(tickLabels)

		#removes all other y ticks (like row num)
		ax.yaxis.set_minor_locator(NullLocator())

	#sets max row and col values
	maxDispRows = clampRows(maxDispRows)
	maxDispCols = clampCols(maxDispCols)

	def renderView():
		nonlocal maxDispRows, maxDispCols

		x0, x1 = ax.get_xlim()
		y0, y1 = ax.get_ylim()

		x0 = max(xExtent[0], min(xExtent[1], x0))
		x1 = max(xExtent[0], min(xExtent[1], x1))
		y0 = max(yExtent[0], min(yExtent[1], y0))
		y1 = max(yExtent[0], min(yExtent[1], y1))

		col0 = int((min(x0, x1) - xExtent[0]) / (xExtent[1] - xExtent[0]) * nCols)
		col1 = int((max(x0, x1) - xExtent[0]) / (xExtent[1] - xExtent[0]) * nCols)
		row0 = int(min(y0, y1))
		row1 = int(max(y0, y1))

		col0 = max(0, min(nCols - 1, col0))
		col1 = max(1, min(nCols, col1))
		row0 = max(0, min(nRows - 1, row0))
		row1 = max(1, min(nRows, row1))

		view = twoDListOfValues[row0:row1, col0:col1]
		vr, vc = view.shape

		stepR = max(1, int(np.ceil(vr / maxDispRows)))
		stepC = max(1, int(np.ceil(vc / maxDispCols)))

		viewDs = view[::stepR, ::stepC]

		im.set_data(viewDs)
		im.set_extent([xExtent[0] + col0 / nCols * (xExtent[1] - xExtent[0]),
			xExtent[0] + col1 / nCols * (xExtent[1] - xExtent[0]), row1, row0])

		updateTimeTicks(row0, row1)
		updateHud()
		fig.canvas.draw_idle()

	# initial downsampled full view
	fullStepR = max(1, int(np.ceil(nRows / maxDispRows)))
	fullStepC = max(1, int(np.ceil(nCols / maxDispCols)))
	viewFull = twoDListOfValues[::fullStepR, ::fullStepC]

	im = ax.imshow(viewFull, cmap='inferno', norm=norm, aspect='auto', extent=[xExtent[0], xExtent[1], 
		yExtent[1], yExtent[0]], interpolation="nearest", rasterized=True)

	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label('Power (dBm)', fontsize=18)
	cbar.ax.tick_params(labelsize=14)

	ax.set_title('Digital Spectrometer Data on Nov. 5, 2025 - Lunar Dry Lake', fontsize = 18)
	ax.set_xlabel('Frequency (MHz)', fontsize = 18)
	ax.set_ylabel('Time (hh:mm:ss in UTC)', fontsize = 18)
	ax.tick_params(axis="both", labelsize=14)

	updateTimeTicks(0, nRows)
	updateHud()

	# lock out horizontal panning
	lastXlim = ax.get_xlim()
	inZoom = False

	def onXlimChanged(ax_):
		nonlocal lastXlim, inZoom
		if not inZoom:
			ax_.set_xlim(lastXlim)
			ax_.figure.canvas.draw_idle()

	ax.callbacks.connect('xlim_changed', onXlimChanged)

	origXlim = ax.get_xlim()
	origYlim = ax.get_ylim()

    #rectangle selection object, set by makeSelector()
	rectSelector = None

    #event function for click
	def onSelect(eclick, erelease):
		nonlocal lastXlim, inZoom

		x1, y1 = eclick.xdata, eclick.ydata
		x2, y2 = erelease.xdata, erelease.ydata
		if None in (x1, y1, x2, y2):
			return

		xmin, xmax = sorted([x1, x2])
		ymin, ymax = sorted([y1, y2])

		if abs(xmax - xmin) < 1e-12 or abs(ymax - ymin) < 1e-12:
			return

		inZoom = True
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(ymax, ymin)
		lastXlim = ax.get_xlim()
		inZoom = False

		renderView()

    #makes rectSelector object
	def makeSelector():
		nonlocal rectSelector
		if rectSelector is not None:
			rectSelector.disconnect_events()
			rectSelector.set_active(False)
			rectSelector = None

		rectSelector = RectangleSelector(ax, onSelect, useblit=True, button=[1], interactive=False,spancoords="data")

	makeSelector()
	

	#Rows button
	axBtnRows = fig.add_axes([0.22, 0.11, 0.07, 0.055])
	btnRows = Button(axBtnRows, "Set Rows")

	#Rows text box
	axBoxRows = fig.add_axes([0.22, 0.045, 0.12, 0.045])
	boxRows = TextBox(axBoxRows, "Rows", initial=str(maxDispRows))

    #Cols button
	axBtnCols = fig.add_axes([0.55, 0.11, 0.07, 0.055])
	btnCols = Button(axBtnCols, "Set Cols")

	#Cols text box
	axBoxCols = fig.add_axes([0.55, 0.045, 0.12, 0.045])
	boxCols = TextBox(axBoxCols, "Cols", initial=str(maxDispCols))


    #Sets rows val
	def submitRows(text):
		nonlocal maxDispRows
		try:
			val = int(float(text))
		except ValueError:
			boxRows.set_val(str(maxDispRows))
			return
		maxDispRows = clampRows(val)
		boxRows.set_val(str(maxDispRows))
		renderView()

    #Sets cols value
	def submitCols(text):
		nonlocal maxDispCols
		try:
			val = int(float(text))
		except ValueError:
			boxCols.set_val(str(maxDispCols))
			return
		maxDispCols = clampCols(val)
		boxCols.set_val(str(maxDispCols))
		renderView()

	boxRows.on_submit(submitRows)
	boxCols.on_submit(submitCols)

    #key event r/R resets zoom, and z/Z zooms out by a standard factor
	def onKey(event):
		nonlocal lastXlim, inZoom

		# reset view
		if event.key in ["r", "R"]:
			inZoom = True
			ax.set_xlim(origXlim)
			ax.set_ylim(origYlim)
			lastXlim = ax.get_xlim()
			inZoom = False

			renderView()
			makeSelector()
			return

		# zoom out by a standard factor
		if event.key in ["z", "Z"]:
			baseScale = 1.5	 # zoom out by 50%

			#Current edges
			curX0, curX1 = ax.get_xlim()
			curY0, curY1 = ax.get_ylim()

			#Center coords
			xCenter = 0.5 * (curX0 + curX1)
			yCenter = 0.5 * (curY0 + curY1)

			xRange = (curX1 - curX0) * baseScale
			yRange = (curY0 - curY1) * baseScale

			newX0 = xCenter - xRange / 2.0
			newX1 = xCenter + xRange / 2.0
			newY1 = yCenter - yRange / 2.0
			newY0 = yCenter + yRange / 2.0	 # keep newY0 > newY1

			#Extend to total data limits so we don't go outside the image
			newX0 = max(xExtent[0], newX0)
			newX1 = min(xExtent[1], newX1)
			if newX1 <= newX0:
				return	 # don't collapse/invert x

			newY0 = max(yExtent[0], min(yExtent[1], newY0))
			newY1 = max(yExtent[0], min(yExtent[1], newY1))
			if newY0 <= newY1:
				return	 # don't collapse/invert y

			# apply limits in "zoom mode" so xlim_changed hook doesn't fight us
			inZoom = True
			ax.set_xlim(newX0, newX1)
			ax.set_ylim(newY0, newY1)
			lastXlim = ax.get_xlim()
			inZoom = False

			# redraw the downsampled view at the new zoom level
			renderView()
			return
		
	fig.canvas.mpl_connect("key_press_event", onKey)


	plt.show()


#Most Calib
#print(len(calibSpectra[0:len(calibSpectra) - 2*len(calibSpectra)//9:1]))
#compileHeatData(calibSpectra[0:len(calibSpectra) - 2*len(calibSpectra)//9:1], "calib")

#Some Calib
# firstFrac = 2*len(calibSpectra)//6
# secondFrac = 3*len(calibSpectra)//6
# firstFrac = 0
# secondFrac = len(calibSpectra)
# print(firstFrac, secondFrac)
# print(len(calibSpectra[firstFrac:secondFrac:1]))
# compileHeatData(calibSpectra[firstFrac:secondFrac:1], "calib")

#Some Sky Data
print(len(allSpectra[0:len(allSpectra) - 5*len(allSpectra)//9:1]))
compileHeatData(allSpectra[0:len(allSpectra) - 5*len(allSpectra)//9:1])

#All Sky Data
# print(len(allSpectra[0:len(allSpectra):1]))
# compileHeatData(allSpectra[0:len(allSpectra):1])