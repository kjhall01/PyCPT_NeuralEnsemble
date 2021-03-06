import os
import sys
import warnings
import platform
import struct
import xarray as xr
import numpy as np
import pandas as pd
import copy
from scipy.stats import t
from scipy.stats import invgamma
import cartopy.crs as ccrs
from cartopy import feature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import fileinput
#import pycurl
import subprocess
import numpy as np
import matplotlib as mpl
from IPython import get_ipython
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import netCDF4 as ns
import json
import kyle_ensemble as ke
import torch as te

warnings.filterwarnings("ignore")
plt.ioff()

#Supporting class for custom pyplot/matplotlib colormaps
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# Ignoring masked values and all kinds of edge cases to make a
		# simple example..
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))

#class for handling the absurd number of parameters for running PyCPT and passing them to various functions
class PYCPT():
	def callSys(self, arg):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		if self.showPlot:
			get_ipython().system(arg)
		else:
			subprocess.check_output(arg, shell=True)

	def __init__(self, work, workdir, cptdir, models, met, obs, station, MOS, xmodes_min, xmodes_max, ymodes_min, ymodes_max, ccamodes_min, ccamodes_max, nmodes, PREDICTAND, PREDICTOR, mons, tgti, tgtf, tgts, tini, tend, monf, fyr, force_download, nla1, sla1, wlo1, elo1, nla2, sla2, wlo2, elo2, shp_file='shp_file', use_default='True', localobs="None", lonkey="None", latkey="None", timekey="None", datakey="None", showPlot=True):
		self.isInitialized = 1
		#These are the variables set by the user
		self.showPlot = showPlot
		self.models = models
		self.work = work
		self.workdir = workdir
		self.shp_file = shp_file
		self.use_default = use_default
		self.met = met
		self.obs = obs
		self.cptdir = cptdir
		self.station = station
		self.MOS = MOS
		self.xmodes_min = xmodes_min
		self.xmodes_max = xmodes_max
		self.ymodes_min = ymodes_min
		self.ymodes_max = ymodes_max
		self.ccamodes_min = ccamodes_min
		self.ccamodes_max = ccamodes_max
		self.eof_modes = nmodes
		self.PREDICTAND = PREDICTAND
		self.PREDICTOR = PREDICTOR
		self.mons = mons
		self.tgti = tgti
		self.tgtf = tgtf
		self.tgts = tgts
		self.tini = tini
		self.tend = tend
		self.monf = monf
		self.fyr = fyr
		self.force_download = force_download
		self.nla1 = nla1
		self.sla1 = sla1
		self.wlo1 = wlo1
		self.elo1 = elo1
		self.nla2 = nla2
		self.sla2 = sla2
		self.wlo2 = wlo2
		self.elo2 = elo2
		self.localobs = localobs
		self.lonkey =  lonkey
		self.latkey = latkey
		self.timekey = timekey
		self.datakey = datakey
		self.L = ['1']
		#Following are to be updated during 'setup_params'
		self.ndays = 0
		self.nmonths = 0
		self.rainfall_frequency = 0
		self.wetday_threshold = 0
		self.threshold_pctle = False
		self.obs_source = ''
		self.hdate_last = 0
		self.mpref = ''
		self.ntrain = 0
		self.fprefix=''

		#a dictionary to use for dynamic formatting in the 'get data' function with the url dictionary
		self.arg_dict = {
			#set by the user, others depend on which tgt were looking at or which model or obs source
			'tini': self.tini,
			'tend': self.tend,
			'nla1': self.nla1,
			'sla1': self.sla1,
			'wlo1': self.wlo1,
			'elo1': self.elo1,
			'nla2': self.nla2,
			'sla2': self.sla2,
			'wlo2': self.wlo2,
			'elo2': self.elo2,
			'fyr': self.fyr
		}
		self.url_dict = {
		  'Hindcasts': {
		    'PRCP': {	'CanSIPSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPSv2/.HINDCAST/.MONTHLY/.prec/SOURCES/.Models/.NMME/.CanSIPSv2/.FORECAST/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.HINDCAST/.MONTHLY/.prec/SOURCES/.Models/.NMME/.CMC1-CanCM3/.FORECAST/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.HINDCAST/.MONTHLY/.prec/SOURCES/.Models/.NMME/.CMC2-CanCM4/.FORECAST/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p1-aer04': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p1-aer04/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES/.MONTHLY/.prec/SOURCES/.Models/.NMME/.NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/M/%281%29%2824%29RANGE/%5BM%5D/average/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'UQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%2812%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'VQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%281%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'RFREQ': {	'CanSIPSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPSv2/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p1-aer04': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p1-aer04/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		},
		  'Obs': {
		    'RFREQ': {
		        True: 'https://iridl.ldeo.columbia.edu/{obs_source}/Y/{sla2}/{nla2}/RANGE/X/{wlo2}/{elo2}/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201982)/(31%20Dec%202010)/RANGEEDGES/%5BT%5Dpercentileover/{wetday_threshold}/flagle/T/{ndays}/runningAverage/{ndays}/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		        False:'http://datoteca.ole2.org/SOURCES/.UEA/.CRU/.TS4p0/.monthly/.wet/lon/%28X%29/renameGRID/lat/%28Y%29/renameGRID/time/%28T%29/renameGRID/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/{sla2}/{nla2}/RANGEEDGES/X/{wlo2}/{elo2}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'},
		    'PRCP': {   'Chilestations': 'http://iridl.ldeo.columbia.edu/{obs_source}/T/%28{tar}%29/seasonalAverage/-999/setmissing_value/%5B%5D%5BT%5Dcptv10.tsv',
		                'ENACTS-BD':'https://datalibrary.bmd.gov.bd/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'CPC-CMAP-URD': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'TRMM': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'CPC': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'CHIRPS': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'GPCC': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
		    }
		  },
		  'Forecasts': {
		    'PRCP': {	'CanSIPSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPSv2/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
					    'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p1-aer04': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p1-aer04/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.FORECAST/.EARLY_MONTH_SAMPLES/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'UQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%281%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'VQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%281%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'RFREQ': {	'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
					    'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p1-aer04': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p1-aer04/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		  }
		}
		self.setupParams(0)

	def __str__(self):
		return json.dumps(vars(self), sort_keys=True, indent=4)

	def __repr__(self):
		return json.dumps(vars(self), sort_keys=True, indent=4)

	def __eq__(self, other):
		dct1, dct2 = vars(self), vars(other)
		flag = 0
		for key in dct1.keys():
			if dct1[key] != dct2[key]:
				flag = 1
		if flag:
			return False
		else:
			return True

	def __sub__(self, other):
		dct1, dct2 = vars(self), vars(other)
		dct_diff = {}
		for key in dct1.keys():
			if dct1[key] != dct2[key]:
				dct_diff[key] = (dct1[key], dct2[key])
		return dct_diff

	@classmethod
	def from_dict(self, dct):
		#These are the variables set by the user
		showPlot = dct["showPlot"]
		models = dct["models"]
		work = dct["work"]
		workdir = dct["workdir"]
		shp_file = dct["shp_file"]
		use_default = dct["use_default"]
		met = dct["met"]
		obs = dct["obs"]
		cptdir = dct["cptdir"]
		station = dct["station"]
		MOS = dct["MOS"]
		xmodes_min = dct["xmodes_min"]
		xmodes_max = dct["xmodes_max"]
		ymodes_min = dct["ymodes_min"]
		ymodes_max = dct["ymodes_max"]
		ccamodes_min = dct["ccamodes_min"]
		ccamodes_max = dct["ccamodes_max"]
		eof_modes = dct["eof_modes"]
		PREDICTAND = dct["PREDICTAND"]
		PREDICTOR = dct["PREDICTOR"]
		mons = dct["mons"]
		tgti = dct["tgti"]
		tgtf = dct["tgtf"]
		tgts = dct["tgts"]
		tini = dct["tini"]
		tend = dct["tend"]
		monf = dct["monf"]
		fyr = dct["fyr"]
		force_download = dct["force_download"]
		nla1 = dct["nla1"]
		sla1 = dct["sla1"]
		wlo1 = dct["wlo1"]
		elo1 = dct["elo1"]
		nla2 = dct["nla2"]
		sla2 = dct["sla2"]
		wlo2 = dct["wlo2"]
		elo2 = dct["elo2"]
		localobs = dct["localobs"]
		lonkey =  dct["lonkey"]
		latkey = dct["latkey"]
		timekey = dct["timekey"]
		datakey = dct["datakey"]
		return PYCPT( work, workdir,cptdir, models, met, obs, station, MOS, xmodes_min, xmodes_max, ymodes_min, ymodes_max, ccamodes_min, ccamodes_max, eof_modes, PREDICTAND, PREDICTOR, mons, tgti, tgtf, tgts, tini, tend, monf, fyr, force_download, nla1, sla1, wlo1, elo1, nla2, sla2, wlo2, elo2, shp_file, use_default, showPlot=showPlot)

	@classmethod
	def load(self, filename):
		return PYCPT._load(filename)

	@classmethod
	def _load(self, filename):
		dct = json.load(open(filename, 'r'))
		return self.from_dict(dct)


	def save(self, filename):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		f = open(filename, 'w')
		f.write(json.dumps(vars(self), sort_keys=True, indent=4))
		f.close()

	def setupParams(self, tar_ndx):
		#global rainfall_frequency,threshold_pctle,wetday_threshold,obs_source,hdate_last,mpref,L,ntrain,fprefix, nmonths, ndays
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		days_in_month_dict = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		if '-' in self.tgts[tar_ndx]:
			mon_ini, mon_fin = self.tgts[tar_ndx].split('-')
			self.nmonths, self.ndays, flag = 0, 0, 0
			found_end, count = 0, 0
			while found_end == 0 and count < 24:
				if flag == 1:
					self.nmonths += 1
					self.ndays += days_in_month_dict[months[count % 12]]
					if months[count % 12] == mon_fin:
						flag, found_end = 0, 1

				if months[count % 12] == mon_ini:
					flag = 1
					self.nmonths += 1
					self.ndays += days_in_month_dict[months[count % 12]]
				count += 1

		else:
			mon_ini = self.tgts[tar_ndx]
			self.nmonths, self.ndays, self.flag = 0, 0, 0
			for i in months:
				if i == mon_ini:
					flag = 1
					self.nmonths += 1
					self.ndays += days_in_month_dict[i]

		# Predictor switches
		if self.PREDICTOR=='PRCP' or self.PREDICTOR=='UQ' or self.PREDICTOR=='VQ':
			self.rainfall_frequency = False  #False uses total rainfall for forecast period, True uses frequency of rainy days
			self.threshold_pctle = False
			self.wetday_threshold = -999 #WET day threshold (mm) --only used if rainfall_frequency is True!
		elif self.PREDICTOR=='RFREQ':
			self.rainfall_frequency = True  #False uses total rainfall for forecast period, True uses frequency of rainy days
			self.wetday_threshold = 3 #WET day threshold (mm) --only used if rainfall_frequency is True!
			self.threshold_pctle = False    #False for threshold in mm; Note that if True then if counts DRY days!!!

		if self.rainfall_frequency:
			pass#print('Predictand is Rainfall Frequency; wet day threshold = '+str(self.wetday_threshold)+' mm')
		else:
			pass#print('Predictand is Rainfall Total (mm)')

		########Observation dataset URLs
		if self.obs == 'CPC-CMAP-URD':
		    self.obs_source = 'SOURCES/.Models/.NMME/.CPC-CMAP-URD/prate'
		    self.hdate_last = 2010
		elif self.obs == 'TRMM':
		    self.obs_source = 'SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/-180./1.5/180./GRID/Y/-50/1.5/50/GRID'
		    self.hdate_last = 2014
		elif self.obs == 'CPC':
		    self.obs_source = 'SOURCES/.NOAA/.NCEP/.CPC/.UNIFIED_PRCP/.GAUGE_BASED/.GLOBAL/.v1p0/.extREALTIME/.rain/X/-180./1.5/180./GRID/Y/-90/1.5/90/GRID'
		    self.hdate_last = 2018
		elif self.obs == 'CHIRPS':
		    self.obs_source = 'SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/'+str(self.ndays)+'/mul'
		    self.hdate_last = 2018
		elif self.obs == 'Chilestations':
		    self.obs_source = 'home/.xchourio/.ACToday/.CHL/.prcp'
		    self.hdate_last = 2019
		elif self.obs == 'GPCC':
		    self.obs_source = 'SOURCES/.WCRP/.GCOS/.GPCC/.FDP/.version7/.0p5/.prcp/'+str(self.nmonths)+'/mul'
		    self.hdate_last = 2013
		elif self.obs == 'Chilestations':
		    self.obs_source = 'home/.xchourio/.ACToday/.CHL/.prcp'
		    self.hdate_last = 2019
		elif self.obs == 'ENACTS-BD':
			self.obs_source = 'SOURCES/.Bangladesh/.BMD/.monthly/.rainfall/.rfe_merged/'+str(self.nmonths)+'/mul'
			self.hdate_last = 2020
		else:
		    print ("Obs option is invalid")

		########MOS-dependent parameters
		if self.MOS=='None':
		    self.mpref='noMOS'
		elif self.MOS=='CCA':
		    self.mpref='CCA'
		elif self.MOS=='PCR':
		    self.mpref='PCR'
		elif self.MOS=='ELR':
		    self.mpref='ELRho'

		self.L=['1'] #lead for file name (TO BE REMOVED --requested by Xandre)
		if self.tgts[tar_ndx] in ['Nov-Jan', 'Dec-Feb', 'Jan-Mar']:
			self.ntrain= self.tend-self.tini # length of training period
		else:
			self.ntrain= self.tend-self.tini + 1# length of training period

		self.fprefix = self.PREDICTOR

		self.arg_dict['mon'] = self.mons[tar_ndx]
		self.arg_dict['tar'] = self.tgts[tar_ndx]
		self.arg_dict['tgti'] = self.tgti[tar_ndx]
		self.arg_dict['tgtf'] = self.tgtf[tar_ndx]
		self.arg_dict['monf'] = self.monf[tar_ndx]
		self.arg_dict['ndays'] = self.ndays
		self.arg_dict['nmonths30'] = self.nmonths*30
		self.arg_dict['obs_source'] = self.obs_source
		self.file='FCST_xvPr'
		self.NGMOS = 'None'

		#If you have local observations data, please provide some metadata here!
		local_obs_files = ['none', 'none'] #['../local/Merged_data_OND_1982_2019.tsv', '../local/MERGED_DATA_JJAS_1982_2019.tsv']#['../MERGED_DATA_S_ASIA_1982_2019_JJAS.nc','../Merged_data_yrmean_OND_1982_2019.nc'] #either 'None' if you don't have local data, or the path of your local obs data file
		local_obs_LongitudeKey = 'LONGITUDE' #ncdf key for accessing latitude - must be the same for all files
		local_obs_LatitudeKey = 'LATITUDE'   #ncdf key for accessing longitude
		local_obs_timeKey = 'time'           #ncdf key for accessing time
		local_obs_datakey = 'rf'             #ncdf key for accessing data itself

	def prepFiles(self, tar_ndx, model_ndx,showPlot=True):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		x= "Function to download (or not) the needed files"
		print('Preparing CPT files for '+self.models[model_ndx]+' and initialization '+self.mons[tar_ndx]+'...')
		self.getData( tar_ndx, model_ndx, 'Hindcasts')
		self.getData( tar_ndx, model_ndx, 'Obs')
		self.getData( tar_ndx, model_ndx, 'Forecasts')

	def getData(self,  tar_ndx, model_ndx, datatype):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		#set the model to the current focus
		self.arg_dict['model'] = self.models[model_ndx]
		found=0
		if datatype=='Obs' and False: #we are not doing local data at this time
			if os.path.isfile(self.localobs[tar_ndx]):
				print('Found Local data - trying to open as netCDF... ')
				try:
					self.convertNCDF_CPT(tar_ndx)
					print('Success! Converted to CPT for you')
					found = 1
				except:
					print('Could not open as NetCDF - We are trusting that it is correct CPT format')
					self.cutOutLocalData(tar_ndx)
					found = 1
			else:
				print("No local data by that name,, looking for previously downloaded")
				found = 0

		if datatype != 'Obs' or found == 0:
			if not self.force_download:
				try:
					if datatype == 'Hindcasts':
						ff=open("./input/"+self.models[model_ndx]+"_{}_".format(self.fprefix)+self.tgts[tar_ndx]+"_ini"+self.mons[tar_ndx]+".tsv",  'r')
						s = ff.readline()
					elif datatype == "Obs":
						if self.fprefix == 'RFREQ':
							fpre = 'RFREQ'
						else:
							fpre = 'PRCP'
						ff = open("./input/obs_"+fpre+"_"+self.tgts[tar_ndx]+".tsv", 'r')
						s = ff.readline()
					else:
						ff=open("./input/"+self.models[model_ndx]+"fcst_PRCP_"+self.tgts[tar_ndx]+"_ini"+self.monf[tar_ndx]+str(self.fyr)+".tsv")
						s = ff.readline()
				except OSError as err:
					print("\033[1mWarning:\033[0;0m {0}".format(err))
					print("{} precip file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m".format(datatype))
					self.force_download = True

			if self.force_download:
				if datatype == 'Obs':
					url = self.url_dict[datatype][self.fprefix][self.threshold_pctle if self.fprefix == 'RFREQ' else self.obs ]
				else:
					url = self.url_dict[datatype][self.fprefix][self.models[model_ndx]]
				print("\n {} (Freq) data URL: \n\n ".format(datatype)+url.format(**self.arg_dict))
				if datatype == 'Hindcasts':
					self.callSys("curl -k "+url.format(**self.arg_dict)+" > ./input/"+self.models[model_ndx]+"_{}_".format(self.fprefix)+self.tgts[tar_ndx]+"_ini"+self.mons[tar_ndx]+".tsv")
				elif datatype == 'Obs':
					if self.fprefix == 'RFREQ':
						fpre = 'RFREQ'
					else:
						fpre = 'PRCP'
					self.callSys("curl -k "+url.format(**self.arg_dict)+" > ./input/obs_"+fpre+"_"+self.tgts[tar_ndx]+".tsv")
				else:
					self.callSys("curl -k "+url.format(**self.arg_dict)+" > ./input/"+self.models[model_ndx]+"fcst_{}_".format(self.fprefix)+self.tgts[tar_ndx]+"_ini"+self.monf[tar_ndx]+str(self.fyr)+".tsv")

		if self.obs_source=='home/.xchourio/.ACToday/.CHL/.prcp':   #weirdly enough, Ingrid sends the file with nfields=0. This is my solution for now. AGM
			replaceAll("obs_"+predictand+"_"+tar+".tsv","cpt:nfields=0","cpt:nfields=1")

		print('{} file ready to go'.format(datatype))
		print('----------------------------------------------')

	def CPTscript(self, tar_ndx, model_ndx=-1):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		flag=0
		if model_ndx == -1:
			self._tempmods = copy.deepcopy(self.models)
			self.models = ['NextGen']
			self._tempmos = copy.deepcopy(self.MOS)
			self.MOS = self.NGMOS
			model_ndx=0
			flag=1

		if model_ndx == -2:
			self._tempmods = copy.deepcopy(self.models)
			self.models = ['Neural']
			self._tempmos = copy.deepcopy(self.MOS)
			self.MOS = self.NGMOS
			model_ndx=0
			flag=1

		# Set up CPT parameter file
		f=open("./scripts/params","w")
		if self.MOS=='CCA':
			# Opens CCA
			f.write("611\n")
		elif self.MOS=='PCR':
			# Opens PCR
			f.write("612\n")
		elif self.MOS=='ELR':
			# Opens GCM; because the calibration takes place via sklearn.linear_model (in the Jupyter notebook)
			f.write("614\n")
		elif self.MOS=='None':
			# Opens GCM (no calibration performed in CPT)
			f.write("614\n")
		else:
			print ("MOS option is invalid")

		# First, ask CPT to stop if error is encountered
		f.write("571\n")
		f.write("3\n")

		# Opens X input file
		f.write("1\n")
		file='./input/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.mons[tar_ndx]+'.tsv\n'
		f.write(file)
		# Nothernmost latitude
		f.write(str(self.nla1)+'\n')
		# Southernmost latitude
		f.write(str(self.sla1)+'\n')
		# Westernmost longitude
		f.write(str(self.wlo1)+'\n')
		# Easternmost longitude
		f.write(str(self.elo1)+'\n')

		if self.MOS=='CCA' or self.MOS=='PCR':
			# Minimum number of X modes
			f.write("{}\n".format(self.xmodes_min))
			# Maximum number of X modes
			f.write("{}\n".format(self.xmodes_max))

			# Opens forecast (X) file
			f.write("3\n")
			file='./input/'+self.models[model_ndx]+'fcst_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'.tsv\n'
			f.write(file)
			#Start forecast:
			f.write("223\n")
			if self.monf[tar_ndx]=="Dec":
				f.write(str(self.fyr+1)+"\n")
			else:
				f.write(str(self.fyr)+"\n")

		# Opens Y input file
		f.write("2\n")
		file='./input/obs_'+self.PREDICTAND+'_'+self.tgts[tar_ndx]+'.tsv\n'
		f.write(file)
		if self.station==False:
			# Nothernmost latitude
			f.write(str(self.nla2)+'\n')
			# Southernmost latitude
			f.write(str(self.sla2)+'\n')
			# Westernmost longitude
			f.write(str(self.wlo2)+'\n')
			# Easternmost longitude
			f.write(str(self.elo2)+'\n')
		if self.MOS=='CCA':
			# Minimum number of Y modes
			f.write("{}\n".format(self.ymodes_min))
			# Maximum number of Y modes
			f.write("{}\n".format(self.ymodes_max))

			# Minimum number of CCA modes
			f.write("{}\n".format(self.ccamodes_min))
			# Maximum number of CCAmodes
			f.write("{}\n".format(self.ccamodes_max))

		# X training period
		f.write("4\n")
		# First year of X training period
		if self.monf[tar_ndx] in ['Dec', 'Nov']:
			f.write("{}\n".format(self.tini+1))
		else:
			f.write("{}\n".format(self.tini))
		# Y training period
		f.write("5\n")
		# First year of Y training period
		if self.monf[tar_ndx] in ['Dec', 'Nov']:
			f.write("{}\n".format(self.tini+1))
		else:
			f.write("{}\n".format(self.tini))


		# Goodness index
		f.write("531\n")
		# Kendall's tau
		f.write("3\n")

		# Option: Length of training period
		f.write("7\n")
		# Length of training period
		f.write(str(self.ntrain)+'\n')
		#	%store 55 >> params
		# Option: Length of cross-validation window
		f.write("8\n")
		# Enter length
		f.write("3\n")

		if self.MOS!="None":
			# Turn ON transform predictand data
			f.write("541\n")

		if self.fprefix=='RFREQ':
			# Turn ON zero bound for Y data	 (automatically on by CPT if variable is precip)
			f.write("542\n")
		# Turn ON synchronous predictors
		f.write("545\n")
		# Turn ON p-values for masking maps
		#f.write("561\n")

		### Missing value options
		f.write("544\n")
		# Missing value X flag:
		blurb='-999\n'
		f.write(blurb)
		# Maximum % of missing values
		f.write("10\n")
		# Maximum % of missing gridpoints
		f.write("10\n")
		# Number of near-neighbors
		f.write("1\n")
		# Missing value replacement : best-near-neighbors
		f.write("4\n")
		# Y missing value flag
		blurb='-999\n'
		f.write(blurb)
		# Maximum % of missing values
		f.write("10\n")
		# Maximum % of missing stations
		f.write("10\n")
		# Number of near-neighbors
		f.write("1\n")
		# Best near neighbor
		f.write("4\n")

		# Transformation settings
		#f.write("554\n")
		# Empirical distribution
		#f.write("1\n")

		#######BUILD MODEL AND VALIDATE IT	!!!!!

		# NB: Default output format is GrADS format
		# select output format
		f.write("131\n")
		# GrADS format
		f.write("3\n")

		# save goodness index
		f.write("112\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Kendallstau_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# Build cross-validated model
		f.write("311\n")

		# save EOFs
		if self.MOS=='CCA' or self.MOS=='PCR' :    #kjch092120
			f.write("111\n")
			#X EOF
			f.write("302\n")
			file= './output/'+self.models[model_ndx] +'_'+ self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
			f.write(file)
			#Exit submenu
			f.write("0\n")
		if self.MOS=='CCA' :        #kjch092120
			f.write("111\n")
			#Y EOF
			f.write("312\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
			f.write(file)
			#Exit submenu
			f.write("0\n")

		# cross-validated skill maps
		f.write("413\n")
		# save Pearson's Correlation
		f.write("1\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Pearson_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save Spearmans Correlation
		f.write("2\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Spearman_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save 2AFC score
		f.write("3\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_2AFC_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save RocBelow score
		f.write("15\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RocBelow_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save RocAbove score
		f.write("16\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RocAbove_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save RocAbove score
		f.write("7\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RMSE_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)



		if self.MOS=='CCA' or self.MOS=='PCR' or self.MOS=='None':  #kjch092120 #DO NOT USE CPT to compute probabilities if MOS='None' --use IRIDL for direct counting
			#######FORECAST(S)	!!!!!
			# Probabilistic (3 categories) maps
			f.write("455\n")
			# Output results
			f.write("111\n")
			# Forecast probabilities
			f.write("501\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_P_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			#502 # Forecast odds
			#Exit submenu
			f.write("0\n")

			# Compute deterministc values and prediction limits
			f.write("454\n")
			# Output results
			f.write("111\n")
			# Forecast values
			f.write("511\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_V_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			#502 # Forecast odds


			#######Following files are used to plot the flexible format
			# Save cross-validated predictions
			f.write("201\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_xvPr_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save deterministic forecasts [mu for Gaussian fcst pdf]
			f.write("511\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_mu_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save prediction error variance [sigma^2 for Gaussian fcst pdf]
			f.write("514\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_var_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save z
			f.write("532\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_z_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save predictand [to build predictand pdf]
			f.write("102\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_Obs_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)

			#Exit submenu
			f.write("0\n")

			# Change to ASCII format to send files to DL
			f.write("131\n")
			# ASCII format
			f.write("2\n")
			# Output results
			f.write("111\n")
			# Save cross-validated predictions
			f.write("201\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_xvPr_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save deterministic forecasts [mu for Gaussian fcst pdf]
			f.write("511\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_mu_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Forecast probabilities
			f.write("501\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_P_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save prediction error variance [sigma^2 for Gaussian fcst pdf]
			f.write("514\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_var_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save z
			f.write("532\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_z_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Save predictand [to build predictand pdf]
			f.write("102\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_Obs_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)

			# cross-validated skill maps
			if self.MOS=="PCR" or self.MOS=="CCA" or self.MOS=="None": #kjch092120
				f.write("0\n")

			# cross-validated skill maps
			f.write("413\n")
			# save 2AFC score
			f.write("3\n")
			file='./output/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'_2AFC_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'\n'
			f.write(file)
			# Stop saving  (not needed in newest version of CPT)

		###########PFV --Added by AGM in version 1.5
		#Compute and write retrospective forecasts for prob skill assessment.
		#Re-define forecas file if PCR or CCA
		if self.MOS=="PCR" or self.MOS=="CCA" or self.MOS=="None" : #kjch092120
			f.write("3\n")
			file='./input/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.mons[tar_ndx]+'.tsv\n'  #here a conditional should choose if rainfall freq is being used
			f.write(file)
		#Forecast period settings
		f.write("6\n")
		# First year to forecast. Save ALL forecasts (for "retroactive" we should only assess second half)
		if self.monf[tar_ndx]=="Oct" or self.monf[tar_ndx]=="Nov" or self.monf[tar_ndx]=="Dec":
			f.write(str(self.tini+1)+'\n')
		else:
			f.write(str(self.tini)+'\n')
		#Number of forecasts option
		f.write("9\n")
		# Number of reforecasts to produce
		if self.monf[tar_ndx]=="Oct" or self.monf[tar_ndx]=="Nov" or self.monf[tar_ndx]=="Dec":
			f.write(str(self.ntrain-1)+'\n')
		else:
			f.write(str(self.ntrain)+'\n')
		# Change to ASCII format
		f.write("131\n")
		# ASCII format
		f.write("2\n")
		# Probabilistic (3 categories) maps
		f.write("455\n")
		# Output results
		f.write("111\n")
		# Forecast probabilities --Note change in name for reforecasts:
		f.write("501\n")
		file='./output/'+self.models[model_ndx]+'_RFCST_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'\n'
		f.write(file)
		#502 # Forecast odds
		#Exit submenu
		f.write("0\n")

		# Close X file so we can access the PFV option
		f.write("121\n")
		f.write("Y\n")  #Yes to cleaning current results:# WARNING:
		#Select Probabilistic Forecast Verification (PFV)
		f.write("621\n")
		# Opens X input file
		f.write("1\n")
		file='./output/'+self.models[model_ndx]+'_RFCST_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'.txt\n'
		f.write(file)
		# Nothernmost latitude
		f.write(str(self.nla2)+'\n')
		# Southernmost latitude
		f.write(str(self.sla2)+'\n')
		# Westernmost longitude
		f.write(str(self.wlo2)+'\n')
		# Easternmost longitude
		f.write(str(self.elo2)+'\n')

		f.write("5\n")
		# First year of the PFV
		# for "retroactive" only first half of the entire training period is typically used --be wise, as sample is short)
		if self.monf[tar_ndx]=="Oct" or self.monf[tar_ndx]=="Nov" or self.monf[tar_ndx]=="Dec":
			f.write(str(self.tini+1)+'\n')
		else:
			f.write(str(self.tini)+'\n')

		#If these prob forecasts come from a cross-validated prediction (as it's coded right now)
		#we don't want to cross-validate those again (it'll change, for example, the xv error variances)
		#Forecast Settings menu
		f.write("552\n")
		#Conf level at 50% to have even, dychotomous intervals for reliability assessment (as per Simon suggestion)
		f.write("50\n")
		#Fitted error variance option  --this is the key option: 3 is 0-leave-out cross-validation, so no cross-validation!
		f.write("3\n")
		#-----Next options are required but not really used here:
		#Ensemble size
		f.write("10\n")
		#Odds relative to climo?
		f.write("N\n")
		#Exceedance probabilities: show as non-exceedance?
		f.write("N\n")
		#Precision options:
		#Number of decimal places (Max 8):
		f.write("3\n")
		#Forecast probability rounding:
		f.write("1\n")
		#End of required but not really used options ----

		#Verify
		f.write("313\n")

		#Reliability diagram
		f.write("431\n")
		f.write("Y\n") #yes, save results to a file
		file='./output/'+self.models[model_ndx]+'_RFCST_reliabdiag_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'.tsv\n'
		f.write(file)

		# select output format -- GrADS, so we can plot it in Python
		f.write("131\n")
		# GrADS format
		f.write("3\n")

		# Probabilistic skill maps
		f.write("437\n")
		# save Ignorance (all cats)
		f.write("101\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Ignorance_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# Probabilistic skill maps
		f.write("437\n")
		# save Ranked Probability Skill Score (all cats)
		f.write("122\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RPSS_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)

		# Probabilistic skill maps
		f.write("437\n")
		# save Ranked Probability Skill Score (all cats)
		f.write("131\n")
		file='./output/'+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_GROC_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'\n'
		f.write(file)


		# Exit
		f.write("0\n")
		f.write("0\n")
		f.close()
		if platform.system() == 'Windows':
			self.callSys("cd scripts && copy params "+self.models[model_ndx]+"_"+self.fprefix+"_"+self.mpref+"_"+self.tgts[tar_ndx]+"_"+self.mons[tar_ndx]+".cpt")
		else:
			self.callSys("cp ./scripts/params ./scripts/"+self.models[model_ndx]+"_"+self.fprefix+"_"+self.mpref+"_"+self.tgts[tar_ndx]+"_"+self.mons[tar_ndx]+".cpt")

		if flag:
			self.models = self._tempmods

	def run(self, tar_ndx, model_ndx=-1):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		flag=0
		if model_ndx == -1:
			self._tempmods = copy.deepcopy(self.models)
			self.models=['NextGen']
			flag = 1
			model_ndx = 0

		print('Executing CPT for '+self.models[model_ndx]+' and initialization '+self.mons[tar_ndx]+'...')

		try:
			if platform.system() == "Windows":
				subprocess.check_output([ self.cptdir+'CPT_batch.exe', '<', './scripts/params', '>', './scripts/CPT_stout_train_'+self.models[model_ndx]+'_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'.txt'] , shell=True)
			else:
				subprocess.check_output(self.cptdir+'CPT.x < ./scripts/params > ./scripts/CPT_stout_train_'+self.models[model_ndx]+'_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'.txt',stderr=subprocess.STDOUT, shell=True)
		except subprocess.CalledProcessError as e:
			print("CPT Windows version throws an error right at the end of its operation- everything should be fine for the rest of this notebook, but you need to click 'close' on the 'Access Violation' Window that pops up for now. ")

		print('----------------------------------------------')
		print('Calculations for '+self.mons[tar_ndx]+' initialization completed!')
		print('See output folder, and check scripts/CPT_stout_train_'+self.models[model_ndx]+'_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'.txt for errors')
		print('----------------------------------------------')
		print('----------------------------------------------\n\n\n')

		if flag:
			self.MOS = self._tempmos
			self.models = self._tempmods

	def pltdomain(self, use_shp_file=False):
		if use_shp_file:
			try:
				self.shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
			except:
				print('Failed to load custom shape file')
		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
			#name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')
		#ax.add_feature(cfeature.LAND)
	    #ax.add_feature(cfeature.COASTLINE)
	    #ax.add_feature(states_provinces, edgecolor='gray')


		fig = plt.subplots(figsize=(15,15), subplot_kw=dict(projection=ccrs.PlateCarree()))
		loni = [self.wlo1,self.wlo2]
		lati = [self.nla1,self.nla2]
		lone = [self.elo1,self.elo2]
		late = [self.sla1,self.sla2]
		title = ['Predictor', 'Predictand']

		for i in range(2):

			ax = plt.subplot(1, 2, i+1, projection=ccrs.PlateCarree())
			ax.set_extent([loni[i],lone[i],lati[i],late[i]], ccrs.PlateCarree())
			ax.add_feature(feature.LAND)
			ax.add_feature(feature.OCEAN)
			ax.set_title(title[i]+" domain")
			pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
			pl.xlabels_top = False
			pl.ylabels_left = False
			pl.xformatter = LONGITUDE_FORMATTER
			pl.yformatter = LATITUDE_FORMATTER
			if self.use_default == 'True':
				ax.add_feature(states_provinces, edgecolor='black')
			if use_shp_file:
				try:
					ax.add_feature(self.shape_feature, edgecolor='black')
				except:
					print('failed to load shape file')
		plt.savefig("./images/domain.png",dpi=300, bbox_inches='tight') #SAVE_FILE 0_domain.png
		if self.showPlot:
			plt.show()
		else:
			plt.close()

	def pltmap(self, score_ndx, isNextGen=-1, use_shp_file=False, isNeural=-1):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		if isNextGen != -1:
			self.models = ['NextGen']

		if isNeural != -1:
			self.models=['Neural']

		if use_shp_file:
			try:
				self.shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
			except:
				print('Failed to load custom shape file')

		nmods=len(self.models)
		nsea=len(self.mons)
		if self.obs == 'ENACTS-BD':
			x_offset = 0.6
			y_offset = 0.4
		else:
			x_offset = 0
			y_offset = 0

		fig, ax = plt.subplots(nrows=nmods, ncols=nsea, figsize=(6*nsea, 6*nmods),sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
		#fig = plt.figure(figsize=(20,40))
		#ax = [ plt.subplot2grid((nmods+1, nsea), (int(np.floor(nd / nsea)), int(nd % nsea)),rowspan=1, colspan=1, projection=ccrs.PlateCarree()) for nd in range(nmods*nsea) ]
		#ax.append(plt.subplot2grid((nmods+1, nsea), (nmods, 0), colspan=nsea ) )
		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]
		if self.met[score_ndx] not in ['Pearson','Spearman']:
			#urrent_cmap = plt.cm.get_cmap('RdYlBu', 10 )
			current_cmap = self.make_cmap(10)
		else:
			#current_cmap = plt.cm.get_cmap('RdYlBu', 14 )
			current_cmap = self.make_cmap(14)
		#current_cmap.set_bad('white',1.0)
		#current_cmap.set_under('white', 1.0)
		print()
		print(self.met[score_ndx])
		for i in range(nmods):
			for j in range(nsea):
				mon=self.mons[j]
				#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
				with open('./output/'+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_2AFC_'+self.tgts[j]+'_'+mon+'.ctl', "r") as fp:
					for line in self.lines_that_contain("XDEF", fp):
						W = int(line.split()[1])
						XD= float(line.split()[4])
				with open('./output/'+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_2AFC_'+self.tgts[j]+'_'+mon+'.ctl', "r") as fp:
					for line in self.lines_that_contain("YDEF", fp):
						H = int(line.split()[1])
						YD= float(line.split()[4])

	#			ax = plt.subplot(nwk/2, 2, wk, projection=ccrs.PlateCarree())

				#ax = plt.subplot(nmods,nsea, k, projection=ccrs.PlateCarree())
				if self.obs == 'ENACTS-BD':
				#	wlo2,elo2,sla2,nla2 => loni,lone,lati,late
					ax[i][j].set_extent([self.wlo2+x_offset,self.wlo2+W*XD+x_offset,self.sla2+y_offset,self.sla2+H*YD+y_offset], ccrs.PlateCarree())
				else:
					ax[i][j].set_extent([self.wlo2+x_offset,self.wlo2+W*XD+x_offset,self.sla2+y_offset,self.sla2+H*YD+y_offset], ccrs.PlateCarree())
				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
	#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				#pl.xlabels_bottom = False
				#if i == nmods - 1: change so long vals in every plot
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				if use_shp_file:
					try:
						ax[i][j].add_feature(self.shape_feature, edgecolor='black')
					except:
						print('failed to add your shape file')
				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')
				ax[i][j].set_ybound(lower=self.sla2, upper=self.nla2)

				if j == 0:
					ax[i][j].text(-0.42, 0.5, self.models[i],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
				if i == 0:
					ax[i][j].set_title(self.tgts[j])
				#for i, axi in enumerate(axes):  # need to enumerate to slice the data
				#	axi.set_ylabel(model, fontsize=12)


				if self.met[score_ndx] == 'CCAFCST_V' or self.met[score_ndx] == 'PCRFCST_V':
					f=open('./output/'+self.models[i]+'_'+self.fprefix+'_'+self.met[score_ndx]+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat','rb')
					if platform.system() == "Windows":
						garb = struct.unpack('s', f.read(1))[0]
					recl=struct.unpack('i',f.read(4))[0]
					numval=int(recl/np.dtype('float32').itemsize)
					#Now we read the field
					A=np.fromfile(f,dtype='float32',count=numval)
					var = np.transpose(A.reshape((W, H), order='F'))
					var[var==-999.]=np.nan #only sensible values

					CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						#vmin=-max(np.max(var),np.abs(np.min(var))), #vmax=np.max(var),
						norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
					ax[i][j].set_title("Deterministic forecast for Week "+str(wk))
					if self.fprefix == 'RFREQ':
						label ='Freq Rainy Days (days)'
					elif self.fprefix == 'PRCP':
						label = 'Rainfall anomaly (mm/week)'
						f.close()
					#current_cmap = plt.cm.get_cmap()
					#current_cmap.set_bad(color='white')
					#current_cmap.set_under('white', 1.0)
				else:
					#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
					f=open('./output/'+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_'+self.met[score_ndx]+'_'+self.tgts[j]+'_'+mon+'.dat','rb')
					if platform.system() == "Windows":
						garb = struct.unpack('s', f.read(1))[0]
					recl=struct.unpack('i',f.read(4))[0]
					numval=int(recl/np.dtype('float32').itemsize)
					#Now we read the field
					A=np.fromfile(f,dtype='float32',count=numval)
					var = np.transpose(A.reshape((W, H), order='F'))
					#define colorbars, depending on each score	--This can be easily written as a function
					if self.met[score_ndx] == '2AFC':
						var[var<0]=np.nan #only positive values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=0,vmax=100,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = '2AFC (%)'

					if self.met[score_ndx] == 'RMSE':
						var[var<0]=np.nan #only positive values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=0,vmax=1000,
						cmap=plt.get_cmap('Reds', 10),
						transform=ccrs.PlateCarree())
						label = 'RMSE'
					if self.met[score_ndx] == 'RocAbove' or self.met[score_ndx]=='RocBelow':
						var[var<0]=np.nan #only positive values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=0,vmax=1,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'ROC area'

					if self.met[score_ndx] == 'Spearman' or self.met[score_ndx]=='Pearson':
						var[var<-1.]=np.nan #only sensible values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=-1.05,vmax=1.05,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'Correlation'

					if self.met[score_ndx] == 'Ignorance':
						var[var<-1.]=np.nan #only sensible values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						#vmin=0,vmax=1,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'Ignorance'

					if self.met[score_ndx] == 'RPSS':
						var[var<-1.]=np.nan #only sensible values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						#vmin=-1,vmax=1,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'RPSS'

					if self.met[score_ndx] == 'GROC':
						var[var<-1.]=np.nan #only sensible values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						#vmin=0,vmax=1,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'GROC'

				#Make true if you want cbar on left, default is cbar on right
				is_left = False
				if is_left:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center left',
		                   bbox_to_anchor=(-0.22, 0., 1, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
					if self.met[score_ndx] in ['Pearson','Spearman']:
						 bounds = [-0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
						 cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins,  orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == '2AFC':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == 'RMSE':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02)#, ticks=bounds)
					else:
						bounds = [round(0.1*gt,1) for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02)#, ticks=bounds)
					#cbar.set_label(label) #, rotation=270)\
					#axins.yaxis.tick_left()
				else:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center right',
		                   bbox_to_anchor=(0., 0., 1.15, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
					if self.met[score_ndx] in ['Pearson','Spearman']:
						 bounds = [-0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
						 cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins,  orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == '2AFC':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == 'RMSE':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02)#, ticks=bounds)
					else:
						bounds = [round(0.1*gt,1) for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02)#, ticks=bounds)
					cbar.set_label(label) #, rotation=270)\
					#axins.yaxis.tick_left()
				f.close()
		if self.models[0] == 'NextGen':
			filename =  'NextGen_' + self.met[score_ndx]
		else:
			filename = 'Models_' + self.met[score_ndx]
		fig.savefig('./images/' + filename + '.png', dpi=500, bbox_inches='tight')
		if self.showPlot:
			plt.show()
		else:
			plt.close()

	def plteofs(self, mode, use_shp_file=False):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		if use_shp_file:
			try:
				self.shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
			except:
				print('Failed to load custom shape file')

		M = self.eof_modes
		#mol=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
		if self.MOS=='None':
			print('No EOFs are computed if MOS=None is used')
			return
		if self.showPlot:
			print('\n\n\n-------------EOF {}-------------\n'.format(mode+1))

		if self.mpref == 'CCA':
			nmods=len(self.models) + 1 #nmods + obs
		else:
			nmods = len(self.models)
		nsea=len(self.mons)
		tari=self.tgts[0]
		model=self.models[0]
		monn=self.mons[0]

		data = [[[[] for k in range(M)] for j in range(nsea)] for i in range(nmods)]

		with open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+tari+'_'+monn+'.ctl', "r") as fp:
			for line in self.lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+tari+'_'+monn+'.ctl', "r") as fp:
			for line in self.lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])

		if self.mpref=='CCA':
			with open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+tari+'_'+monn+'.ctl', "r") as fp:
				for line in self.lines_that_contain("XDEF", fp):
					Wy = int(line.split()[1])
					XDy= float(line.split()[4])
			with open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+tari+'_'+monn+'.ctl', "r") as fp:
				for line in self.lines_that_contain("YDEF", fp):
					Hy = int(line.split()[1])
					YDy= float(line.split()[4])
			eofy=np.empty([M,Hy,Wy])  #define array for later use
			datay = [[[[] for k in range(M)] for j in range(nsea)] for i in range(nmods)]


		eofx=np.empty([M,H,W])  #define array for later use

		#plt.figure(figsize=(20,10))
		#fig, ax = plt.subplots(figsize=(20,15),sharex=True,sharey=True)
		fig, ax = plt.subplots(nrows=nmods, ncols=nsea, sharex=False,sharey=False, figsize=(10*nsea,6*nmods), subplot_kw={'projection': ccrs.PlateCarree()})
		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]

		#current_cmap = plt.cm.get_cmap('RdYlBu', 14)
		current_cmap = self.make_cmap(14)
		#current_cmap.set_bad('white',1.0)
		#current_cmap.set_under('white', 1.0)

		for i in range(nmods):
			for j in range(nsea):

				tari=self.tgts[j]
				if i == 0 and self.mpref=='CCA':
					model = self.models[0]
				elif self.mpref=='CCA' and i > 0:
					model=self.models[i-1]
				else:
					model = self.models[i]



				if i == 0 and self.obs == 'ENACTS-BD':
					ax[i][j].set_extent([87.5,93,20.5,27], crs=ccrs.PlateCarree())
				else:
					if self.mpref=='PCR':
						ax[i][j].set_extent([self.wlo1,self.wlo1+W*XD,self.sla1,self.sla1+H*YD], crs=ccrs.PlateCarree())  #EOF domains will look different between CCA and PCR if X and Y domains are different
					else:
						ax[i][j].set_extent([self.wlo1,self.wlo1+Wy*XDy,self.sla1,self.sla1+Hy*YDy], crs=ccrs.PlateCarree())

				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
	#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)

				#tick_spacing=0.5
				#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				if use_shp_file:
					try:
						ax[i][j].add_feature(self.shape_feature, edgecolor='black')
					except:
						print('failed to load your shapefile')
				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')


				if self.mpref == 'CCA':
					if self.obs == 'ENACTS-BD' and i ==0:
						ax[i][j].set_ybound(lower=20.5, upper=27)
						ax[i][j].set_xbound(lower=87.5, upper=93)
					elif self.obs != 'ENACTS-BD' and i == 0:
						ax[i][j].set_ybound(lower=self.sla2, upper=self.nla2)
						ax[i][j].set_xbound(lower=self.wlo2, upper=self.elo2)
					else:
						ax[i][j].set_ybound(lower=self.sla1, upper=self.nla1)
						ax[i][j].set_xbound(lower=self.wlo1, upper=self.elo1)
				else:
					ax[i][j].set_ybound(lower=self.sla1, upper=self.nla1)
					ax[i][j].set_xbound(lower=self.wlo1, upper=self.elo1)


				if j == 0:
					if i == 0 and self.mpref == 'CCA':
						ax[i][j].text(-0.42, 0.5, 'Obs',rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
					elif i > 0 and self.mpref == 'CCA':
						ax[i][j].text(-0.42, 0.5, self.models[i-1],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
					else:
						ax[i][j].text(-0.42, 0.5, self.models[i],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)


				if i == 0: #kjch101620
					ax[i][j].set_title(self.tgts[j])
				#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
				if i ==0:
					tari=self.tgts[j]
					model=self.models[0]
					monn=self.mons[0]
					nsea=len(self.mons)
					tar = self.tgts[j]
					mon=self.mons[j]
					if self.mpref == 'CCA':
						#f=open('./output/'+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_Obs_'+mons[j]+'_'+mon+str(fyr)+'.dat','rb')
						#f=open('./output/'+models[i-1]+'_'+fprefix+predictand+'_'+mpref+'_EOFX_'+mons[j]+'_'+mon+'.dat','rb')
						f=open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+tar+'_'+mon+'.dat','rb')
						#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
						for mo in range(M):
							#Now we read the field
							if platform.system() == "Windows":
								garb = struct.unpack('s', f.read(1))[0]
							recl=struct.unpack('i',f.read(4))[0]
							numval=int(recl/np.dtype('float32').itemsize) #this if for each time/EOF stamp
							A0=np.fromfile(f,dtype='float32',count=numval)
							endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
							if platform.system() == "Windows":
								garb = struct.unpack('s', f.read(1))[0]
							A0[A0==-999.] = np.nan
							data[i][j][mo]= np.transpose(A0.reshape((Wy, Hy), order='F'))
						eofy[eofy==-999.]=np.nan #nans


						if self.obs == 'ENACTS-BD':
							CS=ax[i][j].pcolormesh(np.linspace(87.6, 93.0,num=Wy), np.linspace(27.1, 20.4, num=Hy), data[i][j][mode],
							vmin=-.1,vmax=.1,
							cmap=current_cmap,
							transform=ccrs.PlateCarree())
						else:
						#CS=ax[i][j].pcolormesh(np.linspace(loni, loni+Wy*XDy,num=Wy), np.linspace(lati+Hy*YDy, lati, num=Hy), eofy[mode,:,:],
							CS=ax[i][j].pcolormesh(np.linspace(self.wlo1, self.elo1,num=Wy), np.linspace(self.nla1, self.sla1, num=Hy), data[i][j][mode],
							vmin=-.1,vmax=.1,
							cmap=current_cmap,
							transform=ccrs.PlateCarree())

						label = 'EOF charges'
					else:
						mon=self.mons[j]
						f=open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+self.tgts[j]+'_'+mon+'.dat','rb')
						#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
						for mo in range(M):
							#Now we read the field
							if platform.system() == "Windows":
								garb = struct.unpack('s', f.read(1))[0]
							recl=struct.unpack('i',f.read(4))[0]
							numval=int(recl/np.dtype('float32').itemsize) #this if for each time/EOF stamp
							A0=np.fromfile(f,dtype='float32',count=numval)
							endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
							if platform.system() == "Windows":
								garb = struct.unpack('s', f.read(1))[0]
							A0[A0==-999.]=np.nan
							data[i][j][mo] = np.transpose(A0.reshape((W, H), order='F'))
							eofx[mo,:,:]= np.transpose(A0.reshape((W, H), order='F'))

						eofx[eofx==-999.]=np.nan #nans
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo1, self.wlo1+W*XD,num=W), np.linspace(self.sla1+H*YD, self.sla1, num=H), data[i][j][mode],
						vmin=-.105, vmax=.105,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'EOF charges'

				else:
					mon=self.mons[j]
					if self.mpref == 'CCA':
						f=open('./output/'+self.models[i-1]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+self.tgts[j]+'_'+mon+'.dat','rb')
					else:
						f=open('./output/'+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+self.tgts[j]+'_'+mon+'.dat','rb')

					#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
					for mo in range(M):
						#Now we read the field
						if platform.system() == "Windows":
							garb = struct.unpack('s', f.read(1))[0]
						recl=struct.unpack('i',f.read(4))[0]
						numval=int(recl/np.dtype('float32').itemsize) #this if for each time/EOF stamp
						A0=np.fromfile(f,dtype='float32',count=numval)
						endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
						if platform.system() == "Windows":
							garb = struct.unpack('s', f.read(1))[0]
						A0[A0==-999.]=np.nan
						data[i][j][mo] = np.transpose(A0.reshape((W, H), order='F'))
						eofx[mo,:,:]= np.transpose(A0.reshape((W, H), order='F'))

					eofx[eofx==-999.]=np.nan #nans
					CS=ax[i][j].pcolormesh(np.linspace(self.wlo1, self.wlo1+W*XD,num=W), np.linspace(self.sla1+H*YD, self.sla1, num=H), data[i][j][mode],
					vmin=-.105, vmax=.105,
					cmap=current_cmap,
					transform=ccrs.PlateCarree())
					label = 'EOF charges'

				is_left = False
				if is_left:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center left',
		                   bbox_to_anchor=(-0.22, 0., 1, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
				else:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center right',
		                   bbox_to_anchor=(0., 0., 1.15, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
				cbar = plt.colorbar(CS,ax=ax[i][j], cax=axins, orientation='vertical', pad=0.01, ticks= [-0.09, -0.075, -0.06, -0.045, -0.03, -0.015, 0, 0.015, 0.03, 0.045, 0.06, 0.075, 0.09])
				#cbar.set_label(label) #, rotation=270)
				#axins.yaxis.tick_left()
				f.close()
		model_names = ['obs']
		model_names.extend(self.models)
		if self.models[0] == 'NextGen':
			fig.savefig('./images/EOF{}_NextGen.png'.format(mode+1, dpi=500, bbox_inches='tight'))
		else:
			fig.savefig('./images/EOF{}_Models.png'.format(mode+1, dpi=500, bbox_inches='tight'))
				#plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
				#cbar_ax = plt.add_axes([0.85, 0.15, 0.05, 0.7])
				#plt.tight_layout()

				#plt.autoscale(enable=True)
		#plt.subplots_adjust(bottom=0.15, top=0.9)
		#cax = plt.axes([0.2, 0.08, 0.6, 0.04])
		if self.showPlot:
			plt.show()
		else:
			plt.close()

	def setNextGenModels(self, models):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		self.original_models = self.models
		self.models = models
		self.original_MOS = self.MOS
		self.MOS = "None"


	def setNeuralModels(self, models):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		self.original_models = self.models
		self.models = models
		self.original_MOS = self.MOS
		self.MOS = "None"

	def restoreModels(self):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		self.models = self.original_models
		self.MOS = self.original_MOS

	def make_cmap_blue(self,x):
		colors = [(244, 255,255),
		(187, 252, 255),
		(160, 235, 255),
		(123, 210, 255),
		(89, 179, 238),
		(63, 136, 254),
		(52, 86, 254)]
		colors = [ (colors[i][0] / 255.0, colors[i][1] / 255.0, colors[i][2] / 255.0) for i in range(len(colors))]
		#colors.reverse()
		return LinearSegmentedColormap.from_list( "matlab_clone", colors, N=x)

	def make_cmap(self, x):
		colors = [(238, 43, 51),
		(255, 57, 67),
		(253, 123, 91),
		(248, 175, 123),
		(254, 214, 158),
		(252, 239, 188),
		(255, 254, 241),
		(244, 255,255),
		(187, 252, 255),
		(160, 235, 255),
		(123, 210, 255),
		(89, 179, 238),
		(63, 136, 254),
		(52, 86, 254)
		]
		colors = [ (colors[i][0] / 255.0, colors[i][1] / 255.0, colors[i][2] / 255.0) for i in range(len(colors))]
		colors.reverse()
		return LinearSegmentedColormap.from_list( "matlab_clone", colors, N=x)

	def make_cmap_gray(self, x):
		colors = [(55,55,55),(235,235,235)]
		colors = [ (colors[i][0] / 255.0, colors[i][1] / 255.0, colors[i][2] / 255.0) for i in range(len(colors))]
		colors.reverse()
		return LinearSegmentedColormap.from_list( "matlab_clone", colors, N=x)

	def readGrADSctl(self, models,fprefix,predictand,mpref,id,tar,monf,fyr):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		#Read grads binary file size H, W, T
		with open('./output/'+str(models[0])+'_'+fprefix+predictand+'_'+mpref+id+'_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
			for line in self.lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				Wi= float(line.split()[3])
				XD= float(line.split()[4])
		with open('./output/'+str(models[0])+'_'+fprefix+predictand+'_'+mpref+id+'_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
			for line in self.lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				Hi= float(line.split()[3])
				YD= float(line.split()[4])
		with open('./output/'+str(models[0])+'_'+fprefix+predictand+'_'+mpref+id+'_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
			for line in self.lines_that_contain("TDEF", fp):
				T = int(line.split()[1])
				Ti= int((line.split()[3])[-4:])
				TD= 1  #not used
		return (W, Wi, XD, H, Hi, YD, T, Ti, TD)

	def lines_that_contain(self, string, fp):
		return [line for line in fp if string in line]

	def writeCPT(self, tar_ndx, var, outfile):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		vari = 'prec'
		varname = vari
		units = 'mm'
		var[np.isnan(var)]=-999. #use CPT missing value
		tar = self.tgts[tar_ndx]
		L=0.5*(float(self.tgtf[tar_ndx])+float(self.tgti[tar_ndx]))
		monthdic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
		S=monthdic[self.mons[tar_ndx]]
		if '-' in tar:
			mi=monthdic[tar.split("-")[0]]
			mf=monthdic[tar.split("-")[1]]

			#Read grads file to get needed coordinate arrays
			W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,tar,self.monf[tar_ndx],self.fyr)
			if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False
		else:
			mi=monthdic[tar]
			mf=monthdic[tar]

			#Read grads file to get needed coordinate arrays
			W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,tar,self.monf[tar_ndx],self.fyr)
			if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False

		Tarr = np.arange(Ti, Ti+T)
		Xarr = np.linspace(Wi, Wi+W*XD,num=W+1)
		Yarr = np.linspace(Hi+H*YD, Hi,num=H+1)

		#Now write the CPT file
		f = open(outfile, 'w')
		f.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/\n")
		#f.write("xmlns:cf=http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/\n")   #not really needed
		f.write("cpt:nfields=1\n")
		#f.write("cpt:T	" + str(Tarr)+"\n")  #not really needed
		for it in range(T):
			if xyear==True:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+str(Tarr[it]+1)+"-"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
			else:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
			#f.write("\t")
			np.savetxt(f, Xarr[0:-1], fmt="%.6f",newline='\t') #f.write(str(Xarr)[1:-1])
			f.write("\n") #next line
			for iy in range(H):
				#f.write(str(Yarr[iy]) + "\t" + str(var[it,iy,0:-1])[1:-1]) + "\n")
				np.savetxt(f,np.r_[Yarr[iy+1],var[it,iy,0:]],fmt="%.6f", newline='\t')  #excise extra line
				f.write("\n") #next line
		f.close()

	def NeuralEnsemble(self, tar_ndx):
		pt2 = copy.deepcopy(self)
		print('Loading PyCPT...', end='')
		sys.stdout.flush()

		pt2.tgts = [self.tgts[tar_ndx]]
		pt2.mons = [self.mons[tar_ndx]]
		pt2.tgti = [self.tgti[tar_ndx]]
		pt2.tgtf = [self.tgtf[tar_ndx]]
		pt2.monf = [self.monf[tar_ndx]]

		dataloader = ke.DataLoader(self)
		print('Finished')


		print('Fetching IRI Data... ', end='')
		sys.stdout.flush()

		#dataloader.fetch_iri_data()
		print('Finished')

		print('Running PyCPT... ', end='')
		sys.stdout.flush()
		#pt.run_test()
		print('Finished')
		print('Pre-processing Data... ', end='')
		sys.stdout.flush()

		ct = dataloader.read_all_data(big=True)
		#for i in range(len(ct)):
		#	for k in range(len(ct[i])):
		#		print(ct[i][k].t[0])

		print('Finished')
		NE = np.empty([dataloader.n_years[tar_ndx], len(dataloader.climatetensors[tar_ndx][0].obs_lat), len(dataloader.climatetensors[tar_ndx][0].obs_lon)])
		count = 0
		for k in range(dataloader.n_years[tar_ndx]):
			print('Training CNN, skipping {}...  '.format(dataloader.climatetensors[tar_ndx][k].t[0]), end='')
			sys.stdout.flush()
			coords = []
			cn = ke.CNNensemble(6, dataloader, ntrain=1)
			optimizer = te.optim.SGD(cn.parameters(), lr=0.2)
			cn.add_optimizer(optimizer)
			criterion=te.nn.MSELoss()
			cn.add_criterion(criterion)

			for l in range(dataloader.n_years[tar_ndx]):
				if l != k:
					coords.append([tar_ndx, l, k])
			cn.batch_train(coords,epochs=count, big=True)
			count += 1
			#cn.visualize()
			#cn.save('./toeb{}_{}.cnn'.format(k, dataloader.tgts[j]))

			#print('Validating on {}... '.format(dataloader.climatetensors[j][k].t[0]), end='')
			#print('Producting predictions for {} {}... '.format(self.tgts[tar_ndx], dataloader.climatetensors[tar_ndx][k].t[0]), end='')
			#sys.stdout.flush()
			coords = [[tar_ndx, k, k]]
			preds = cn.predict(coords, big=True, Test=True)
			#NG = cn.dataloader.pycpt.get_NGensemble(j)
			NE[k,:,:] = preds
			for coord in range(preds.shape[0]):
				cn.dataloader.refresh(tar_ndx, k)
				cn.dataloader.climatetensors[tar_ndx][k].add_cnn_ensemble(preds[coord, :, :])
				#cn.dataloader.climatetensors[tar_ndx][k].add_nextgen_ensemble(NG, k)
		#	print('Finished')
				#cn.dataloader.climatetensors[j][k].compare( coords[coord][1],tgts[coords[coord][0]] )
			#print('Finished. Check Validation{}_{}.png for output.'.format(dataloader.climatetensors[j][k].t[0],tgts[j]))
		if self.file=='FCST_xvPr':
			self.writeCPT(tar_ndx, NE,'./input/Neural_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.mons[tar_ndx]+'.tsv')
			print('Cross-validated prediction files successfully produced')
		if self.file=='FCST_mu':
			self.writeCPT(tar_ndx, NE,'./output/Neural_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_mu_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'.tsv')
			print('Forecast files successfully produced')
		if self.file=='FCST_var':
			self.writeCPT(tar_ndx, NE,'./output/Neural_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_var_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'.tsv')
			print('Forecast error files successfully produced')


	def NGensemble(self, tar_ndx):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return

		nmods=len(self.models)

		W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,self.tgts[tar_ndx],self.monf[tar_ndx],self.fyr)

		ens  =np.empty([nmods,T,H,W])  #define array for later use

		k=-1
		for model in self.models:
			print('Preparing CPT files for '+model+' and initialization '+self.mons[tar_ndx]+'...')

			k=k+1 #model
			memb0=np.empty([T,H,W])  #define array for later use

			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+self.file+'_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'.dat','rb')
			#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
			for it in range(T):
				#Now we read the field
				if platform.system() == "Windows":
					garb = struct.unpack('s', f.read(1))[0]
				recl=struct.unpack('i',f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize) #this if for each time stamp
				A0=np.fromfile(f,dtype='float32',count=numval)
				endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
				if platform.system() == "Windows":
					garb = struct.unpack('s', f.read(1))[0]
				memb0[it,:,:]= np.transpose(A0.reshape((W, H), order='F'))

			memb0[memb0==-999.]=np.nan #identify NaNs

			ens[k,:,:,:]=memb0

		# NextGen ensemble mean (perhaps try median too?)
		NG=np.nanmean(ens, axis=0)  #axis 0 is ensemble member

		#Now write output:
		#writeCPT(NG,'./output/NextGen_'+fprefix+'_'+tar+'_ini'+mon+'.tsv',models,fprefix,predictand,mpref,id,tar,mon,tgti,tgtf,monf,fyr)
		if self.file=='FCST_xvPr':
			self.writeCPT(tar_ndx, NG,'./input/NextGen_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.mons[tar_ndx]+'.tsv')
			print('Cross-validated prediction files successfully produced')
		if self.file=='FCST_mu':
			self.writeCPT(tar_ndx, NG,'./output/NextGen_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_mu_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'.tsv')
			print('Forecast files successfully produced')
		if self.file=='FCST_var':
			self.writeCPT(tar_ndx, NG,'./output/NextGen_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_var_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(fyr)+'.tsv')
			print('Forecast error files successfully produced')

	def get_NGensemble(self, tar_ndx):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return

		nmods=len(self.models)

		W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,self.tgts[tar_ndx],self.monf[tar_ndx],self.fyr)

		ens  =np.empty([nmods,T,H,W])  #define array for later use

		k=-1
		for model in self.models:
			#print('Preparing CPT files for '+model+' and initialization '+self.mons[tar_ndx]+'...')

			k=k+1 #model
			memb0=np.empty([T,H,W])  #define array for later use

			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('./output/'+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+self.file+'_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'.dat','rb')
			#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
			for it in range(T):
				#Now we read the field
				if platform.system() == "Windows":
					garb = struct.unpack('s', f.read(1))[0]
				recl=struct.unpack('i',f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize) #this if for each time stamp
				A0=np.fromfile(f,dtype='float32',count=numval)
				endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
				if platform.system() == "Windows":
					garb = struct.unpack('s', f.read(1))[0]
				memb0[it,:,:]= np.transpose(A0.reshape((W, H), order='F'))

			memb0[memb0==-999.]=np.nan #identify NaNs

			ens[k,:,:,:]=memb0

		# NextGen ensemble mean (perhaps try median too?)
		NG=np.nanmean(ens, axis=0)  #axis 0 is ensemble member
		return NG

	def plt_ng_probabilistic(self, use_shp_file=False):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return

		if use_shp_file:
			try:
				self.shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
			except:
				print('Failed to load custom shape file')


		self._tempmods = copy.deepcopy(self.models)
		self.models = ['NextGen']
		cbar_loc, fancy = 'bottom', True
		nmods=len(self.models)
		nsea=len(self.tgts)
		xdim=1
		#
		list_probabilistic_by_season = [[[], [], []] for i in range(nsea)]
		list_det_by_season = [[] for i in range(nsea)]
		for i in range(nmods):
			for j in range(nsea):
				if platform.system() == "Windows":
					plats, plongs, av = self.read_forecast_bin('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				else:
					plats, plongs, av = self.read_forecast('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				for kl in range(av.shape[0]):
					list_probabilistic_by_season[j][kl].append(av[kl])
				if platform.system() == "Windows":
					dlats, dlongs, av = self.read_forecast_bin('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				else:
					dlats, dlongs, av = self.read_forecast('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				list_det_by_season[j].append(av[0])

		ng_probfcst_by_season = []
		ng_detfcst_by_season = []
		pbn, pn, pan = [],[],[]
		for j in range(nsea):
			p_bn_array = np.asarray(list_probabilistic_by_season[j][0])
			p_n_array = np.asarray(list_probabilistic_by_season[j][1])
			p_an_array = np.asarray(list_probabilistic_by_season[j][2])

			p_bn = np.nanmean(p_bn_array, axis=0) #average over the models
			p_n = np.nanmean(p_n_array, axis=0)   #some areas are NaN
			p_an = np.nanmean(p_an_array, axis=0) #if they are Nan for All, mark

			all_nan = np.zeros(p_bn.shape)
			for ii in range(p_bn.shape[0]):
				for jj in range(p_bn.shape[1]):
					if np.isnan(p_bn[ii,jj]) and np.isnan(p_n[ii,jj]) and np.isnan(p_an[ii,jj]):
						all_nan[ii,jj] = 1
			missing = np.where(all_nan > 0)

			max_ndxs = np.argmax(np.asarray([p_bn, p_n, p_an]), axis=0)
			p_bn[np.where(max_ndxs!= 0)] = np.nan
			p_n[np.where(max_ndxs!= 1)] = np.nan
			p_an[np.where(max_ndxs!= 2)] = np.nan
			pbn.append(p_bn)
			pn.append(p_n)
			pan.append(p_an)

		fig, ax = plt.subplots(nrows=xdim, ncols=nsea, figsize=(nsea*13, xdim*10), sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

		if nsea == 1:
			ax = [ax]
		ax = [ax]

		for i in range(xdim):
			for j in range(nsea):
				current_cmap = plt.get_cmap('BrBG')
				current_cmap.set_under('white', 0.0)

				current_cmap_copper = plt.get_cmap('YlOrRd', 9)
				current_cmap_binary = plt.get_cmap('Greens', 4)
				current_cmap_ylgn = self.make_cmap_blue(9)

				lats, longs = plats, plongs

				ax[i][j].set_extent([longs[0],longs[-1],lats[0],lats[-1]], ccrs.PlateCarree())

				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
		#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				#pl.xlabels_bottom = False
				#if i == nmods - 1: change so long vals in every plot
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				pl.xlabel_style = {'size': 8}#'rotation': 'vertical'}
				if use_shp_file:
					try:
						ax[i][j].add_feature(self.shape_feature, edgecolor='black')
					except:
						print('failed to load your shapefile')
				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')

				ax[i][j].set_ybound(lower=self.sla2, upper=self.nla2)
				titles = ["Deterministic Forecast", "Probabilistic Forecast (Dominant Tercile)"]


				if j == 0:
					ax[i][j].text(-0.25, 0.5, "Probabilistic Forecast (Dominant Tercile)",rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)

				labels = ['Rainfall (mm)', 'Probability (%)']
				ax[i][j].set_title(self.tgts[j])


				#fancy probabilistic
				CS1 = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), pbn[j],
					vmin=35, vmax=80,
					#norm=MidpointNormalize(midpoint=0.),
					cmap=current_cmap_copper)
				CS2 = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), pn[j],
					vmin=35, vmax=55,
					#norm=MidpointNormalize(midpoint=0.),
					cmap=current_cmap_binary)
				CS3 = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), pan[j],
					vmin=35, vmax=80,
					#norm=MidpointNormalize(midpoint=0.),
					cmap=current_cmap_ylgn)

				bounds = [40,45,50,55,60,65,70,75]
				nbounds = [40,45,50]

				#fancy probabilistic cb bottom
				axins_f_bottom = inset_axes(ax[i][j],
	            	width="40%",  # width = 5% of parent_bbox width
	               	height="5%",  # height : 50%
	               	loc='lower left',
	               	bbox_to_anchor=(-0.2, -0.15, 1.2, 1),
	               	bbox_transform=ax[i][j].transAxes,
	               	borderpad=0.1 )
				axins2_bottom = inset_axes(ax[i][j],
	            	width="20%",  # width = 5% of parent_bbox width
	               	height="5%",  # height : 50%
	               	loc='lower center',
	               	bbox_to_anchor=(-0.0, -0.15, 1, 1),
	               	bbox_transform=ax[i][j].transAxes,
	               	borderpad=0.1 )
				axins3_bottom = inset_axes(ax[i][j],
	            	width="40%",  # width = 5% of parent_bbox width
	               	height="5%",  # height : 50%
	               	loc='lower right',
	               	bbox_to_anchor=(0, -0.15, 1.2, 1),
	               	bbox_transform=ax[i][j].transAxes,
	               	borderpad=0.1 )
				cbar_fbl = fig.colorbar(CS1, ax=ax[i][j], cax=axins_f_bottom, orientation='horizontal', ticks=bounds)
				cbar_fbl.set_label('BN Probability (%)') #, rotation=270)\

				cbar_fbc = fig.colorbar(CS2, ax=ax[i][j],  cax=axins2_bottom, orientation='horizontal', ticks=nbounds)
				cbar_fbc.set_label('N Probability (%)') #, rotation=270)\

				cbar_fbr = fig.colorbar(CS3, ax=ax[i][j],  cax=axins3_bottom, orientation='horizontal', ticks=bounds)
				cbar_fbr.set_label('AN Probability (%)') #, rotation=270)\

		fig.savefig('./images/NG_Probabilistic_RealtimeForecasts.png', dpi=500, bbox_inches='tight')
		if self.showPlot:
			plt.show()
		else:
			plt.close()
		self.models = self._tempmods

	def read_forecast_bin(self, fcst_type, model, predictand, mpref, mons, mon, fyr):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		if fcst_type == 'deterministic':
			f = open("./output/" + model + '_' + predictand + predictand +'_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.dat', 'rb')
		elif fcst_type == 'probabilistic':
			f = open("./output/" + model + '_' + predictand +predictand + '_' + mpref + 'FCST_P_' +mons + '_' +mon+str(fyr)+'.dat', 'rb')
		else:
			print('invalid fcst_type')
			return

		if fcst_type == 'deterministic':
			with  open("./output/" + model + '_' + predictand +predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.ctl', 'r') as fp:
				for line in self.lines_that_contain("XDEF", fp): #lons
					W = int(line.split()[1])
					XD= float(line.split()[4])
					Wi = float(line.split()[3])
			with  open("./output/" + model + '_' + predictand + predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.ctl', 'r') as fp:
				for line in self.lines_that_contain("YDEF", fp):  #lats
					H = int(line.split()[1])
					YD= float(line.split()[4])
					Hi = float(line.split()[3])


			garb = struct.unpack('s', f.read(1))[0]
			recl = struct.unpack('i', f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			A0 = np.fromfile(f, dtype='float32', count=numval)
			var = np.transpose(A0.reshape((W, H), order='F'))
			var[var==-999.]=np.nan #only sensible values
			recl = struct.unpack('i', f.read(4))[0]
			garb = struct.unpack('s', f.read(1))[0]
			lats, lons = np.linspace(Hi+H*YD, Hi, num=H+1), np.linspace(Wi, Wi+W*XD, num=W+1)
			return lats, lons, np.asarray([var])

		if fcst_type == 'probabilistic':
			with  open("./output/" + model + '_' + predictand + predictand +'_' + mpref + 'FCST_P_' +mons + '_' +mon+str(fyr)+'.ctl', 'r') as fp:
				for line in self.lines_that_contain("XDEF", fp): #lons
					W = int(line.split()[1])
					XD= float(line.split()[4])
					Wi = float(line.split()[3])
			with  open("./output/" + model + '_' + predictand +predictand + '_' + mpref + 'FCST_P_' +mons + '_' +mon+str(fyr)+'.ctl', 'r') as fp:
				for line in self.lines_that_contain("YDEF", fp):  #lats
					H = int(line.split()[1])
					YD= float(line.split()[4])
					Hi = float(line.split()[3])

			vars = []
			for ii in range(3):
				garb = struct.unpack('s', f.read(1))[0]
				recl = struct.unpack('i', f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize)
				A0 = np.fromfile(f, dtype='float32', count=numval)
				var = np.transpose(A0.reshape((W, H), order='F'))
				var[var==-1.]=np.nan #only sensible values
				recl = struct.unpack('i', f.read(4))[0]
				garb = struct.unpack('s', f.read(1))[0]
				vars.append(var)
			lats, lons = np.linspace(Hi+H*YD, Hi, num=H+1), np.linspace(Wi, Wi+W*XD, num=W+1)
			return lats, lons, np.asarray(vars)


	def read_forecast(self, fcst_type, model, predictand, mpref, mons, mon, fyr):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		if fcst_type == 'deterministic':
			f = open("./output/" + model + '_' + predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.txt', 'r')
		elif fcst_type == 'probabilistic':
			f = open("./output/" + model + '_' + predictand + '_' + mpref + 'FCST_P_' +mons + '_' +mon+str(fyr)+'.txt', 'r')
		else:
			print('invalid fcst_type')
			return
		lats, all_vals, vals = [], [], []
		flag = 0
		for line in f:
			if line[0:4] == 'cpt:':
				if flag == 2:
					vals = np.asarray(vals, dtype=float)
					if fcst_type == 'deterministic':
						vals[vals == -999.0] = np.nan
					if fcst_type == 'probabilistic':
						vals[vals == -1.0] = np.nan
					all_vals.append(vals)
					lats = []
					vals = []
				flag = 1
			elif flag == 1 and line[0:4] != 'cpt:':
				longs = line.strip().split('\t')
				longs = [float(i) for i in longs]
				flag = 2
			elif flag == 2:
				latvals = line.strip().split('\t')
				lats.append(float(latvals.pop(0)))
				vals.append(latvals)
		vals = np.asarray(vals, dtype=float)
		if fcst_type == 'deterministic':
			vals[vals == -999.0] = np.nan
		if fcst_type == 'probabilistic':
			vals[vals == -1.0] = np.nan
		all_vals.append(vals)
		all_vals = np.asarray(all_vals)
		return lats, longs, all_vals

	def ensemblefiles(self,models,work):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		if platform.system() == 'Windows':
			self.callSys("cd output && mkdir NextGen")
			self.callSys("cd output\\NextGen &&  del /s /q *_NextGen.tgz")
			self.callSys("cd output\\NextGen && del /s /q *.txt")
			for i in range(len(self.models)):
				self.callSys("cd " + os.path.normpath("output/") + " && copy "+ os.path.normpath("*"+self.models[i]+"*.ctl") + " NextGen")
				self.callSys("cd " + os.path.normpath("output/") + " && copy "+ os.path.normpath("*"+self.models[i]+"*.dat") + " NextGen")

			self.callSys("cd " + os.path.normpath("output/") + " && copy "+ os.path.normpath("*NextGen*.ctl") + " NextGen")
			self.callSys("cd " + os.path.normpath("output/") + " && copy "+ os.path.normpath("*NextGen*.dat") + " NextGen")

			self.callSys("cd " + os.path.normpath("output/NextGen/") + " && tar cvzf " + work + "_NextGen.tgz *") #this ~should~ be fine ? unless they have a computer older than last march 2019
			self.callSys("cd " + os.path.normpath("output/NextGen/") + " && del /s /q *.ctl")
			self.callSys("cd " + os.path.normpath("output/NextGen/") + " && del /s /q *.dat")
			self.callSys("echo %cd%")

		else:
			self.callSys("mkdir ./output/NextGen/") #this is fine
			self.callSys("cd ./output/NextGen/; rm -Rf *_NextGen.tgz *.txt")
			for i in range(len(self.models)):
				self.callSys("cd ./output/NextGen; cp ../*"+self.models[i]+"*.txt .")
			self.callSys("cd ./output/NextGen; cp ../*NextGen*.txt .")
			self.callSys("cd ./output/NextGen; tar cvzf " + work+"_NextGen.tgz *.txt") #this ~should~ be fine ? unless they have a computer older than last march 2019
			self.callSys("cd ./output/NextGen/; rm -Rf *.txt")
			self.callSys('pwd')

		print("Compressed file "+work+"_NextGen.tgz created in output/NextGen/")
		print("Now send that file to your contact at the IRI")

	def plt_ng_deterministic(self, use_shp_file=False):
		if self.isInitialized == 0:
			print("You need to initialize this PyCPT Run by loading from file, or by passing the arguments, or by passing an argument dictionary!")
			return
		if use_shp_file:
			try:
				self.shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
			except:
				print('failed to load shp file')
		self._tempmods = copy.deepcopy(self.models)
		self.models=['NextGen']
		cbar_loc, fancy = 'bottom', True
		nmods=len(self.models)
		nsea=len(self.tgts)
		xdim = 1
		list_probabilistic_by_season = [[[], [], []] for i in range(nsea)]
		list_det_by_season = [[] for i in range(nsea)]
		for i in range(nmods):
			for j in range(nsea):
				if platform.system() == "Windows":
					plats, plongs, av = self.read_forecast_bin('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				else:
					plats, plongs, av = self.read_forecast('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				list_probabilistic_by_season[j][0].append(av[0])
				list_probabilistic_by_season[j][1].append(av[1])
				list_probabilistic_by_season[j][2].append(av[2])
				if platform.system() == "Windows":
					dlats, dlongs, av = self.read_forecast_bin('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				else:
					dlats, dlongs, av = self.read_forecast('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
				list_det_by_season[j].append(av[0])

		ng_probfcst_by_season = []
		ng_detfcst_by_season = []
		for j in range(nsea):
			d_array = np.asarray(list_det_by_season[j])
			d_nanmean = np.nanmean(d_array, axis=0)
			ng_detfcst_by_season.append(d_nanmean)

		fig, ax = plt.subplots(nrows=xdim, ncols=nsea, figsize=(nsea*13, xdim*10), sharex=True,sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
		if nsea == 1:
			ax = [ax]
		ax = [ax]




		for i in range(xdim):
			for j in range(nsea):
				current_cmap = plt.get_cmap('BrBG')
				current_cmap.set_bad('white',0.0)
				current_cmap.set_under('white', 0.0)

				lats, longs = dlats, dlongs
				ax[i][j].set_extent([longs[0],longs[-1],lats[0],lats[-1]], ccrs.PlateCarree())

				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
		#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)
				if use_shp_file:
					try:
						ax[i][j].add_feature(self.shape_feature, edgecolor='black')
					except:
						print('failed to load shp file')

				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')

				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				#pl.xlabels_bottom = False
				#if i == nmods - 1: change so long vals in every plot
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				ax[i][j].add_feature(states_provinces, edgecolor='black')
				ax[i][j].set_ybound(lower=self.sla2, upper=self.nla2)
				pl.xlabel_style = {'size': 8}#'rotation': 'vertical'}

				titles = ["Deterministic Forecast", "Probabilistic Forecast (Dominant Tercile)"]


				if j == 0:
					ax[i][j].text(-0.25, 0.5, "Deterministic Forecast",rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)

				labels = ['Rainfall (mm)', 'Probability (%)']
				ax[i][j].set_title(self.tgts[j])

				#fancy deterministic
				var = ng_detfcst_by_season[j]
				CS_det = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), var,
					norm=MidpointNormalize(midpoint=0.),
					cmap=current_cmap)

				if cbar_loc == 'left':
					#fancy deterministic cb left
					axins_det = inset_axes(ax[i][j],
		            	width="5%",  # width = 5% of parent_bbox width
		               	height="100%",  # height : 50%
		               	loc='center left',
		               	bbox_to_anchor=(-0.25, 0., 1, 1),
		               	bbox_transform=ax[i][j].transAxes,
		               	borderpad=0.1 )
					cbar_ldet = fig.colorbar(CS_det, ax=ax[i][j], cax=axins_det,  orientation='vertical', pad=0.02)
					cbar_ldet.set_label(labels[i]) #, rotation=270)\
					axins_det.yaxis.tick_left()
				else:
					#fancy deterministic cb bottom
					axins_det = inset_axes(ax[i][j],
		            	width="100%",  # width = 5% of parent_bbox width
		               	height="5%",  # height : 50%
		               	loc='lower center',
		               	bbox_to_anchor=(-0.1, -0.15, 1.1, 1),
		               	bbox_transform=ax[i][j].transAxes,
		               	borderpad=0.1 )
					cbar_bdet = fig.colorbar(CS_det, ax=ax[i][j],  cax=axins_det, orientation='horizontal', pad = 0.02)
					cbar_bdet.set_label(labels[i])
		fig.savefig('./images/NG_Deterministic_RealtimeForecasts.png', dpi=500, bbox_inches='tight')
		if self.showPlot:
			plt.show()
		else:
			plt.close()
		self.models = self._tempmods

	def my_call(self, arg):
		if self.showPlot:
			get_ipython().system(arg)
		else:
			x = subprocess.call(arg, shell=True )


	#set up the required directory structure lol
	def setup_directories(self, showPlot=True):
		os.chdir(self.workdir)

		if self.force_download and os.path.isdir(self.work):
			if platform.system() == 'Windows':
				print('Windows deleting folders')
				self.my_call('del /S /Q {}/{}'.format(self.work))
				self.my_call('rmdir /S /Q {}/{}'.format(self.work + '/scripts'))
				self.my_call('rmdir /S /Q {}/{}'.format(self.work + '/input'))
				self.my_call('rmdir /S /Q {}/{}'.format(self.work + '/output'))
				self.my_call('rmdir /S /Q {}/{}'.format(self.work + '/images'))
				self.my_call('rmdir /S /Q {}/{}'.format(self.work))
			else:
				print('Mac deleting folders')
				self.my_call('rm -rf {}'.format(self.work))

		if not os.path.isdir(self.work):
			self.my_call('mkdir {}'.format(self.work))

		if not os.path.isdir(self.work + '/scripts'):
			if platform.system() == 'Windows':
				self.my_call('cd {} && mkdir scripts'.format(self.work))
			else:
				self.my_call('mkdir {}/scripts'.format(self.work))

		if not os.path.isdir(self.work + '/images'):
			if platform.system() == "Windows":
				self.my_call('cd {} && mkdir images'.format(self.work))
			else:
				self.my_call('mkdir {}/images'.format(self.work))

		if not os.path.isdir(self.work + '/input'):
			if platform.system() == "Windows":
				self.my_call('cd {} && mkdir input'.format(self.work))
			else:
				self.my_call('mkdir {}/input'.format(self.work))

		if not os.path.isdir(self.work + '/output'):
			if platform.system() == "Windows":
				self.my_call('cd {} && mkdir output'.format(self.work))
			else:
				self.my_call('mkdir {}/output'.format(self.work))

		os.environ["CPT_BIN_DIR"] = self.cptdir
		os.chdir(self.work)

	def run_test(self):
		self.setup_directories()
		self.pltdomain()

		for model in range(len(self.models)):
			for tgt in range(len(self.mons)):
				self.setupParams(tgt)
				self.prepFiles(tgt, model)
				self.CPTscript(tgt, model)
				self.run(tgt, model)

		for ime in range(len(self.met)):
			self.pltmap(ime)

		for imod in range(self.eof_modes):
			self.plteofs(imod)

		if len(self.models) > 2:
			models = self.models[0: int(len(self.models)/2) ]
		else:
			models = self.models[0]

		self.setNextGenModels(models)

		for tgt in range(len(self.tgts)):
			self.NGensemble(tgt)
			self.CPTscript(tgt)
			self.run(tgt)

		for ime in range(len(self.met)):
			self.pltmap(ime, isNextGen=1)

		self.plt_ng_deterministic()
		self.plt_ng_probabilistic()

		self.ensemblefiles(['NextGen'], self.work)
















def exceedprob(x,dof,lo,sc):
	return t.sf(x, dof, loc=lo, scale=sc)*100

def skilltab(score,wknam,lon1,lat1,lat2,lon2,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk):


	#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
	with open('./output/'+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
		for line in lines_that_contain("XDEF", fp):
			W = int(line.split()[1])
			XD= float(line.split()[4])
	with open('./output/'+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
		for line in lines_that_contain("YDEF", fp):
			H = int(line.split()[1])
			YD= float(line.split()[4])

	#Find the gridbox:
	lonrange = np.linspace(loni, loni+W*XD,num=W)
	latrange = np.linspace(lati+H*YD, lati, num=H)  #need to reverse the latitudes because of CPT (GrADS YREV option)
	lon_grid, lat_grid = np.meshgrid(lonrange, latrange)
	#first point
	a = abs(lat_grid-lat1)+abs(lon_grid-lon1)
	i1,j1 = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude
	#second point
	a = abs(lat_grid-lat2)+abs(lon_grid-lon2)
	i2,j2 = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude

	df = pd.DataFrame(index=wknam[0:nwk])
	for L in range(nwk):
		wk=L+1
		for S in score:
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('./output/'+model+'_'+fprefix+'_'+mpref+'_'+str(S)+'_'+training_season+'_wk'+str(wk)+'.dat','rb')
			if platform.system() == "Windows":
				garb = struct.unpack('s', f.read(1))[0]
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var = np.transpose(A.reshape((W, H), order='F'))
			var[var==-999.]=np.nan #only sensible values
			df.at[wknam[L], str(S)] = round(np.nanmean(np.nanmean(var[i1:i2,j1:j2], axis=1), axis=0),2)
			df.at[wknam[L], 'max('+str(S)+')']  = round(np.nanmax(var[i1:i2,j1:j2]),2)
			df.at[wknam[L], 'min('+str(S)+')']  = round(np.nanmin(var[i1:i2,j1:j2]),2)
	return df
	f.close()

def pltmapProb(loni,lone,lati,late,fprefix,mpref,training_season, mon, fday, nwk):

	#Need this score to be defined by the calibration method!!!
	score = 'CCAFCST_P'

	plt.figure(figsize=(15,20))

	for L in range(nwk):
		wk=L+1
		#Read grads binary file size H, W  --it assumes that 2AFC file exists (template for final domain size)
		with open('./output/'+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('./output/'+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])

		#Prepare to read grads binary file  [float32 for Fortran sequential binary files]
		Record = np.dtype(('float32', H*W))

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
#			name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')


		f=open('./output/'+model+'_'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat','rb')

		tit=['Below Normal','Normal','Above Normal']
		for i in range(3):
				#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
				if platform.system() == "Windows":
					garb = struct.unpack('s', f.read(1))[0]
				recl=struct.unpack('i',f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize)
				#We now read the field for that record (probabilistic files have 3 records: below, normal and above)
				B=np.fromfile(f,dtype='float32',count=numval) #astype('float')
				endrec=struct.unpack('i',f.read(4))[0]
				if platform.system() == "Windows":
					garb = struct.unpack('s', f.read(1))[0]
				var = np.flip(np.transpose(B.reshape((W, H), order='F')),0)
				var[var<0]=np.nan #only positive values
				ax2=plt.subplot(nwk, 3, (L*3)+(i+1),projection=ccrs.PlateCarree())
				ax2.set_title("Week "+str(wk)+ ": "+tit[i])
				ax2.add_feature(feature.LAND)
				ax2.add_feature(feature.COASTLINE)
				#ax2.set_ybound(lower=lati, upper=late)
				pl2=ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl2.xlabels_top = False
				pl2.ylabels_left = True
				pl2.ylabels_right = False
				pl2.xformatter = LONGITUDE_FORMATTER
				pl2.yformatter = LATITUDE_FORMATTER
				ax2.add_feature(states_provinces, edgecolor='gray')
				ax2.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

				#ax2.set_ybound(lower=lati, upper=late)
				#ax2.set_xbound(lower=loni, upper=lone)
				#ax2.set_adjustable('box')
				#ax2.set_aspect('auto',adjustable='datalim',anchor='C')
				CS=ax2.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati,lati+H*YD, num=H), var,
				vmin=0,vmax=100,
				cmap=plt.cm.RdYlBu,
				transform=ccrs.PlateCarree())
				#plt.show(block=False)

	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.15, top=0.9)
	cax = plt.axes([0.2, 0.08, 0.6, 0.04])
	cbar = plt.colorbar(CS,cax=cax, orientation='horizontal', pad=0.01)
	cbar.set_label('Probability (%)') #, rotation=270)
	f.close()

def pltmapff(models,predictand,thrs,ntrain,loni,lone,lati,late,fprefix,mpref,monf,fyr,mons,tgts, obs):

	#Implement: read degrees of freedom from CPT file
	#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
	if obs == 'ENACTS-BD':
		x_offset = 0.6
		y_offset = 0.4
	else:
		x_offset = 0
		y_offset = 0
	dof=ntrain
	nmods=len(models)
	tar=tgts[mons.index(monf)]
	#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
	with open('./output/'+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("XDEF", fp):
			W = int(line.split()[1])
			XD= float(line.split()[4])
	with open('./output/'+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("YDEF", fp):
			H = int(line.split()[1])
			YD= float(line.split()[4])

	#plt.figure(figsize=(15,20))
	fig, ax = plt.subplots(figsize=(6,6),sharex=True,sharey=True)
	k=0
	for model in models:
		k=k+1
		#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
		f=open('./output/'+model+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		muf = np.transpose(A.reshape((W, H), order='F'))
		muf[muf==-999.]=np.nan #only sensible values

		#Read variance
		f=open('./output/'+model+'_'+fprefix+predictand+'_'+mpref+'FCST_var_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		if platform.system() == "Windows":
			garb = struct.unpack('s', f.read(1))[0]
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		vari = np.transpose(A.reshape((W, H), order='F'))
		vari[vari<0.]=np.nan #only positive values

		#Compute scale parameter for the t-Student distribution
		scalef=np.sqrt((dof-2)/dof*vari)

		fprob = exceedprob(thrs,dof,muf,scalef)

		ax = plt.subplot(nmods, 1, k, projection=ccrs.PlateCarree())
		ax.set_extent([loni+x_offset,loni+W*XD+x_offset,lati+y_offset,lati+H*YD+y_offset], ccrs.PlateCarree())

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
#			name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')

		ax.add_feature(feature.LAND)
		ax.add_feature(feature.COASTLINE)

		if k==1:
			ax.set_title('Probability (%) of Exceeding '+str(thrs)+" mm/day")
		#current_cmap = plt.cm.get_cmap('RdYlBu', 10)
		current_cmap = self.make_cmap(10)
		#current_cmap.set_bad('white',1.0)
		current_cmap.set_under('white', 1.0)
		pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
			linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
		pl.xlabels_top = False
		pl.ylabels_left = True
		pl.ylabels_right = False
		pl.xformatter = LONGITUDE_FORMATTER
		pl.yformatter = LATITUDE_FORMATTER
		ax.add_feature(states_provinces, edgecolor='gray')
		ax.set_ybound(lower=lati+y_offset, upper=late+y_offset)
		CS=plt.pcolormesh(np.linspace(loni+x_offset, loni+W*XD+x_offset,num=W), np.linspace(lati+H*YD+y_offset, lati+y_offset, num=H), fprob,
			vmin=0,vmax=100,
			cmap=current_cmap,
			transform=ccrs.PlateCarree())
		label = 'Probability (%) of Exceedance'

		#plt.autoscale(enable=True)

		#plt.tight_layout()
		plt.subplots_adjust(hspace=0)
		plt.subplots_adjust(bottom=0.15, top=0.9)
		cbar = fig.colorbar(CS,ax=ax, orientation='vertical', pad=0.05)

		# for i, row in enumerate(ax):
		# 	for j, cell in enumerate(row):
		# 		if i == len(ax) - 1:
		# 			cell.set_xlabel("noise column: {0:d}".format(j + 1))
		# 		if j == 0:
		# 			cell.set_ylabel("noise row: {0:d}".format(i + 1))

		ax.set_ylabel(model, rotation=90)
		cbar.set_label(label) #, rotation=270)
		f.close()

def pltprobff(models,predictand,thrs,ntrain,lon,lat,loni,lone,lati,late,fprefix,mpref,monf,fyr,mons,tgts):

	#Implement: read degrees of freedom from CPT file
	#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
	dof=ntrain
	tar=tgts[mons.index(monf)]
	#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
	with open('./output/'+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("XDEF", fp):
			W = int(line.split()[1])
			XD= float(line.split()[4])
	with open('./output/'+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("YDEF", fp):
			H = int(line.split()[1])
			YD= float(line.split()[4])
	with open('./output/'+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_Obs_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("TDEF", fp):
			T = int(line.split()[1])
			TD= 1  #not used

	#Find the gridbox:
	lonrange = np.linspace(loni, loni+W*XD,num=W)
	latrange = np.linspace(lati+H*YD, lati, num=H)  #need to reverse the latitudes because of CPT (GrADS YREV option)
	lon_grid, lat_grid = np.meshgrid(lonrange, latrange)
	a = abs(lat_grid-lat)+abs(lon_grid-lon)
	i,j = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude

	#Now compute stuff and plot
	#plt.figure(figsize=(15,15))

	k=0
	for model in models:
		k=k+1
		#Forecast files--------
		#Read mean
		#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
		f=open('./output/'+model+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		if platform.system() == "Windows":
			garb = struct.unpack('s', f.read(1))[0]
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		muf = np.transpose(A.reshape((W, H), order='F'))
		muf[muf==-999.]=np.nan #only sensible values
		muf=muf[i,j]

		#Read variance
		f=open('./output/'+model+'_'+fprefix+predictand+'_'+mpref+'FCST_var_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		if platform.system() == "Windows":
			garb = struct.unpack('s', f.read(1))[0]
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		varf = np.transpose(A.reshape((W, H), order='F'))
		varf[varf==-999.]=np.nan #only sensible values
		varf=varf[i,j]

		#Obs file--------
		#Compute obs mean and variance.
		#
		muc0=np.empty([T,H,W])  #define array for later use
		#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
		f=open('./output/'+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_Obs_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
		for it in range(T):
			#Now we read the field
			if platform.system() == "Windows":
				garb = struct.unpack('s', f.read(1))[0]
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize) #this if for each time stamp
			A0=np.fromfile(f,dtype='float32',count=numval)
			endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
			if platform.system() == "Windows":
				garb = struct.unpack('s', f.read(1))[0]
			muc0[it,:,:]= np.transpose(A0.reshape((W, H), order='F'))

		muc0[muc0==-999.]=np.nan #identify NaNs
		muc=np.nanmean(muc0, axis=0)  #axis 0 is T
		#Compute obs variance
		varc=np.nanvar(muc0, axis=0)  #axis 0 is T
		#Select gridbox values
		muc=muc[i,j]
		#print(muc)   #Test it's actually zero
		varc=varc[i,j]

		#Compute scale parameter for the t-Student distribution
		scalef=np.sqrt(dof*varf)   #due to transformation from Gamma
		scalec=np.sqrt((dof-2)/dof*varc)

		x = np.linspace(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)),max(t.ppf(0.9999, dof, loc=muf, scale=scalef),t.ppf(0.9999, dof, loc=muc, scale=scalec)), 100)

		style = dict(size=10, color='black')

		#cprob = special.erfc((x-muc)/scalec)
		cprob = exceedprob(thrs,dof,muc,scalec)
		fprob = exceedprob(thrs,dof,muf,scalef)
		cprobth = round(t.sf(thrs, dof, loc=muc, scale=scalec)*100,2)
		fprobth = round(t.sf(thrs, dof, loc=muf, scale=scalef)*100,2)
		cpdf=t.pdf(x, dof, loc=muc, scale=scalec)*100
		fpdf=t.pdf(x, dof, loc=muf, scale=scalef)*100
		oddsrc =(fprobth/cprobth)

		fig, ax = plt.subplots(1, 2,figsize=(12,4))
		#font = {'family' : 'Palatino',
		#        'size'   : 16}
		#plt.rc('font', **font)
		#plt.rc('text', usetex=True)
		#plt.rc('font', family='serif')

		#plt.subplot(1, 2, 1)
		ax[0].plot(x, t.sf(x, dof, loc=muc, scale=scalec)*100,'b-', lw=5, alpha=0.6, label='clim')
		ax[0].plot(x, t.sf(x, dof, loc=muf, scale=scalef)*100,'r-', lw=5, alpha=0.6, label='fcst')
		ax[0].axvline(x=thrs, color='k', linestyle=(0,(2,4)))
		ax[0].plot(thrs, fprobth,'ok')
		ax[0].plot(thrs, cprobth,'ok')
		ax[0].text(thrs+0.05, cprobth, str(cprobth)+'%', **style)
		ax[0].text(thrs+0.05, fprobth, str(fprobth)+'%', **style)
		#ax[0].text(0.1, 10, r'$\frac{P(fcst)}{P(clim)}=$'+str(round(oddsrc,1)), **style)
		ax[0].text(min(t.ppf(0.0001, dof, loc=muf, scale=scalef),t.ppf(0.0001, dof, loc=muc, scale=scalec)), -20, 'P(fcst)/P(clim)='+str(round(oddsrc,1)), **style)
		ax[0].legend(loc='best', frameon=False)
		# Add title and axis names
		ax[0].set_title('Probabilities of Exceedance')
		ax[0].set_xlabel('Rainfall')
		ax[0].set_ylabel('Probability (%)')
		# Limits for the Y axis
		#ax[0].set_xlim(left=min(t.ppf(0.001, dof, loc=muf, scale=scalef),t.ppf(0.001, dof, loc=muc, scale=scalec)),right=max(t.ppf(0.99, dof, loc=muf, scale=scalef),t.ppf(0.99, dof, loc=muc, scale=scalec)))
		#ax[0].set_ylim(left=0, right=1)
		#ax[1].subplot(1, 2, 2)
		ax[1].plot(x, cpdf,'b-', lw=5, alpha=0.6, label='clim')
		ax[1].plot(x, fpdf,'r-', lw=5, alpha=0.6, label='fcst')
		ax[1].axvline(x=thrs, color='k', linestyle=(0,(2,4)))
		ax[1].legend(loc='best', frameon=False)
		# Add title and axis names
		ax[1].set_title('Probability Density Functions')
		ax[1].set_xlabel('Rainfall')
		ax[1].set_ylabel('')
		#ax[1].set_ylim(left=0, right=1)

		# Limits for the Y axis
		#print()
		ax[1].set_xlim(left=min(t.ppf(0.001, dof, loc=muf, scale=scalef),t.ppf(0.001, dof, loc=muc, scale=scalec)),right=max(t.ppf(0.99, dof, loc=muf, scale=scalef),t.ppf(0.99, dof, loc=muc, scale=scalec)))


	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.15, top=0.9)
	#cax = plt.axes([0.2, 0.08, 0.6, 0.04])
	#cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
	#cbar.set_label(label) #, rotation=270)
	f.close()
