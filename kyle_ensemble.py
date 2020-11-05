import torch as tr
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from pycpt_functions_seasonal import *
from scipy.interpolate import griddata
import sys
import pickle as pkl
import copy

#plotting libraries
import cartopy.crs as ccrs
from cartopy import feature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

class ClimateTensor():
	"""Holds model data, ensemble data, & obs data for one year"""
	def __init__(self):
		self.t = [] #stores a list of years for which there is data for each model
		self.lat = [] #stores a list of latitudes for which there is data for each model
		self.lon = [] #stores a list of longitudes for which there is data for each model
		self.data = [] #stores an array of data of shape (nyears, W, H) for each model
		self.N = 0
		self.big = []
		self.new_data = []
		self.model_names = []
		self.nextgen_ensemble = []
		self.cropped_data = []
		self.new_xs, self.new_ys = [], []

	def add_observations(self, data_tuple, yr_ndx):
		""" Adds observation data to the ClimateTensor, where
			data_tuple = (	[years list],
							[lats list],
							[lons list],
							[array of shape ( W, H)] )"""
		self.obs_t = data_tuple[0][yr_ndx]
		self.obs_lat = data_tuple[1]
		self.obs_lon = data_tuple[2]
		self.obs_data = data_tuple[3][yr_ndx]
		self.obs_nla = self.obs_lat[0]
		self.obs_sla = self.obs_lat[-1]
		self.obs_elo = self.obs_lon[-1]
		self.obs_wlo = self.obs_lon[0]

		self.obs_dx = round(self.obs_lat[0] - self.obs_lat[1], 2)
		self.obs_dy = round(self.obs_lon[1] - self.obs_lon[0], 2)
		self.model_names.insert(0, 'Observations')
		self.x_offset, self.y_offset = 0,0

	def plot_inputs(self, year, tgt):
		nsea, nmods = 2, self.N+1
		fig, ax = plt.subplots(nrows=self.N + 1, ncols=nsea, figsize=(5*nsea, 5*nmods),sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]
		vmin,vmax = 0.0,np.nanmax(self.obs_data)
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
		for i in range(nmods):
			for j in range(nsea):
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')


				ax[i][j].add_feature(feature.LAND)
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER

				ax[i][j].add_feature(states_provinces, edgecolor='black')
				ax[i][j].set_ybound(lower=self.obs_sla, upper=self.obs_nla)
				if j == 0:
					ax[i][j].text(-0.42, 0.5, self.model_names[i-1],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
				if i == 0:
					ax[i][j].set_title(tgt)
				if i == 0:
					var = self.obs_data
				else:
					if j ==0:
						var = self.cropped_data[i-1]
					else:
						var = self.data[i-1]
				if i > 0:
					if j == 0:
						ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())
						CS=ax[i][j].pcolormesh(  np.linspace(self.obs_lon[0], self.obs_lon[-1], num=len(self.obs_lon)+1), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat)+1 ),var,
						vmin=vmin, vmax=vmax, #vmax=np.max(var),
						norm=norm,
						cmap=plt.get_cmap('BrBG'),
						transform=ccrs.PlateCarree())
					else:
						ax[i][j].set_extent([self.lon[i-1][0]+self.x_offset,self.lon[i-1][-1]+self.x_offset,self.lat[i-1][-1]+self.y_offset,self.lat[i-1][0]+self.y_offset], ccrs.PlateCarree())
						CS=ax[i][j].pcolormesh(np.linspace(self.lon[i-1][0], self.lon[i-1][-1], num=len(self.lon[i-1])+1), np.linspace(self.lat[i-1][0],self.lat[i-1][-1], num=len(self.lat[i-1])+1 ), var,
						vmin=vmin, vmax=vmax, #vmax=np.max(var),
						norm=norm,
						cmap=plt.get_cmap('BrBG'),
						transform=ccrs.PlateCarree())
				else:
					ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())
					CS=ax[i][j].pcolormesh(np.linspace(self.obs_lon[0], self.obs_lon[-1], num=len(self.obs_lon)+1), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat)+1 ), var,
					vmin=vmin, vmax=vmax, #vmax=np.max(var),
					norm=norm,
					cmap=plt.get_cmap('BrBG'),
					transform=ccrs.PlateCarree())

				axins = inset_axes(ax[i][j],
					   width="5%",  # width = 5% of parent_bbox width
					   height="100%",  # height : 50%
					   loc='center right',
					   bbox_to_anchor=(0., 0., 1.15, 1),
					   bbox_transform=ax[i][j].transAxes,
					   borderpad=0.1,
					   )

				cbar = fig.colorbar(CS, ax=ax[i][j], norm=norm, cax=axins, orientation='vertical', pad=0.02)

		fig.savefig('{}_{}_in.png'.format( tgt, year), dpi=500, bbox_inches='tight')
		if False:
			plt.show()
		plt.close()

	def add_model(self, data_tuple, yr_ndx):
		""" Adds a model's data to the ClimateTensor, where
			data_tuple = (	[years list],
							[lats list],
							[lons list],
							[array of shape (nyears, W, H)]
							name of model )"""
		self.t.append(data_tuple[0][yr_ndx])
		self.lat.append(data_tuple[1])
		self.lon.append(data_tuple[2])
		self.data.append(data_tuple[3][yr_ndx])

		self.N += 1
		self.model_names.append(data_tuple[4])

	def rectify(self):
		""" Cuts out all data across time & space where
			any of the models is missing a datapoint"""


		for i in range(self.N):
			#Increasing resolution to make points overlap with observation data

			self.new_y = copy.deepcopy(self.obs_lat)
			self.cur_y = self.new_y[0]
			while self.cur_y < self.lat[i][0]:
				#print(new_y[0], new_y[-1])
				self.new_y.insert(0,round(self.cur_y + round(self.obs_dy, 3), 3))
				self.cur_y = self.new_y[0]

			self.cur_y = self.new_y[-1]
			while self.cur_y > self.lat[i][-1] - self.obs_dy:
			#	#print(new_y[0], new_y[-1])
				self.new_y.append(round(self.cur_y - round(self.obs_dy, 3), 3))
				self.cur_y = self.new_y[-1]


			self.new_x = copy.deepcopy(self.obs_lon)
			self.cur_x = self.new_x[0]
			while self.cur_x > self.lon[i][0] - self.obs_dx:
				self.new_x.insert(0,round(self.cur_x - round(self.obs_dx, 3), 3))
				self.cur_x = self.new_x[0]

			self.cur_x = self.new_x[-1]
			while self.cur_x <= self.lon[i][-1]:
				self.new_x.append( round(self.cur_x + round(self.obs_dx, 3), 3))
				self.cur_x = self.new_x[-1]

			#print('newx/y', len(self.new_x), len(self.new_y))

			xx, yy = np.mgrid[ self.new_x[0]:self.new_x[-1]:complex(0, len(self.new_x)), self.new_y[0]:self.new_y[-1]:complex(0, len(self.new_y))]

		#	print('xx/yy', xx.shape, yy.shape)
			points, vals = [], []
			for j in range(len(self.lat[i])):
				for k in range(len(self.lon[i])):
					points.append([self.lon[i][k], self.lat[i][j]] )
					vals.append(self.data[i][j][k])

			self.new_data.append( griddata(points, vals, (xx, yy), method='cubic'))
			self.new_xs.append(self.new_x)
			self.new_ys.append(self.new_y)

	def crop(self):
		for i in range(self.N):
			self.xndxs, count = [], 0
			for j in range(len(self.new_xs[i])):
				if self.new_xs[i][j] <= self.obs_lon[-1]  and self.new_xs[i][j] >= self.obs_lon[0]:
					#print(self.obs_lon[0], self.new_xs[i][j], self.obs_lon[-1], count)
					count += 1
					self.xndxs.append(j)
			self.yndxs, count = [], 0
			for j in range(len(self.new_ys[i])):
				if self.new_ys[i][j] <= self.obs_lat[0]  and self.new_ys[i][j] >= self.obs_lat[-1]:
					#print(self.obs_lat[-1], self.new_ys[i][j], self.obs_lat[0], count)
					count += 1
					self.yndxs.append(j)
			#print('x/yndxs', len(self.xndxs), len(self.yndxs))
			self.cropped_data.append([])
			self.cropped_data[i] = self.new_data[i][ self.xndxs[0]:self.xndxs[-1]+1, self.yndxs[0]:self.yndxs[-1]+1]
			#print('cropped', self.cropped_data[i].shape)

			self.cropped_data[i] = np.transpose(self.cropped_data[i])
			#print('transposed', self.cropped_data[i].shape)
			#print()
			nans = np.where(np.isnan(self.obs_data))
			self.cropped_data[i][nans] = np.nan

	def model_data(self):
		"""returns input data for training cnn
		np.array like this:
		[
			[
				[
					[W in # Longitude points] * H in Latitute points
				]
			] * N Models for X tensor (input)

			and then at index = -1:

			[
				[W in # Longitude Points] * H in latitute points
			] * 1 observations data for Y tensor (output )
		]
		"""


		data = np.asarray(self.cropped_data)
		obs = self.obs_data.reshape((1, self.obs_data.shape[0], self.obs_data.shape[1]))
		return np.vstack((data, obs))

	def add_cnn_ensemble(self, data):
		#print(np.nanmean(data), np.nanmax(data))
		self.cnn_ensemble = data

	def add_nextgen_ensemble(self, data, yr_ndx):
		self.nextgen_ensemble = data[yr_ndx]
		#print(data.shape, self.nextgen_ensemble.shape)

	def plot_data(self, year, tgt, big=False):
		nsea, nmods = 3, self.N+1
		fig, ax = plt.subplots(nrows=self.N + 1, ncols=nsea, figsize=(5*nsea, 5*nmods),sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]
		vmin,vmax = 0.0,np.nanmax(self.obs_data)
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
		for i in range(nmods):
			for j in range(nsea):
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')


				ax[i][j].add_feature(feature.LAND)
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER

				ax[i][j].add_feature(states_provinces, edgecolor='black')
				ax[i][j].set_ybound(lower=self.obs_sla, upper=self.obs_nla)
				if j == 0:
					ax[i][j].text(-0.42, 0.5, self.model_names[i-1],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
				if i == 0:
					ax[i][j].set_title(tgt)
				if i == 0:
					var = self.obs_data
				else:
					if j ==0:
						var = self.cropped_data[i-1]
					elif j == 1:
						var = self.data[i-1]
					else:
						var = self.cnn_ensemble
				if i > 0:
					if j == 0:
						ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())
						CS=ax[i][j].pcolormesh(  np.linspace(self.obs_lon[0], self.obs_lon[-1], num=len(self.obs_lon)+1), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat)+1 ),var,
						vmin=vmin, vmax=vmax, #vmax=np.max(var),
						norm=norm,
						cmap=plt.get_cmap('BrBG'),
						transform=ccrs.PlateCarree())
					elif j == 1:
						ax[i][j].set_extent([self.lon[i-1][0]+self.x_offset,self.lon[i-1][-1]+self.x_offset,self.lat[i-1][-1]+self.y_offset,self.lat[i-1][0]+self.y_offset], ccrs.PlateCarree())
						CS=ax[i][j].pcolormesh(np.linspace(self.lon[i-1][0], self.lon[i-1][-1], num=len(self.lon[i-1])+1), np.linspace(self.lat[i-1][0],self.lat[i-1][-1], num=len(self.lat[i-1])+1 ), var,
						#vmin=vmin, vmax=vmax, #vmax=np.max(var),
						#norm=norm,
						cmap=plt.get_cmap('BrBG'),
						transform=ccrs.PlateCarree())
					else:
						if not big:
							ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())
							CS=ax[i][j].pcolormesh(  np.linspace(self.obs_lon[1], self.obs_lon[-2], num=len(self.obs_lon)-2), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat)-2 ),var,
							vmin=vmin, vmax=vmax, #vmax=np.max(var),
							norm=norm,
							cmap=plt.get_cmap('BrBG'),
							transform=ccrs.PlateCarree())
						else:
							ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())
							CS=ax[i][j].pcolormesh(  np.linspace(self.obs_lon[0], self.obs_lon[-1], num=len(self.obs_lon)), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat) ),var,
							vmin=vmin, vmax=vmax, #vmax=np.max(var),
							norm=norm,
							cmap=plt.get_cmap('BrBG'),
							transform=ccrs.PlateCarree())
				else:
					ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())
					CS=ax[i][j].pcolormesh(np.linspace(self.obs_lon[0], self.obs_lon[-1], num=len(self.obs_lon)+1), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat)+1 ), var,
					vmin=vmin, vmax=vmax, #vmax=np.max(var),
					norm=norm,
					cmap=plt.get_cmap('BrBG'),
					transform=ccrs.PlateCarree())

				axins = inset_axes(ax[i][j],
					   width="5%",  # width = 5% of parent_bbox width
					   height="100%",  # height : 50%
					   loc='center right',
					   bbox_to_anchor=(0., 0., 1.15, 1),
					   bbox_transform=ax[i][j].transAxes,
					   borderpad=0.1,
					   )

				cbar = fig.colorbar(CS, ax=ax[i][j],  cax=axins, orientation='vertical', pad=0.02)

		fig.savefig('LeftOut{}_{}.png'.format(  self.t[0], tgt), dpi=500, bbox_inches='tight')
		if False:
			plt.show()

	def compare(self, year, tgt):
		nsea, nmods = 3, 3
		fig, ax = plt.subplots(nrows=nmods, ncols=nsea, figsize=(5*nsea, 5*nmods),sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]
		vmin,vmax = 0.0,np.nanmax(self.obs_data)
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
		vars = [[self.obs_data, self.nextgen_ensemble - self.obs_data, self.cnn_ensemble - self.obs_data],
				[self.obs_data - self.nextgen_ensemble, self.nextgen_ensemble, self.cnn_ensemble - self.nextgen_ensemble],
				[self.obs_data - self.cnn_ensemble, self.nextgen_ensemble - self.cnn_ensemble, self.cnn_ensemble]	]


		for i in range(nmods):
			for j in range(nsea):
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')


				ax[i][j].add_feature(feature.LAND)
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER

				ax[i][j].add_feature(states_provinces, edgecolor='black')
				ax[i][j].set_ybound(lower=self.obs_sla, upper=self.obs_nla)

				if i == 0:
					if j == 0:
						ax[i][j].set_title('Observations')
					elif j == 1:
						ax[i][j].set_title('NextGen')
					else:
						ax[i][j].set_title('Neural Ensemble')

				if j == 0:
					if i == 0:
						ax[i][j].text(-0.42, 0.5, 'Observations',rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
					elif i == 1:
						ax[i][j].text(-0.42, 0.5, 'NextGen',rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
					else:
						ax[i][j].text(-0.42, 0.5, 'Neural Ensemble',rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)

				var = vars[i][j]

				if i == j:
					cm = plt.get_cmap('BrBG')
				else:
					cm = plt.get_cmap('RdYlBu')
				ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())
				CS=ax[i][j].pcolormesh(  np.linspace(self.obs_lon[0], self.obs_lon[-1], num=len(self.obs_lon)), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat) ),var,
				vmin=vmin, vmax=vmax, #vmax=np.max(var),
				norm=norm,
				cmap=cm,
				transform=ccrs.PlateCarree())

				axins = inset_axes(ax[i][j],
					   width="5%",  # width = 5% of parent_bbox width
					   height="100%",  # height : 50%
					   loc='center right',
					   bbox_to_anchor=(0., 0., 1.15, 1),
					   bbox_transform=ax[i][j].transAxes,
					   borderpad=0.1,
					   )

				cbar = fig.colorbar(CS, ax=ax[i][j],  cax=axins, orientation='vertical', pad=0.02)

		fig.savefig('Validation{}_{}.png'.format(  self.t[0], tgt), dpi=500, bbox_inches='tight')
		if False:
			plt.show()

	def __str__(self):
		dct = { 't': self.t,  #stores a list of years for which there is data for each model
				'lat': self.lat[0],  #stores a list of latitudes for which there is data for each model
				'lon': self.lon[0], #stores a list of longitudes for which there is data for each model
				'data': [self.data[i].shape for i in range(len(self.data))],#stores an array of data of shape (nyears, W, H) for each model
				'N': self.N,
				'big': self.big.shape,
				'new_data': [self.new_data[i].shape for i in range(len(self.new_data))],
				'models': self.model_names,
				'cropped': [self.cropped_data[i].shape for i in range(len(self.cropped_data))],
				}#'newxy': (self.new_xs, self.new_ys) }
		return json.dumps(dct, indent=4, sort_keys=True)

	def __repr__(self):
		dct = { 't': self.t,  #stores a list of years for which there is data for each model
				'lat': self.lat[0],  #stores a list of latitudes for which there is data for each model
				'lon': self.lon[0], #stores a list of longitudes for which there is data for each model
				'data': [self.data[i].shape for i in range(len(self.data))],#stores an array of data of shape (nyears, W, H) for each model
				'N': self.N,
				'big': self.big.shape,
				'new_data': [self.new_data[i].shape for i in range(len(self.new_data))],
				'models': self.model_names,
				'cropped': [self.cropped_data[i].shape for i in range(len(self.cropped_data))],
				}#'newxy': (self.new_xs, self.new_ys) }
		return json.dumps(dct, indent=4, sort_keys=True)


class DataLoader():
	def __init__(self, pycpt):
		self.pycpt = pycpt
		self.models = pycpt.models
		self.tgts = pycpt.tgts
		self.n_tgts = len(self.tgts)
		self.n_years = []
		self.mons = pycpt.mons
		self.model_ndx = -1
		self.pycpt.setup_directories()
		self.climatetensors = []

	def read_data(self, f):
		"""reads an input data file and returns its data_tuple"""
		self.model_ndx += 1
		lats, all_vals, vals = [], [], []
		flag = 0
		for line in f:
			if line[0:4] == 'cpt:':
				if line[4] == 'T':
					line = line.strip()
					line = line.split('-')[:-1]
					years = [int(i[len(i)-4:len(i)]) for i in line]
				if flag == 2:
					vals = np.asarray(vals, dtype=float)
					vals[vals == -999.0] = np.nan
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
		vals[vals == -999.0] = np.nan
		all_vals.append(vals)
		all_vals = all_vals
		f.close()
		if self.model_ndx < len(self.models):
			name = self.models[self.model_ndx]
		else:
			name = "Observations"
		return (years, lats, longs, all_vals, name)

	def refresh(self, tgt_ndx, yr_ndx, big=False):
		if big:
			self.climatetensors[tgt_ndx][yr_ndx] = ClimateTensor()
			for i in range(len(self.models)):
				f = open("./input/{}_PRCP_{}_ini{}.tsv".format(self.models[i], self.tgts[tgt_ndx], self.mons[tgt_ndx]), 'r')
				data_tuple = self.read_data(f)

				self.climatetensors[tgt_ndx][yr_ndx].add_model(data_tuple, yr_ndx)

			f = open('./input/obs_PRCP_{}.tsv'.format(self.tgts[tgt_ndx]), 'r')
			data_tuple = self.read_data(f)
			self.climatetensors[tgt_ndx][yr_ndx].add_observations(data_tuple, yr_ndx) #assume that there is observation data for every year of each model
			self.climatetensors[tgt_ndx][yr_ndx].big = self.load(tgt_ndx, yr_ndx, big=True)
		else:
			pass #needs to be implemented i guess

	def read_all_data(self, big=False):
		"""Returns a 3d list of shape (nmodels, ntgts, years) ClimateTensor for each [model,tgt,year]"""
		climatetensors = [[] for j in range(len(self.tgts))]  #create climate tensors object

		for i in range(len(self.models)): #add a climate tensor for each [model,tgt,year]
			for j in range(len(self.tgts)):
				f = open("./input/{}_PRCP_{}_ini{}.tsv".format(self.models[i], self.tgts[j], self.mons[j]), 'r')
				data_tuple = self.read_data(f)

				for year in range(len(data_tuple[0])):
					if len(climatetensors[j]) < len(data_tuple[0]):
						climatetensors[j].append(ClimateTensor())
					climatetensors[j][year].add_model(data_tuple, year)

		for j in range(self.n_tgts):
			self.n_years.append(len(climatetensors[j]))
			f = open('./input/obs_PRCP_{}.tsv'.format(self.tgts[j]), 'r')
			data_tuple = self.read_data(f)
			#for i in range(len(self.models)):
			for k in range(len(climatetensors[j])):
					# data_tuple[0].index(climatetensors[i][j].t[k]) represents the index of the model year within the observation years
				climatetensors[j][k].add_observations(data_tuple, data_tuple[0].index(climatetensors[j][k].t[0])) #assume that there is observation data for every year of each model
		self.climatetensors = climatetensors


		for j in range(self.n_tgts):
			for k in range(len(climatetensors[j])):
				if big:
					climatetensors[j][k].big = self.load(j, k, big=True)

		for j in range(self.n_tgts):
			for k in range(len(climatetensors[j])):
				if big:
					climatetensors[j][k].add_nextgen_ensemble(self.pycpt.get_NGensemble(j), k)
		return climatetensors

	def fetch_iri_data(self):
		for model in range(len(self.models)):
			for tgt in range(len(self.tgts)):
				self.pycpt.setupParams(tgt)
				self.pycpt.prepFiles(tgt, model)

	def load(self, tgt, yr, kernel_size=3, big=False):
		self.train_data = np.asarray([])
		#for i in range(len(self.climatetensors)):
		#	for j in range(len(self.climatetensors[i])):
		self.climatetensors[tgt][yr].rectify()
		self.climatetensors[tgt][yr].crop()
		t_data = self.climatetensors[tgt][yr].model_data()
		if big:
			self.train_data = t_data
			return self.train_data
		for k in range(int(kernel_size/2),  t_data.shape[1]- int(kernel_size/2)):
			for l in range(int(kernel_size/2), t_data.shape[2] - int(kernel_size/2) ):
				input_section = t_data[:, k - int(kernel_size/2):k + int(kernel_size/2)+1,  l - int(kernel_size/2):l + int(kernel_size/2)+1 ]
				if self.train_data.shape[0] == 0:
					self.train_data = np.asarray([input_section])
				else:
					self.train_data = np.vstack((self.train_data, np.asarray([input_section])))
		return self.train_data

	def get(self, tgt, yr):
		return self.climatetensors[tgt][yr].big


class CNNensemble(nn.Module):
	def __init__(self, nmodels, dataloader, ntrain=1000, batch_size=0.5, seed=1693):
		super(CNNensemble,self).__init__()
		self.conv1 = nn.Conv2d(nmodels, 3, 3, stride=1, padding=1)
		self.dataloader = dataloader
		self.conv2 = nn.Conv2d(3, 1, 3, stride=1, padding=1)
		#self.linear = nn.Linear(int(ntrain), int(ntrain))
		self.train_losses = []
		self.batch_size = batch_size
		self.val_losses = []
		self.ntrain=ntrain
		self = self.float()
		np.random.seed(seed)
		self.datamean = -69

	def add_optimizer(self, optimizer):
		self.optimizer = optimizer

	def add_criterion(self, criterion):
		self.criterion = criterion

	def forward(self, x, pred=False):
		#print(x.shape, '- preconvolution')
		y_hat = self.conv1(x)
		#if not pred:
			#y_hat = F.relu(y_hat)
		y_hat = F.relu(y_hat)
		y_hat = self.conv2(y_hat)
		y_hat = F.sigmoid(y_hat)
		#print(y_hat.shape, ' - post convolution')
		#print(y_hat.flatten().shape, ' - flat')

		#y_hat= self.linear(y_hat.flatten())
		#print(y_hat.shape, ' - post-linear')
		return y_hat

	def batch_train(self, coords, epochs=1, big=False):

		x_data, y_data = [], []
		for coord in coords:
			tgt, yr, yr1 = coord
			self.dataloader.refresh(tgt, yr1, big=True)

			x_data.append(self.dataloader.get(tgt,yr))
			y_data.append(self.dataloader.get(tgt,yr1))

		if big:
			x_data, val_data = np.asarray(x_data), np.asarray(y_data)
		else:
			x_data, val_data = self.dataloader.load(tgt,yr), self.dataloader.load(tgt,yr1)

		#print('1', np.where(np.isnan(x_data)), np.where(np.isnan(val_data)))
		self.datamean, self.datastd = np.nanmean(x_data, axis=0), np.nanstd(x_data, axis=0)
		self.datamean[np.where(np.isnan(self.datamean))] = -1
		self.datastd[np.where(np.isnan(self.datastd))] = -1
		self.datastd[np.where(self.datastd==0)] = 1


		self.datamean3, self.datastd3 = np.nanmean(val_data, axis=0), np.nanstd(val_data, axis=0)
		self.datamean3[np.where(np.isnan(self.datamean3))] = -1
		self.datastd3[np.where(np.isnan(self.datastd3))] = -1
		self.datastd3[np.where(self.datastd3==0)] = 1

		#print('2a', np.where(np.isnan(self.datamean)), np.where(np.isnan(self.datamean3)))
		#print('2b', np.where(self.datastd==0), np.where(self.datastd3==0))

		#print('2', np.where(np.isnan(x_data)), np.where(np.isnan(val_data)))

		x_data[np.where(np.isnan(x_data))] = -1
		val_data[np.where(np.isnan(val_data))] = -1
		#print('3', np.where(np.isnan(x_data)), np.where(np.isnan(val_data)))

		#ndxs = np.asarray([i for i in range(x_data.shape[0])])
		#np.random.shuffle(ndxs)
		#x_data, val_data = x_data, val_data
		#if self.datamean == -69:
		x_data = (x_data - self.datamean) /  self.datastd

		val_data = (val_data - self.datamean3) / self.datastd3
		#print('3', np.where(np.isnan(x_data)), np.where(np.isnan(val_data)))

		#training_data = training_data[:int(self.batch_size*self.ntrain),:,:,:]
		x_np, val_np  = tr.from_numpy(x_data), tr.from_numpy(val_data)
		#print('4', np.where(np.isnan(x_np.data.numpy())), np.where(np.isnan(val_np.data.numpy())))

		self._batch_train(x_np, val_np, epochs=epochs)

	def _batch_train(self, x_np, val_np, epochs=1):

		x_train, y_train = Variable(x_np[:, :6, :, :]), Variable(x_np[:, 6:, :, :])
		x_val, y_val = Variable(val_np[:, :6, :, :]), Variable(val_np[:, 6:, :, :])
		#print('5', np.where(np.isnan(x_train.data.numpy())), np.where(np.isnan(x_val.data.numpy())))

		self.batch(x_train, y_train, x_val, y_val, epoch=epochs)

	def batch(self, x_train, y_train, x_val, y_val, epoch=0):
		self.train()
		self.tr_loss = 0
		self.optimizer.zero_grad()

		output_train = self(x_train.float())
		output_val = self(x_val.float())

		#print('6', np.where(np.isnan(output_train.data.numpy())), np.where(np.isnan(output_val.data.numpy())))

		#print(output_train.shape, output_val.shape)
		#print(output_train[0], np.nanmax(output_train.detach().numpy()[0]))
		loss_train = self.criterion(output_train, y_train.float())
		loss_val = self.criterion(output_val, y_val.float())
		self.train_losses.append(loss_train)
		self.val_losses.append(loss_val)
		loss_train.backward()
		self.optimizer.step()
		self.tr_loss = loss_train.item()
		print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

	def save(self, fn):
		f = open(fn, 'wb')
		pkl.dump(self,f)
		f.close()

	def visualize(self):
		plt.plot(self.train_losses, label='Training loss')
		plt.plot(self.val_losses, label='Validation loss')
		plt.legend()
		plt.show()

	def predict(self, coord, big=False, Test=True):
		if not big:
			#self.dataloader.read_all_data()
			tgt, yr, yr1 = coord
			x_data = self.dataloader.load(tgt,yr)
		else:
			if Test:
				tgt, yr, yr1 = coord[0]
				self.dataloader.refresh(tgt,yr,big=True)
				x_data = np.asarray([self.dataloader.get(tgt, yr)])
			else:
				x_data = np.asarray([self.dataloader.get(tgt, yr) for tgt, yr, yr1 in coord])
		#datastd, datamean = np.nanstd(x_data), np.nanmean(x_data)
		#self.dataloader.climatetensors[tgt][yr].plot_inputs(tgt,yr)
		#nans = [[np.where(np.isnan(x_data[j,i,:,:])) for i in range(x_data.shape[1])] for j in range(x_data.shape[0])]
		nans = np.where(np.isnan(x_data))
		self.datastd2, self.datamean2 = np.nanstd(x_data, axis=0),np.nanmean(x_data, axis=0)
		self.datastd2[np.where(self.datastd2==0)] = 1
		self.datastd2[np.where(np.isnan(self.datastd2))] = -1
		self.datamean2[np.where(self.datamean2==0)] = 1
		self.datamean2[np.where(np.isnan(self.datamean2))] = -1

		#for i in range(x_data.shape[1]):
		#	for j in range(x_data.shape[0]):
		x_data[nans] = -1
		nan_mask = np.zeros(x_data.shape)
		nan_mask[nans] = 1
		nan_mask = np.sum(nan_mask, axis=1)
		nan_mask[np.where(nan_mask == 0)] = 1
		nan_mask = nan_mask.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[3])
		#print('2a', np.where(np.isnan(self.datamean2)), np.where(np.isnan(self.datastd2)))
		#print('2b', np.where(self.datamean2==0), np.where(self.datastd2==0))

		#print('2', np.where(np.isnan(x_data)))#, np.where(np.isnan(val_data)))
		#ndxs = np.asarray([i for i in range(x_data.shape[0])])
		#np.random.shuffle(ndxs)
		#_data= x_data[ndxs,:,:,:]
		x_data = (x_data - self.datamean2 ) / self.datastd2
		#nans = np.where(np.isnan(x_data))
		#training_data = training_data[:int(self.batch_size*self.ntrain),:,:,:]
		x_np  = tr.from_numpy(x_data)
		#print('3', np.where(np.isnan(x_np)))
		x_train = Variable(x_np[:, :6, :, :])
		#print('3', np.where(np.isnan(x_train)))

		preds = self(x_train.float(), pred=True).data.numpy()
		#print('4', np.where(np.isnan(preds)))

		if not big:
			preds = (preds.flatten().reshape(132,106) * self.datastd2) + self.datamean2
		else:
			preds = preds.reshape(preds.shape[0], preds.shape[2], preds.shape[3])
			preds = (preds * self.datastd2[-1:,:,:]) + self.datamean2[-1:,:,:]

		if not big:
			preds[nans[0],0,0,0] = np.nan
		else:
			#print(preds.shape, len(nans[0]), len(nans[1]), len(nans[2]), len(nans[3]))
			#for i in range(preds.shape[0]):
			#	for j in range(len(nans[i])):
			#		preds[i,nans[i][j]] = np.nan
			preds = preds * nan_mask
			preds[np.where(preds==0)] = np.nan
			preds[np.where(preds<0)] = np.nan
		#print('4', np.where(np.isnan(preds)))


		return preds #* datamax


	@classmethod
	def load(self, fn):
		f = open(fn, 'rb')
		return pkl.load(f)
