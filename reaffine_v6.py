"""
Combination of ERICHA Flash Flood hazard  
with EU Floods Directive maps, population density, and critical infrastructures (CI) to compute flash flood impact across Europe

1.	Simulate potentially flooded area as hazard level (low-medium-high) in each 1 km floodplain cell 
		- Input: ERICHA flash flood hazard level in the drainage network
		- Based on the most severe flood map available (e.g. in Spain: T=500 years), representing the floodplain
2.	Simulate number of people in potentially flooded area for each (1 km) floodplain cell 
		- Based on a population density grid, cropped in high resolution to the floodplain extent (preprocessed offline)
3. 	Simulate flash flood impact on population (low-medium-high) at 1 km resolution
		- Reclassify population density values in the potentially flooded area into qualitative classes (low-medium-high), based on predefined thresholds
		- Compute flash flood impact by crossing in an impact matrix: The reclassified population density in the floodplain cells with the hazard level in the floodplain cells
4.	Simulate critical infrastructures (CIs) in the flooded area
		- Based on CI point layers, cropped in high resolution to the floodplain extent (preprocessed offline)
5. 	Summarise flash flood impacts at NUTS region level, based on following steps:
		- Take for each NUT the highest floodplain hazard level affecting population to determine the overall impact level of the NUT region
		- Sum in each NUT the amounts of population potentially affected by low (pop1), medium (pop2), and high hazard (pop3), plus the CIs affected by the three hazard levels

Created: 30/06/2020
Author: Josias Ritter

"""
import pdb # pdb.set_trace() (REMOVE AFTER TESTING)

import sys
import os
import glob
import getopt
import configparser
import logging

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
import shapefile
import gdal
import rasterio
from pyproj import proj, transform

#***************************************************************
# THRESHOLD DEFINITIONS
#***************************************************************
# population_thresholds = {'population':[0,1,1000,5000,100000]} # [1/km2] Original thresholds for spatial hazards. Could be used for areas without flood maps
population_thresholds = {'population':[0,1,100,1000,100000]} # [1/km2] Thresholds for areas with flood maps (smaller, since cropping the population density map to the floodplain reduces 1 km cell values significantly!)
pop_keys = {'none':0,'low':1,'moderate':2,'high':3}

impact_thresholds = {'impact':[0,1,3,6,10]} # [result from multiplication in impact matrix]
impact_keys = {'none':0,'low':1,'moderate':2,'high':3}

#***************************************************************

def main(config):
	logging.info('********** START flash flood impact module *********')
	# files = glob.glob(os.path.join(config['PATHS']['hazard_dir'], '*.nc'))

	# Find FF relevant hazard files in directory and subdirectories
	ff_files = np.array([])
	for root, dirs, files in os.walk(config['PATHS']['hazard_dir']):
		for file in files:
			if file.endswith(".nc"):
				ff_files = np.append(ff_files, os.path.join(root,file))

	valid_ind = np.where((np.asarray([i[-22:-12] for i in ff_files]) == 'hazard_max') | (np.asarray([i[-22:-16] for i in ff_files]) == 'haz24h') | (np.asarray([i[-28:-16] for i in ff_files]) == np.asarray([i[-15:-3] for i in ff_files])))
	if np.size(valid_ind) > 0:
		ff_files_valid = ff_files[valid_ind]
	else: logging.warning('No flash flood hazard files found')

	for path_f in sorted(ff_files_valid):
		
		# Define output directory and filenaming
		out_basepath = config['PATHS']['out_dir']
		if path_f[-22:-12] == 'hazard_max':
			out_path = out_basepath
			out_tag = path_f[-15:-3]
		elif path_f[-22:-16] == 'haz24h':
			out_path = out_basepath+path_f[-36:-22]
			out_tag = path_f[-19:-3]
		else: 
			out_path = out_basepath+path_f[-52:-38] 
			out_tag = path_f[-28:-16]

		# # testing
		# floodplain_hazard0 = gdal.Open(os.path.join(out_path,'floodplainhaz_'+out_tag+'.tiff'))
		# floodplain_hazard = floodplain_hazard0.ReadAsArray()
		# impact, poprast = pop_impact(config, floodplain_hazard, population_thresholds, impact_thresholds, out_path, out_tag) # Population impact module
		# ci_aff_df = crit_inf(config, floodplain_hazard, poprast, out_path, out_tag) # Critical infrastructure module
		# impactsum(config, floodplain_hazard, impact, poprast, ci_aff_df, out_path, out_tag) # Impact summary at NUTS region level
		# pdb.set_trace()

		poprast = gdal.Open(os.path.join(config['PATHS']['population_path'])) # Read population density raster

		hazard = readhazard(config, path_f, poprast, out_path, out_tag) # Read ERICHA FF hazard

		if np.max(hazard) > 0:
			floodplain_hazard = inundate(config, hazard, out_path, out_tag) # Flood map module
			
			if np.max(floodplain_hazard) > 0:
				# pdb.set_trace()
				impact = pop_impact(config, floodplain_hazard, poprast, population_thresholds, impact_thresholds, out_path, out_tag) # Population impact module
				ci_aff_df = crit_inf(config, floodplain_hazard, poprast, out_path, out_tag) # Critical infrastructure module

				if not (np.max(impact) == 0 and ci_aff_df.empty): 
					impactsum(config, floodplain_hazard, poprast, impact, ci_aff_df, out_path, out_tag) # Impact summary at NUTS region level

	logging.info('********** END flash flood impact module *********')


#***************************************************************
# IMPACT
#***************************************************************

def readhazard(config, hazard_path, poprast, out_path, out_tag):
	# Read and crop FF hazard to project grid (of population density and CI layers)
	# hazard_reproj = gdal.Warp('', hazard_path, srcSRS= 'EPSG:3035', dstSRS='EPSG:3035', format='VRT', outputType=gdal.GDT_Float32, outputBounds=[2630000.0, 1380000.0, 5960000.0, 5430000.0], xRes=1000, yRes=1000, dstNodata = 0) # crop hazard to popdens extent
	# hazard_reproj_np = hazard_reproj.ReadAsArray()

	# For Markus (GDAL 3.1.4):
	hazard_raw = gdal.Open(hazard_path)
	inbnd = [2500000.0, 750000.0, 7500000.0, 5430000.0]
	outbnd = [2630000.0, 1380000.0, 5960000.0, 5430000.0]
	dx = 1000
	hazard_reproj_np_raw = hazard_raw.ReadAsArray()
	x0=int((outbnd[0]-inbnd[0])/dx)
	x1=int((outbnd[2]-inbnd[0])/dx)
	y0=int((inbnd[3]-outbnd[3])/dx) # np y-axis defined as top->bottom
	y1=int((inbnd[3]-outbnd[1])/dx)
	hazard_reproj_np = hazard_reproj_np_raw[y0:y1, x0:x1] # Crop to output bounds of population density!

	if config['SETTINGS']['mask'] == 'True':
		# Set hazard values outside mask to nodata (i.e. outside Germany and Spain)
		mask = gdal.Open(os.path.join(config['PATHS']['mask_path']))
		mask_np = mask.ReadAsArray()
		hazard_nodataval = 9999
		hazard_reproj_np[np.where(mask_np < 0)] = hazard_nodataval
		hazard_reproj_np = hazard_reproj_np.astype('uint16')
		haz_ind = np.where((hazard_reproj_np > 0) & (hazard_reproj_np <= 3))
		
		# Save clipped FF hazard
		if len(haz_ind[0]) > 0:
			driver = gdal.GetDriverByName('GTiff')
			options= ['COMPRESS=DEFLATE']
			nx = poprast.RasterXSize
			ny = poprast.RasterYSize
			dataset = driver.Create(os.path.join(out_path,'FFhaz_clip_'+out_tag+'.tiff'), nx, ny, 1, gdal.GDT_UInt16, options)
			dataset.GetRasterBand(1).WriteArray(hazard_reproj_np)
			dataset.GetRasterBand(1).SetNoDataValue(hazard_nodataval)
			geotrans=poprast.GetGeoTransform()  
			# proj=poprast.GetProjection()  
			dataset.SetGeoTransform(geotrans)
			# dataset.SetProjection(proj)
			dataset.FlushCache()
			dataset=None

	# Prepare for combination with flood maps
	haz0_ind = np.where(hazard_reproj_np > 3)
	hazard_reproj_np[haz0_ind] = 0
	hazard = hazard_reproj_np.astype('uint8')
	print(out_tag+' - Max. hazard level in drainage network: '+str(np.max(hazard)))

	return hazard


def inundate(config, hazard, out_path, out_tag):
	# Spread FF hazard from drainage network cells into the floodplains (based on .sav file containing the assignment matrix between stream cells and flood cells)
	from os.path import dirname, join as pjoin
	import scipy.io as sio
	from scipy.io import readsav

	# Read table containing the connections (indices) between flood cells and stream cells
	fc2ad8_path = os.path.join(config['PATHS']['fc2ad8_path'])
	sav_data = readsav(fc2ad8_path)
	fc2ad8 = sav_data['fc2ad8']
	dim = np.shape(fc2ad8)
	
	# Find (IDL-like) 1D indices of hazard cells
	hazard = np.flipud(hazard) # y-axis of hazard needs to be inverted for indices to match orientation in fc2ad8 (numpy y-axis direction defined as top->bottom, IDL opposite!)
	dim_haz = np.shape(hazard)
	haz_ind2D = np.where(hazard > 0)
	haz_ind2D_np = np.asarray(haz_ind2D)
	haz_ind = haz_ind2D_np[1] + dim_haz[1] * haz_ind2D_np[0] # Convert 2D indices to 1D indices (IDL-like, as provided in fc2ad8 table)
	n_hazcells = np.size(haz_ind)

	# Read floodcell grid (containing % values)
	floodmap = gdal.Open(os.path.join(config['PATHS']['floodmap_path']))
	floodmap_np = floodmap.ReadAsArray()
	floodmap_np = np.flipud(floodmap_np) # y-axis needs to be inverted for indices to match orientation in fc2ad8 (numpy y-axis defined as top->bottom, IDL opposite!)
	dimQ = np.shape(floodmap_np)
	fc_ind = np.asarray(np.where(floodmap_np > 0))

	# Create output floodplain hazard raster. Fill floodplain cells with value 0 and cells outside floodplain with nodata
	nodataval_floodplain = 255
	floodplain_hazard = np.full(dimQ, nodataval_floodplain, dtype='uint8')
	floodplain_hazard[np.where(floodmap_np > 0)] = 0 

	# # testing
	# import matplotlib.pyplot as plt # testing
	# img = floodmap_np[0:1000,0:2000]
	# plt.imshow(img)
	# plt.show()
	# pdb.set_trace()

	# Fill for each hazard cell all the linked flood cells with the given hazard level
	for i in range(n_hazcells):
		if (i%1000) == 0: #if ((i/n_hazcells)*100).is_integer(): 
			print(out_tag+' - Computation progress: FF hazard cell '+str(i)+'/'+str(n_hazcells)) 
		for ii in range(dim[0]):	# loop over secondary dimension of fc2ad8 matrix, because: one 1km-flood cell can be equally close (and thus potentially be flooded) from up to 4 1km-stream cells
			flood_pot = np.where(fc2ad8[ii] == haz_ind[i])
			if np.size(flood_pot) > 0:
 				flood_pot_list = [np.asscalar(k) for k in flood_pot[0]]
 				flood_ind = fc_ind[:, flood_pot_list]

 				fill_ind = np.where(floodplain_hazard[tuple(flood_ind)] < hazard[tuple(haz_ind2D_np[:,i])]) # fill floodplain cells only if hazard level is higher than (possible) exisiting hazard level
 				fill_ind_list = [np.asscalar(k) for k in fill_ind[0]]
 				if np.size(fill_ind_list) > 0:
 					floodplain_hazard[tuple(flood_ind[:, fill_ind_list])] = hazard[tuple(haz_ind2D_np[:,i])]
	
	floodplain_hazard = np.flipud(floodplain_hazard) # flip output y-axis back to numpy orientation
	floodplain_hazard_ind = np.where((floodplain_hazard > 0) & (floodplain_hazard < nodataval_floodplain))
	print(out_tag+' - Max. hazard level in floodplain: '+str(np.max(floodplain_hazard[floodplain_hazard_ind])))

	if config['SETTINGS']['mask'] == 'True':
		# Set hazard values outside mask to nodata (i.e. outside Germany and Spain)
		mask = gdal.Open(os.path.join(config['PATHS']['mask_path']))
		mask_np = mask.ReadAsArray()
		floodplain_hazard[np.where(mask_np < 0)] = nodataval_floodplain

	# Write output raster of floodplain_hazard (if any cell with hazard > 0)
	if len(floodplain_hazard_ind[0]) > 0:
		os.makedirs(out_path, exist_ok=True)
		driver = gdal.GetDriverByName('GTiff')
		options= ['COMPRESS=DEFLATE']
		dataset = driver.Create(os.path.join(out_path,'floodplainhaz_'+out_tag+'.tiff'), dim_haz[1], dim_haz[0], 1, gdal.GDT_Byte, options)
		dataset.GetRasterBand(1).WriteArray(floodplain_hazard)
		dataset.GetRasterBand(1).SetNoDataValue(nodataval_floodplain)
		geotrans=floodmap.GetGeoTransform()
		# proj=floodmap.GetProjection()
		dataset.SetGeoTransform(geotrans)
		# dataset.SetProjection(proj)
		dataset.FlushCache()
		dataset=None

	# (option to add: read percentages of area covered by floodplain in affected flood cells and write output raster of area percentages)
	return floodplain_hazard


def pop_impact(config, floodplain_hazard, poprast, population_thresholds, impact_thresholds, out_path, out_tag):

	poprast_np = poprast.ReadAsArray()
	nx = poprast.RasterXSize
	ny = poprast.RasterYSize

	# Classify population density grid into exposure levels none-low-moderate-high (in raster values: 0-1-2-3) based on predefined thresholds 
	exposure = quantity2quality('population',poprast_np,population_thresholds)

	# Impact matrix for calculation of population impact at pixel level [1 km]
	impact_raw = np.multiply(floodplain_hazard,exposure)
	impact = quantity2quality('impact',impact_raw,impact_thresholds) # reclassify impact values to none-low-moderate-high (values 0-1-2-3) 
	print(out_tag+' - Max. impact on population: '+str(np.max(impact)))

	# Write output raster of flash flood impact (if any cell with impact > 0)
	if np.max(impact) > 0:
		driver = gdal.GetDriverByName('GTiff')
		options= ['COMPRESS=DEFLATE']
		dataset = driver.Create(os.path.join(out_path,'impact_'+out_tag+'.tiff'), nx, ny, 1, gdal.GDT_Byte, options)
		dataset.GetRasterBand(1).WriteArray(impact)
		dataset.GetRasterBand(1).SetNoDataValue(0)
		geotrans=poprast.GetGeoTransform()  
		# proj=poprast.GetProjection()  
		dataset.SetGeoTransform(geotrans)
		# dataset.SetProjection(proj)
		dataset.FlushCache()
		dataset=None

	return impact


def crit_inf(config, floodplain_hazard, poprast, out_path, out_tag):
	# Read point shp file of 1km cell centroids with CIs
	ci_path = os.path.join(config['PATHS']['ci_path'])
	ci_sf = shapefile.Reader(ci_path)
	ci_records = ci_sf.records()
	ci_shapes = ci_sf.shapes()

	# pdb.set_trace()

	# Coordinates of CIs in CRS (EPSG:3035)
	# ci_x = [ci_shapes[i].bbox[0] for i in range(len(ci_records))]
	# ci_y = [ci_shapes[i].bbox[1] for i in range(len(ci_records))]
	ci_x = [ci_shapes[i].points[0][0] for i in range(len(ci_records))]
	ci_y = [ci_shapes[i].points[0][1] for i in range(len(ci_records))]

	# Input raster properties 
	geotrans = poprast.GetGeoTransform()
	dx = geotrans[1]
	x0 = geotrans[0]
	# y0 = geotrans[3] - (poprast.RasterYSize * dx)
	y0 = geotrans[3]

	# CI cell indices with respect to input raster
	ci_indx = np.fix((np.asarray(ci_x) - x0) / dx)
	ci_indx = ci_indx.astype(int)
	ci_indy = np.fix(np.absolute((np.asarray(ci_y) - y0) / dx)) # numpy y-axis direction is north->south
	ci_indy = ci_indy.astype(int)
	ci_indxy = tuple([ci_indy, ci_indx])

	# Filter for CIs in cells with hazard > 0
	haz_lvl_ci = floodplain_hazard[ci_indxy] # hazard level in CI cells
	ci_aff_ind = np.where(haz_lvl_ci > 0)
	if np.size(ci_aff_ind) > 0:
		haz_lvl_ci_aff = haz_lvl_ci[ci_aff_ind]

		# Write shapefile of potentially affected CIs
		def create_critinf_shp(path_shp):
			w = shapefile.Writer(path_shp, shapefile.POINT)
			# w = shapefile.Writer(path_shp, shapefile.MULTIPOINT)
			w.field('regionID', 'N', 5, 0)
			w.field('region', 'C', 70)
			w.field('iso3', 'C', 3)
			w.field('FFhaz_lvl', 'N', 2, 0)
			w.field('EF', 'N', 3, 0)		# education facilities
			w.field('HF', 'N', 3, 0)		# health facilities
			w.field('MG', 'N', 3, 0)		# mass gathering sites
			# w.field('RD_km', 'N', 6, 0)		# road kilometres

			for i in range(len(ci_aff_ind[0])):
				shp_ind = ci_aff_ind[0][i]
				w.shape(ci_shapes[shp_ind])
				# rec = ci_sf.shapeRecords()[shp_ind] #alternative
				# w.shape(rec.shape) #alternative
				w.record(ci_records[shp_ind][0], ci_records[shp_ind][1], ci_records[shp_ind][2], haz_lvl_ci_aff[i], ci_records[shp_ind][3], ci_records[shp_ind][4], ci_records[shp_ind][5]) #, ci_records[shp_ind][6])
			w.close()
			# create the .prj file with EPSG 3035
			prj = open(path_shp+'.prj', 'w')
			prj.write('PROJCS["ETRS89_LAEA_Europe",GEOGCS["GCS_ETRS_1989",DATUM["D_ETRS_1989",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_origin",52],PARAMETER["central_meridian",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["Meter",1]]')
			prj.close()

		path_outshp = os.path.join(out_path,'critinf_'+out_tag)
		create_critinf_shp(path_outshp)

		# Convert attribute table of affected CIs into a Dataframe as input to impactsum
		ci_records_aff = [ci_records[i] for i in ci_aff_ind[0]]
		fields = [x[0] for x in ci_sf.fields][1:]
		ci_aff_df = pd.DataFrame(columns=fields, data=ci_records_aff)
		ci_aff_df.insert(3, 'FFhaz_lvl', haz_lvl_ci_aff, True) # insert FF hazard values as fourth column
	else: ci_aff_df = pd.DataFrame() # empty dataframe

		# pdb.set_trace()
	return ci_aff_df


def impactsum(config, floodplain_hazard, poprast, impact, ci_aff_df, out_path, out_tag):

	# Open NUTS shapefile as template for output impactsum shapefile
	nuts_path = os.path.join(config['PATHS']['nuts_path'])
	nuts_sf = shapefile.Reader(nuts_path)
	nuts_records = nuts_sf.records()
	nuts_names = [nuts_records[i][6] for i in range(len(nuts_records))]
	nuts_names_np = np.asarray(nuts_names)

	# Sum for each NUT: population amounts affected by hazard levels 1, 2, and 3
	with rasterio.open(os.path.join(out_path,'impact_'+out_tag+'.tiff')) as src:
		affine = src.transform
	poprast_np = poprast.ReadAsArray()
	pop1_list, pop2_list, pop3_list = sum4nuts(floodplain_hazard, poprast_np, nuts_path, affine)

	# Maximum population impact level in each NUT
	stats4 = zonal_stats(nuts_path, impact, affine=affine, stats=['max'], all_touched=True) 
	popimp_max = [f['max'] for f in stats4]
	
	# Sum for each NUT: CIs affected by hazard levels 1, 2, and 3 # ADD OTHER CI TYPES!!!
	ci_aff_haz1 = np.empty(len(pop1_list), dtype='<U170')
	ci_aff_haz2 = np.empty(len(pop1_list), dtype='<U170')
	ci_aff_haz3 = np.empty(len(pop1_list), dtype='<U170')
	if not ci_aff_df.empty:
		ci_aff_nuts_df = ci_aff_df.groupby(['Region','FFhaz_lvl'], as_index = False).agg({'EF': 'sum', 'HF': 'sum', 'MG': 'sum'}) #, 'RD_m': 'sum'})
		ci_aff_nuts_df = ci_aff_nuts_df.round({'EF':0, 'HF':0, 'MG':0}) #, 'RD_km':0})

		# Create for each NUT and each hazard level a string that lists the numbers and types of flooded CIs
		ci_aff_str = np.empty(len(ci_aff_nuts_df.index), dtype='<U170')
		ci_type_list = list(ci_aff_nuts_df.columns[2:])
		for ci_type in ci_type_list: # loop over ci_types and consecutively fill a string array listing the affected CIs for each region and hazard level
			ci_vals = ci_aff_nuts_df[ci_type]
			ci_vals_np = ci_vals.to_numpy(int)
			gt0_ind = np.where(ci_vals_np > 0)
			addstr = np.core.defchararray.add(np.core.defchararray.add(ci_vals_np[gt0_ind].astype(str), ' '), (ci_type+', ')) # strings to be added
			ci_aff_str[gt0_ind] = np.core.defchararray.add(ci_aff_str[gt0_ind], addstr) # add the strings to existing string entries
		ci_aff_nuts_df.insert(len(ci_aff_nuts_df.columns), 'CI_aff', ci_aff_str, True) # add string array summarising the affected CIs to the end of the dataframe

		# Create 3 string arrays (i.e. one per hazard level) listing for all regions (not only the affected regions) the CIs affected
		for i in range(len(ci_aff_nuts_df.index)):
			dfline = ci_aff_nuts_df.loc[i]
			nut = dfline['Region']
			nut_ind = np.where(nuts_names_np == nut)
			if dfline['FFhaz_lvl'] == 1: ci_aff_haz1[nut_ind] = dfline['CI_aff']
			if dfline['FFhaz_lvl'] == 2: ci_aff_haz2[nut_ind] = dfline['CI_aff']
			if dfline['FFhaz_lvl'] == 3: ci_aff_haz3[nut_ind] = dfline['CI_aff']

	# Impact level for each NUT, based on amounts of population or CIs affected by different hazard levels (If any population or CI affected by hazard level 1, nut impact = 1, ...)
	impact_lvl_nuts = np.zeros(len(pop1_list))
	impact_lvl_nuts[(np.where((pop1_list > 0) | (ci_aff_haz1 != '')))] = 1
	impact_lvl_nuts[(np.where((pop2_list > 0) | (ci_aff_haz2 != '')))] = 2
	impact_lvl_nuts[(np.where((pop3_list > 0) | (ci_aff_haz3 != '')))] = 3
	
	# # Remove NUTS without impact and sort in descending order using: pop3_list, then pop2_list, then pop1_list
	# sort_ind = np.array([])
	# lvl3 = np.asarray(np.where(impact_lvl_nuts == 3))
	# sort_lvl3 = np.flip(np.argsort(pop3_list[lvl3]))
	# sort_ind = np.append(sort_ind, lvl3[0][sort_lvl3])
	# lvl2 = np.asarray(np.where(impact_lvl_nuts == 2))
	# sort_lvl2 = np.flip(np.argsort(pop2_list[lvl2]))
	# sort_ind = np.append(sort_ind, lvl2[0][sort_lvl2])
	# lvl1 = np.asarray(np.where(impact_lvl_nuts == 1))
	# sort_lvl1 = np.flip(np.argsort(pop1_list[lvl1]))
	# sort_ind = np.append(sort_ind, lvl1[0][sort_lvl1])
	# sort_ind_int = sort_ind.astype(int)

	# Remove NUTS without impact and sort in descending order using: the population affected by all three hazard levels
	poplist_sum = pop1_list + pop2_list + pop3_list
	sort_ind0 = np.flip(np.argsort(poplist_sum))
	imp_gt0_ind = np.asarray(np.where(poplist_sum[sort_ind0] > 0))
	sort_ind = sort_ind0[imp_gt0_ind]
	sort_ind_int = sort_ind[0].astype(int)

	def create_shp(path_shp):
		w = shapefile.Writer(path_shp, shapefile.POLYGON)
		w.field('objectID', 'N', 5, 0)
		w.field('region', 'C', 70)
		w.field('country', 'C', 3)
		w.field('popimp_max', 'N', 1, 0)
		w.field('pop_haz', 'N', 7, 0)
		w.field('pop_haz3', 'N', 7, 0)
		w.field('pop_haz2', 'N', 7, 0)
		w.field('pop_haz1', 'N', 7, 0)
		w.field('ci_haz3', 'C', 170, 0)
		w.field('ci_haz2', 'C', 170, 0)
		w.field('ci_haz1', 'C', 170, 0)
		w.field('impact_lvl', 'N', 1, 0)
		w.field('impactrank', 'N', 6, 0)
		
		rank = 0
		for i in sort_ind_int:
			# print(i)
			rec = nuts_sf.shapeRecords()[i]
			shprec = rec.shape
			w.shape(shprec)
			w.record(nuts_records[i][0], nuts_records[i][6], nuts_records[i][1], popimp_max[i], poplist_sum[i], pop3_list[i], pop2_list[i], pop1_list[i], ci_aff_haz3[i], ci_aff_haz2[i], ci_aff_haz1[i], impact_lvl_nuts[i], rank)
			rank = rank+1
		w.close()

		# create the .prj file with EPSG 3035
		prj = open(path_shp+'.prj', 'w')
		prj.write('PROJCS["ETRS89_LAEA_Europe",GEOGCS["GCS_ETRS_1989",DATUM["D_ETRS_1989",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_origin",52],PARAMETER["central_meridian",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["Meter",1]]')
		prj.close()
	path_outshp = os.path.join(out_path,'impactsum_'+out_tag)
	create_shp(path_outshp)

#***************************************************************

# Conversion from quantitative values to qualitative levels, based on predefined keys and threshold values
def quantity2quality(entity,values,thresholds):
	quality = np.zeros(values.shape,dtype=np.ubyte)
	thresholds1 = thresholds[entity]
	n_thresholds = len(thresholds1)
	for i in range(n_thresholds):
		ind = np.where(values >= thresholds1[i])
		quality[ind] = i
	ind_outlier = np.where(values > thresholds1[-1])
	quality[ind_outlier] = 0
	return quality

# Summing the pixel values for each NUTS region, separately for hazard levels 1, 2, and 3
def sum4nuts(hazard, vuln_ras_np, nuts_path, affine):
	haz1_ind = np.where(hazard == 1)
	haz2_ind = np.where(hazard == 2)
	haz3_ind = np.where(hazard == 3)
	vuln_haz1, vuln_haz2, vuln_haz3 = np.zeros(np.shape(vuln_ras_np)), np.zeros(np.shape(vuln_ras_np)), np.zeros(np.shape(vuln_ras_np))
	
	vuln_haz1[haz1_ind] = vuln_ras_np[haz1_ind]
	stats1 = zonal_stats(nuts_path, vuln_haz1, affine=affine, stats=['sum'], all_touched=True) 
	vuln_list1 = [f['sum'] for f in stats1]
	vuln_list1 = np.asarray(vuln_list1)
	smallnut_ind = np.where(np.asarray(vuln_list1) == None)
	vuln_list1[smallnut_ind] = 0.
	vuln_list1_int = [np.ceil(i) for i in vuln_list1]
	vuln_list1_int = np.asarray(vuln_list1_int)
	vuln_list1_int = vuln_list1_int.astype('uint32')
	
	vuln_haz2[haz2_ind] = vuln_ras_np[haz2_ind]
	stats2 = zonal_stats(nuts_path, vuln_haz2, affine=affine, stats=['sum'], all_touched=True)  
	vuln_list2 = [f['sum'] for f in stats2]
	vuln_list2 = np.asarray(vuln_list2)
	vuln_list2[smallnut_ind] = 0.
	vuln_list2_int = [np.ceil(i) for i in vuln_list2]
	vuln_list2_int = np.asarray(vuln_list2_int)
	vuln_list2_int = vuln_list2_int.astype('uint32')

	vuln_haz3[haz3_ind] = vuln_ras_np[haz3_ind]
	stats3 = zonal_stats(nuts_path, vuln_haz3, affine=affine, stats=['sum'], all_touched=True) 
	vuln_list3 = [f['sum'] for f in stats3]
	vuln_list3 = np.asarray(vuln_list3)
	vuln_list3[smallnut_ind] = 0.
	vuln_list3_int = [np.ceil(i) for i in vuln_list3]
	vuln_list3_int = np.asarray(vuln_list3_int)
	vuln_list3_int = vuln_list3_int.astype('uint32')

	return vuln_list1_int, vuln_list2_int, vuln_list3_int

#####
if __name__=='__main__':

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Reading arguments
    loglevel = 'info'
    fileconf = 'config.properties'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "l:c:", ["loglev=","config="])
    except getopt.GetoptError:
        print('ERROR: Syntax should be:')
        print('python reaffine_v6.py -l <loglevel> -c <configfile>')
        print('loglevels: debug, info, warning, error, critical')
        sys.exit(2)

    if len(opts) == 0:
        print('Using loglevel: ' + loglevel + ' and config: ' + fileconf)
    for opt, arg in opts:
        if opt in ("-l", "--loglev"):
            loglevel = arg
        elif opt in ("-c", "--config"):
            fileconf = arg

    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    # Reading configuration
    if not os.path.exists(fileconf):
        print(fileconf + ' not found. Exit.')
        sys.exit(2)
    config = configparser.ConfigParser()
    config.read_file(open(fileconf))

    # Defining log
    logging.basicConfig(filename=config['FILES']['logFile'], level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d.%m.%Y %H:%M:%S')  

    #############################

    main(config)
