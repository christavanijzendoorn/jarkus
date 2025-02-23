# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:30:42 2013

$Id: transects.py 11076 2014-08-28 17:52:41Z heijer $
$Date: 2014-08-28 19:52:41 +0200 (Thu, 28 Aug 2014) $
$Author: heijer $
$Revision: 11076 $
$HeadURL: https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/jarkus_transects/jarkus/transects.py $

@author: heijer
"""

import logging
logger = logging.getLogger(__name__)

from netCDF4 import Dataset,num2date
import numpy as np
from numpy import asarray
from scipy.interpolate import interp1d


from threading import Lock
# shapely is not thread safe, so implement a lock
shapelock = Lock()

class Transects:
    """
    Wrapper for JARKUS transects
    """
	# CHRISTA: This wrapper was build by people from Deltares. I feel like this could be done in a more clear and accessible way. 
	# E.g. not working coding with 'self' so it is easier for people with less coding experience to get into or at least understand what is going on under the hood.
	# I haven't got around to rebuilding it myself though...
    def __init__(self, *args, **kwargs):
        """
        instantiate the environment
        """
        if 'url' in kwargs:
            self.url = kwargs.pop('url')
        else:
            self.url = 'https://opendap.tudelft.nl/thredds/catalog/data2/deltares/rijkswaterstaat/jarkus/profiles/catalog.html?dataset=scanDatasets2/deltares/rijkswaterstaat/jarkus/profiles/transect.nc'
        try:
            self.ds = Dataset("C://Users//cijzendoornvan//Documents//DuneForce//JARKUS//Transect_Data//transect.nc")
            # CHRISTA: line above is a hack I used to refer to local file. By downloading the Jarkus data and referring to the file here, the scripts become independent of internet access. This could be build in more robustly
            # self.ds = Dataset(self.url)
        except OSError as e:
            err = ('%e. "%s" not found.' %(e,self.url))
            logger.error(err)
            raise err

        self.dims = self.ds.dimensions

        # initiate filter
        self.filter = dict.fromkeys(self.dims.keys())
        self.reset_filter()
        self.set_filter(**kwargs)
            
    def reset_filter(self, *args):
        """
        remove filter for all dimensions (default) or for the specified dimensions only
        """
        if args == ():
            args = self.dims.keys()
        for k in args:
            self.filter[k] = np.ones((self.dims[k].__len__(),)) == 1

    def set_filter(self, **kwargs):
        """
        set filter by one or more keyword arguments
        filters can be specified as boolean (shape must match the dimension's shape), as indices or as variable specification.
        """
        for k,v in kwargs.items():
            isdim = k in self.ds.dimensions.keys()
            isvar = k in self.ds.variables.keys()
            if (isinstance(v, bool) or isinstance(v, np.ndarray) and v.dtype == bool) and len(v) == len(self.dims[k]):
                self.filter[k] = np.logical_and(self.filter[k], v)
            elif isinstance(v, (int, np.integer)) and k in self.dims and np.all(np.abs(np.asarray(v)) < self.dims[k].__len__()):
                self.filter[k] = np.ones((self.dims[k].__len__(),)) == 0
                self.filter[k][v] = True
            elif k == 'year':
                self.filter['time'] = self.year2idx(v)
            elif isvar and not isdim:
                dimname = self.ds.variables[k].dimensions[0]
                self.filter[dimname] = np.logical_and(self.filter[dimname], np.in1d(self.ds.variables[k][:], np.asarray(v)))
    def get_filter(self, key):
        """
        returns filter for specified key
        """
        return self.filter[key]
    def __exit__(self):
        """
        close NetCDF file
        """
        self.close()
    def close(self):
        """
        close NetCDF file
        """
        self.ds.close()        
    def get_data(self, varname):
        """
        returns data for specified variable and applies available filters
        """
        return self.ds.variables[varname][[self.filter[k] for k in self.ds.variables[varname].dimensions]]
    def areaname2areacode(self, areaname):
        """
        returns areaname for a specified areacode as input.
        \nToDo: include in another class of the same package "jarkus_transects", eventually.
        """
        # areas according to RWS definition
        areas = {"Schiermonnikoog":2,"Ameland":3,"Terschelling":4,"Vlieland":5,
		         "Texel":6,"Noord-Holland":7,"Rijnland":8,"Delfland":9,
				 "Maasvlakte":10,"Voorne":11,"Goeree":12,"Schouwen":13,
				 "Noord-Beveland":15,"Walcheren":16,"Zeeuws-Vlaanderen":17}
        if type(areaname) == np.str:
            return areas.get(areaname)
        if type(areaname) == list:
            return list(map(areas.get, areaname))
    def time2year(self, t):
        """
        convert time to year
        """
        time = self.ds.variables['time']
        if type(t) == np.int:
            return num2date(t, time.units).year
        else:
            return np.asarray([y.year for y in np.asarray(num2date(t, time.units))])
    def year2idx(self, year):
        """
        returns boolean index array to be applied to the time dimension
        """
        #time = self.ds.variables['time']
        #years = [y.year for y in num2date(time, time.units)]
        years = self.time2year(self.ds.variables['time'][:])
        if not year:
            year = years
        idx = np.in1d(years, np.asarray(year))
        return idx
    def cross_shore2xyRD(self, cs, transect_id, axis=None):
        """
        returns RD coordinates (epsg 28992) for cross-shore coordinate(s) (wrt to RSP)
        """
        cs = np.asarray(cs)
        transect_id = np.asarray(transect_id)
        aidx = np.in1d(self.ds.variables['id'], transect_id)
        cs_f = np.array((self.ds.variables['cross_shore'][0], self.ds.variables['cross_shore'][-1]))
        x_f = np.array((self.ds.variables['x'][aidx,0], self.ds.variables['x'][aidx,-1]))
        y_f = np.array((self.ds.variables['y'][aidx,0], self.ds.variables['y'][aidx,-1]))
        px = np.polyfit(cs_f, x_f, 1)
        py = np.polyfit(cs_f, y_f, 1)
        x = np.polyval(px, cs)
        y = np.polyval(py, cs)
        return x,y
    def initcc(self):
        """
        initialize coordinate conversion
        """
        if not hasattr(self, 'rd2latlon'):
            from osgeo.osr import SpatialReference, CoordinateTransformation
            
            # Define the Rijksdriehoek projection system (EPSG 28992)
            epsg28992 = SpatialReference()
            epsg28992.ImportFromEPSG(28992)
            # correct the towgs84
            epsg28992.SetTOWGS84(565.237,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812)
            
            # Define the wgs84 system (EPSG 4326)
            epsg4326 = SpatialReference()
            epsg4326.ImportFromEPSG(4326)
            self.rd2latlon = CoordinateTransformation(epsg28992, epsg4326)
            #latlon2rd = CoordinateTransformation(epsg4326, epsg28992)
            # Check the transformation (in case of a missing towgs84)
            #latlonz = rd2latlon.TransformPoint(155000.0, 446000.0)
            #print latlonz # (5.387202946158022, 52.00237563479786, 43.6057764403522)
            
    def cross_shore2lonlat(self, cs, transect_id, axis=None):
        """
        returns WGS84 (lat,lon) coordinates (epsg 4326) for cross-shore coordinate(s) (wrt to RSP)
        """
        x,y = self.cross_shore2xyRD(cs, transect_id, axis=axis)
        self.initcc()
        xy = zip(x,y)
        lat,lon,_ = zip(*self.rd2latlon.TransformPoints(xy))
        return lon,lat
        
    def MKL(self, x=None, z=None, lower=-1, upper=3):
        """
        volume based instantaneous shoreline position (momentane kustlijn ligging; MKL)
        if x and z are provided, they should be 1D arrays.
        if not, x (cross-shore) and z (altitude) are obtained using the available filter settings
        """
        if (upper-lower)<=0:
            # boundaries have to consistent (upper>lower)
            logger.warning('No MKL can be derived with inconsistent boundaries (lower=%g, upper=%g)'%(lower,upper))
            return None

        from shapely.geometry import asShape 
        import shapely.geometry
        
        if x is None and z is None:
            x = self.get_data('cross_shore')
            z = self.get_data('altitude')
            xMKL = np.ones(z.shape[:2]) * np.nan
            zMKL = np.ones(z.shape[:2]) * np.nan
            for it in np.arange(z.shape[0]):
                for il in np.arange(z.shape[1]):
                    mask = z[it,il,].mask
                    result = self.MKL(x=x[~mask], z=z[it,il,].data[~mask], lower=lower, upper=upper)
                    if result:
                        xMKL[it,il] = result['mkl'][0]
                        zMKL[it,il] = result['mkl'][1]
            return xMKL,zMKL
#        try:
#            shapelock.acquire()
        if hasattr(z, 'mask'):
            logger.debug('only non-masked values are retained')
            x = x[z.mask]
            z = z.data[z.mask]
        if len(x) < 3:
            logger.debug('x vector has only %i elements where at least 2 are required', len(x))
            return None
        # look up coordinates
        X = np.c_[x, z]

        # define an interpolation function
        f = interp1d(x, z, kind='linear',bounds_error=False, copy=True)

        # convert them to a shape
        # look up the bounds of the profile
        min_x = x.min()
        min_z = z.min()
        max_x = x.max()

        # we do not want any double points, cause that invalidates a polygon (SFS)
        # go down one extra, because we don't want to go backward through the same points
        coords = np.r_[X,[[max_x, min_z-1],[min_x, min_z-1], X[0,:]]]
        # poly_x = asShape(shapely.geometry.asPolygon(coords))
        poly_x = shapely.geometry.Polygon(coords.astype('float'))
        assert poly_x.is_valid

        # look up the lower intersections with the lower and upper boundary
        # lower
        line_lower = asShape(shapely.geometry.asLineString([[min_x, lower], [max_x, lower]]))
        assert line_lower.is_valid
        intersects_lower = (line_lower.intersection(poly_x))
        assert intersects_lower.is_valid 
        # upper
        line_upper = asShape(shapely.geometry.asLineString([[min_x, upper], [max_x, upper]]))
        assert line_upper.is_valid
        intersects_upper = (line_upper.intersection(poly_x))
        assert intersects_upper.is_valid

        if intersects_lower.is_empty or intersects_upper.is_empty:
            logger.debug('one or both boundaries does not intersect with profile')
            return None
        
        # by using the bounds, the number of intersections doesn't matter
        swb = intersects_lower.bounds[2]
        lwb = intersects_upper.bounds[2]

        # calculate mkl using maximum method
        boundary_box = shapely.geometry.asPolygon([[lwb,upper], [lwb, lower], [swb,lower], [swb, upper], [lwb,upper]])
        mkl_volume = boundary_box.intersection(poly_x)
        if boundary_box.area+mkl_volume.area == 0:
            return None
        mkl_x = lwb + (swb-lwb)*(mkl_volume.area/(boundary_box.area+mkl_volume.area))
        mkl_y = f(mkl_x)

        result = {}

        result['mkl'] = asarray([mkl_x, mkl_y])
        result['lwb'] = asarray([lwb, upper])
        result['swb'] = asarray([swb, lower])
        result['mkl_volume'] = mkl_volume

        result['X'] = X
#        finally:
#            shapelock.release()
        return result
    def get_jrk(self):
        """
        Convert current selection of data to .jrk string
        """
        fmt = '%6i %6i'
        years = self.time2year(self.get_data('time'))
        z = self.get_data('altitude')
        o = self.get_data('origin')
        aids = self.get_data('id')
        x = self.get_data('cross_shore')
        mhw = self.get_data('mean_high_water')
        mlw = self.get_data('mean_low_water')
        max_y = self.get_data('max_altitude_measurement')
        min_y = self.get_data('min_altitude_measurement')
        time_topo = self.get_data('time_topo')
        time_bathy = self.get_data('time_bathy')
        ###s = ''
        s = []
        for ia,aid in enumerate(aids):
            for i,year in enumerate(years):
                zc = np.ma.masked_invalid(np.squeeze(z[i,ia,:]))
                idx = zc.mask == False
                nx = np.count_nonzero(idx)
                if nx == 0:
                    continue
                zc = zc[idx]
                xc = x[idx]
                data = list(zip(xc, zc))
                if not nx%5 == 0:
                    # fill incomplete rows with dummy values
                    dummyvals = [(99999, 999999)] * (5-nx%5)
                    data = data + dummyvals
                s.append([aid, year, data, mhw, mlw, max_y, min_y, time_topo, time_bathy]) 
        return s
    def filter_jrk(self, idx, years_req=[]):
        a = []
        self.set_filter(alongshore=idx, year=years_req)
        a = self.get_jrk()

        # Convert retrieved data for easy visualisation
        x_values = []
        y_values = []  
        years_included = []
        for i in range(len(a)):
            years_included.append(a[i][1])
        
            x = np.array([i[0] for i in a[i][2]])
            x[x == 99999] = np.nan # Convert 99999 to nan values, so they are not included in visualisation
            x_values.append(x)
            
            y = np.array([i[1] for i in a[i][2]])
            y[y == 99999] = np.nan # Convert 99999 to nan values, so they are not included in visualisation
            y_values.append(y)
    
        # Find difference between years included and those requested
        years_missing = list(set(years_req) - set(years_included))
        # If all years were available do nothing,
        if len(years_missing) == 0:
            print("All requested years were available")
            # If years were missing, show error
        else: 
            print("ERROR - For transect {} the following year(s) were not available:".format(a[0][0]))
            print(years_missing)
        return a, x_values, y_values, years_included
    
	# CHRISTA: I build this function to directly convert the data to pandas dataframes. Just because I like working with them better.
    def get_dataframe(self, transect, years_requested): 
        import pandas as pd
        ids = self.get_data('id')
        idxs = np.isin(ids, transect)
        df_jrk_all = pd.DataFrame()
        for idx in np.nonzero(idxs)[0]:   
            a, x_values, y_values, years_included = self.filter_jrk(idx,years_requested)
            df_jrk_yrs = pd.DataFrame()
            
            for indx, yr in enumerate(years_included):
                df_jrk = pd.DataFrame()
                x = x_values[indx]
                y = y_values[indx]
                mhw = float(a[indx][3][0])
                mlw = float(a[indx][4][0])
                max_y = float(a[indx][5][years_requested.index(yr)])
                min_y = float(a[indx][6][years_requested.index(yr)])
                time_topo = float(a[indx][7][years_requested.index(yr)])
                time_bathy = float(a[indx][8][years_requested.index(yr)])
                
                trsct = str(a[indx][0])
                yr = str(a[indx][1])
                
                df_jrk = pd.DataFrame({'transect': trsct, 'year': yr, 'x': x, 'y': y, 'mhw': mhw, 'mlw': mlw, 'max_elevation': max_y, 'min_elevation': min_y, 'time_topo': time_topo, 'time_bathy': time_bathy})
                
                df_jrk_yrs = df_jrk_yrs.append(df_jrk)
            df_jrk_all = df_jrk_all.append(df_jrk_yrs)
        df_jrk_all.set_index(['transect', 'year'], inplace=True)
        years_included = [str(y) for y in years_included]
            
        return df_jrk_all, years_included
