#!/usr/bin/env python

import argparse
import numpy as np
import yaml
from multiprocessing import Pool
import sys
from pyGSI.diags import Conventional
from pyGSI.plot_diags import plot_spatial, plot_histogram, plot_vertical, _calculate_stats
from datetime import datetime

start_time = datetime.now()

def plotting(conv_config, lvls, lvl_type):

    print( "--> plotting\n")

    diagfile = conv_config['conventional input']['path'][0]
    diag_type = conv_config['conventional input']['data type'][0].lower()
    obsid = conv_config['conventional input']['observation id']
    analysis_use = conv_config['conventional input']['analysis use'][0]
    plot_type = conv_config['conventional input']['plot type']
    outdir = conv_config['outdir']

    diag = Conventional(diagfile)

    print( "diagfile  = " + diagfile )
    print( "diag_type = " + diag_type )
    print( "obsid     = ", *obsid)
    print( "plot_type = ", *plot_type )
    print( "analysis_use = ", analysis_use )
    print( f'diag: {diag}\n' )

    if analysis_use:

        diag_components = diagfile.split('/')[-1].split('.')[0].split('_')

        if diag_components[1] == 'conv' and diag_components[2] == 'uv':
            rtn_data = diag.get_data(diag_type, obsid=obsid,
                                 analysis_use=analysis_use, lvls=lvls, lvl_type=lvl_type)
            
            data = { 'u' : {'assimilated' : {},
                            'monitored'   : {}},
                     'v' : {'assimilated' : {},
                            'monitored'   : {}},
                     'windspeed' : {'assimilated' : {},
                                    'monitored'   : {}} 
                   }

            for key in rtn_data.keys():
                data['u']['assimilated'][key] = rtn_data[key]['u']['assimilated']
                data['v']['assimilated'][key] = rtn_data[key]['v']['assimilated']
                data['windspeed']['assimilated'][key] = np.sqrt( np.square(rtn_data[key]['u']['assimilated']) + 
                                                                 np.square(rtn_data[key]['v']['assimilated']) )
                data['u']['monitored'][key] = rtn_data[key]['u']['monitored']
                data['v']['monitored'][key] = rtn_data[key]['v']['monitored']
                data['windspeed']['monitored'][key] = np.sqrt( np.square(rtn_data[key]['u']['monitored']) + 
                                                                 np.square(rtn_data[key]['v']['monitored']) )
        else:
            rtn_data = diag.get_data( diag_type, obsid=obsid,
                                 analysis_use=analysis_use, lvls=lvls, lvl_type=lvl_type )

            data = { 'assimilated' : {},
                     'monitored'   : {} }

            for key in rtn_data.keys():
                data['assimilated'].update( {key: rtn_data[key]['assimilated']} )
                data['monitored'].update( {key: rtn_data[key]['monitored']} )

        metadata = diag.metadata

#        if np.isin('histogram', plot_type):
#            plot_histogram(data, metadata, outdir)
#        if np.isin('spatial', plot_type):
#            lats, lons = diag.get_lat_lon(obsid=obsid, analysis_use=analysis_use)
#            plot_spatial(data, metadata, lats, lons, outdir)
        if np.isin('vertical', plot_type):
            plot_vertical(data, metadata, outdir)

    else:

        diag_components = diagfile.split('/')[-1].split('.')[0].split('_')
        if diag_components[1] == 'conv' and diag_components[2] == 'uv':

            rtn_data = diag.get_data(diag_type, obsid=obsid,
                                 analysis_use=analysis_use, lvls=lvls, lvl_type=lvl_type)
            data = { 'u' : {},
                     'v' : {},
                     'windspeed' : {}
                   }

            for key in rtn_data.keys():
                data['u'][key] = rtn_data[key]['u'] 
                data['v'][key] = rtn_data[key]['v']
                data['windspeed'][key] = np.sqrt( np.square(rtn_data[key]['u']) + np.square(rtn_data[key]['v']) )

        else:
            data = diag.get_data( diag_type, obsid=obsid,
                                 analysis_use=analysis_use, lvls=lvls, lvl_type=lvl_type )


        metadata = diag.metadata

        if np.isin('histogram', plot_type):
            data = diag.get_data( diag_type, obsid=obsid )
            plot_histogram(data, metadata, outdir)
        if np.isin('spatial', plot_type):
            lats, lons = diag.get_lat_lon(obsid=obsid)
            plot_spatial(data, metadata, lats, lons, outdir)
        if np.isin('vertical', plot_type):
            plot_vertical(data, metadata, outdir)
    

###############################################
###############################################


if __name__ == '__main__':
    # called from command line
    print( "--> test_driver, command line\n")

    # Parse command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--nprocs",
                help="Number of tasks/processors for multiprocessing")
    ap.add_argument("-y", "--yaml",
                help="Path to yaml file with diag data")
    ap.add_argument("-o", "--outdir",
                help="Out directory where files will be saved")

    myargs = ap.parse_args()

    if myargs.nprocs:
        nprocs = int(myargs.nprocs)
    else:
        nprocs = 1

    input_yaml = myargs.yaml
    outdir = myargs.outdir

    #with open(input_yaml, 'r') as file:
    #    parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)
    with open( input_yaml ) as parameters:
      parsed_yaml_file = yaml.safe_load(parameters)

    work = (parsed_yaml_file['diagnostic'])

    for w in work:
        w['outdir'] = outdir

    #p = Pool(processes=nprocs)
    #p.map(plotting, work)

    cfig=work[0]

    #-------------------------------------------------------------------------
    # When read in directly lvls ends up as a list of 1 item.  Convert that
    # to a list of n items, by first removing any white space with translate, 
    # then splitting the string on ',' and coverting to list.
    #
    lvls = cfig['conventional input']['levels']
    if lvls != None:
        lvls_str = str(lvls[0])
        lvls_str = lvls_str.translate({ord(i):None for i in ' '})
        lvls = list(lvls_str.split(','))
    
    lvl_type = cfig['conventional input']['level type']
    if lvl_type != None:
        lvl_type = cfig['conventional input']['level type'][0]
  
    plotting( cfig, lvls, lvl_type )

    print( "<-- test_driver\n")
    print(datetime.now() - start_time)
