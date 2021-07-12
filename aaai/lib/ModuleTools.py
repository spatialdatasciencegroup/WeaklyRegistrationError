import os, sys, tempfile, csv, time, shutil
import numpy as np
import rasterio as rio
import geopandas as gpd

from datetime import datetime as dt 


from varname import Wrapper, varname, nameof

from lib.internal import _ctype, _cpath


ROOT_TEST_DIR = '/data/GeometricErrors/tests/aaai_system'


"""
ModuleTest() Attributes:
- module (dict): module defaults and data.
- name (str): name of this test (without ending)
- ending (str): the ending to be applied to files for this test 
- parameter_names (dict): dict of parameter group names, and a list of the parameter names they hold.
    - {'param_group': ['param_1', 'param_2'], ...} 
- section_datas (dict of dicts): each key being a section name referencing the section sub dict:
    - section_dict: {'metric_name': metric_value, ...}
        Defaults:
        - 'name': name of section
        - 'timestamp': time stamped at this section
        - 'runtime': time to run this section
- metrics (dict of dicts): {metric_group: m_group_dict, ...}
    - metric_dict: {'metric_name': metric_value}

Attributes
- dir (str): root dir for this test.

ModuleTest() Methods:

"""

class ModuleTest():
    """ Single test in module. """
    def __init__(self, module, **kwargs):
        super().__init__()
        # Update Module dict
        self.module = module
        self.ModuleUpdate()

        # Update ending if name was specified
        if 'name' in kwargs:
            if kwargs['name'] == None: 
                print("Warning (ModuleTest): Invalid name 'None', using module default.")
            elif isinstance(kwargs['name'], str) and len(kwargs['name']) > 3:
                self.name = kwargs['name']
                self.end = '_{}'.format(self.name)
            else:
                print("Warning (ModuleTest): Passed name is either too short, or invalid. Using module default.")

        self.SetDir()

        self.desc = input('Describe {}:'.format(self.name))
        self.paramNames = []
        self.paramDicts = []
        self.fileNames = [] 
        self.fileDicts = []
        self.resultNames = [] 
        self.resultDicts = []
        self.sectionNames = [] 
        self.sectionDicts = []

    def ModuleUpdate(self):
        self.root_path = os.path.join(ROOT_TEST_DIR, self.module['name'])
        self.csv = os.path.join(self.root_path, self.module['csv'])

        # ensure roor dir exists
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        # Ensure csv exists
        if not os.path.exists(self.csv):
            with open(self.csv, 'w+', newline='\n') as csvfile:
                main_csv = csv.writer(csvfile, delimiter=',')
                main_csv.writerow([self.module['name'] + ' - Data', self.module['desc']])
        
        # get test index
        with open(self.csv, 'r', newline='\n') as test_csv:
            test_csv = csv.reader(test_csv, delimiter=',')
            self.idx = sum(1 for row in test_csv)

        # Set a default test name
        self.end = '_{:02}'.format(self.idx)
        self.name = self.module['test_name'] + self.end

        self.verbose = self.module['verbose']

    def SetDir(self):
        self.dir = os.path.join(self.root_path, self.name)
        if os.path.exists(self.dir):
            delete_old = input("{} was already run, delete old folder? ('y' to delete).".format(self.name))
            if delete_old == 'y':
                # Delete old dir
                shutil.rmtree(self.dir)
                # Delete test from csv
                csvReader = open(self.csv, 'r', newline='\n')
                rows = [row for row in csv.reader(csvReader, delimiter=',') if row[0] != self.name]
                csvReader.close()
                os.remove(self.csv)
                with open(self.csv, 'a+', newline='\n') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    for row in rows:
                        writer.writerow(row)
            else:
                print("Exiting test, please delete old folder or update {} master csv.".format(self.name))
                sys.exit(0)
        os.mkdir(self.dir)

    def Folder(self, fname):
        """ Get the folder of a folder name in test_dir. """
        fpath = os.path.join(self.dir, fname + self.end)
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        return fpath

    def Pgroup(self, groupname, p_dict):
        """ Add pgroup name and parameters to struct. """
        self.paramNames.append(groupname)
        self.paramDicts.append(p_dict)
    
    def Fgroup(self, groupname, f_dict):
        """ Save File Group name and dict. """
        self.fileNames.append(groupname)
        self.fileDicts.append(f_dict)

    def Section(self, section_name, section_dict):
        """ Add Section name and dict. """
        self.sectionNames.append(section_name)
        self.sectionDicts.append(section_dict)

    def Results(self, results_name, results_dict, writeCSV=False):
        """ Add results Group name and dict. """
        self.resultNames.append(results_name)
        self.resultDicts.append(results_dict)

        if writeCSV:
            csvfp = os.path.join(self.root_path, (results_name.replace(" ", "") + '.csv'))
            header = [results_name]
            data = [self.name]
            for key, item in results_dict.items():
                header.append(key)
                data.append(item)
            csvExists = os.path.exists(csvfp)
            with open(csvfp, 'a+', newline='\n') as csvfile:
                results_csv = csv.writer(csvfile, delimiter=',')
                if not csvExists: 
                    results_csv.writerow(header)
                results_csv.writerow(data) 

    def index(self):
        """ index test on master csv. """
        with open(self.csv, 'a+', newline='\n') as csvfile:
            main_csv = csv.writer(csvfile, delimiter=',')
            main_csv.writerow([self.name, self.desc, dt.now().strftime('%a at %I:%M:%S%p')])
    
    def markdown(self):
        """ Write test data to markdown. """
        markdownfp = os.path.join(self.dir, 'info_{}.md'.format(self.name))
        info_md = open(markdownfp, 'w+')
        info_md.write("# {} - {} Info".format(self.module['name'], self.name)) 
        info_md.write("\n\n### {} - {}".format(self.desc, dt.now().strftime('%a at %I:%M:%S%p')))
        info_md.write("\n\n---")

        info_md.write("\n\n## Parameters:")
        for idx, pgroup_name in enumerate(self.paramNames):
            info_md.write("\n\n### {}:".format(pgroup_name))
            for name, param in self.paramDicts[idx].items():
                info_md.write("\n - {} ({}): `{}`".format(name, type(param).__name__, param))
        info_md.write("\n\n---")

        info_md.write("\n\n## Inputs:")
        for idx, fgroup_name in enumerate(self.fileNames):
            info_md.write("\n\n### {}:".format(fgroup_name))
            for name, param in self.fileDicts[idx].items():
                info_md.write("\n - {} ({}):".format(name, type(param[1]).__name__))
                info_md.write("\n   `{}`".format(param[0]))
        info_md.write("\n\n---")

        info_md.write("\n\n## Results:")
        if len(self.resultNames) > 0:
            for idx, result_name in enumerate(self.resultNames):
                info_md.write("\n\n### {}:".format(result_name))
                for key, item in self.resultDicts[idx].items():
                    info_md.write("\n - {} ({}): `{}`".format(key, type(item).__name__, item))
        info_md.write("\n\n---")

        info_md.write("\n\n## Other Test Data:")
        info_md.write("\n - Ending: `'{}'`".format(self.end))
        info_md.write("\n - Dir: `{}`".format(self.dir))
        for idx, section_name in enumerate(self.sectionNames):
            info_md.write("\n - **Section {:02}**: `{}`\n    - Time: `{:.3f} sec`".format(idx, section_name, self.sectionDicts[idx]['time']))
        info_md.close()

def wrap(val, p_dict):
    """ Takes any value, returns a wrapped variable, and optionally appends it to a dict of parameters."""
    try:
        p_dict.update({varname(): val})
    except TypeError:
        print("Error (wrap): Parameter Dict {} has invalid type ({}).".format(varname(), (p_dict)))
        sys.exit(0)
    return val

def wrap_fp(filepath, p_dict=None):
    """ Takes and opens any filepath, then returns a wrapped variable. """
    f_path, f_ext = os.path.splitext(filepath)
    f_path, f_name = os.path.split(f_path)

    if f_ext == '.npy':
        value = np.load(filepath)
    elif f_ext == '.shp':
        value = gpd.read_file(filepath)
    elif f_ext == '.tif':
        value = rio.open(filepath)
    else:
        print("Error (wrap_fp): Can't open extension {}. Only accepting .tif, .npy, .shp.")
        sys.exit(0)
    if p_dict != None:
        try:
            p_dict.update({f_name: (filepath,value)})
        except TypeError:
            print("Error (wrap_fp): Parameter Dict {} has invalid type ({}).".format(varname(), (p_dict)))
            sys.exit(0)       
    return value
