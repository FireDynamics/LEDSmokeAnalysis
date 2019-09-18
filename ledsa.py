import sys
import numpy as np
import matplotlib.pyplot as plt

import led_helper as led

class LEDSA:
    
#switch error handling to led_helper if possible!!!
#by making led.open_file('path/filename')
#when exception is thrown, ask if std conffile should be used
    def __init__(self):
        #configuration class with variables:
        #root_directory
        try:
            self.conf = led.load_config()
        except OSError as e:
            print('An operation system error occured while loading the config ',
                  'file. Maybe the file is not there or there is no reading ',
                  'permission.\n Error Message: ',e)
        except FormatError as e:
            print('The config file could not successfully be red, as there are',
                  ' some formatting errors inside. \n Error Message: ',e)
        except Exception as e:
            print('Some unknown error has occured. \n Error Message: ',e)
        else:
            print('Config file successfully loaded.')
            
        #declarations of global variables
        self.search_areas = False

    def find_search_areas(self, img_filename):
        filename = "data/{}/{}".format(self.conf.root_directory, img_filename)
        out_filename = 'out/{}/led_search_areas.csv'.format(self.conf.root_directory)
        
        data = led.read_file(filename, channel=0)
        self.search_areas = led.find_search_areas(data, skip=1,
                                                  window_radius =
                                                  self.conf.window_radius)
      
        np.savetxt(out_filename, self.search_areas,
                   header='LED id, pixel position x, pixel position y', fmt='%d')
    
    def load_search_areas(self):
        filename = 'out/{}/led_search_areas.csv'.format(self.conf.root_directory)
        try:
            self.search_areas = np.loadtxt(filename, skiprows=1)
        except Exception as e:
            print('Could not load the search areas from{}'.format(filename))
            print('Error Message: ', e)
    
    def plot_search_areas(self, img_filename):
        if type(self.search_areas) == bool:
            self.load_search_areas()
        
        filename = "data/{}/{}".format(self.conf.root_directory, img_filename)    
        data = led.read_file(filename, channel=0)
            
        fig = plt.figure(dpi=1200)
        ax = plt.gca()

        for i in range(self.search_areas.shape[0]):
            ax.add_patch(plt.Circle((self.search_areas[i,2], self.search_areas[i,1]), 
                                    radius=self.conf.window_radius,
                                    color='Red', fill=False, alpha=0.25,
                                    linewidth=0.1))
            ax.text(self.search_areas[i,2] + self.conf.window_radius, 
                    self.search_areas[i,1] + self.conf.window_radius//2, 
                    '{}'.format(self.search_areas[i,0]), fontsize=1)
        
        plt.imshow(data, cmap='Greys')
        plt.colorbar()
        plt.savefig('out/{}/led_search_areas.plot.pdf'.format(
                    self.conf.root_directory))
