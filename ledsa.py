import sys
import numpy as np
import matplotlib.pyplot as plt

import led_helper as led

class LEDSA:
    
    def __init__(self, use_config=True, root_directory = 'root', 
                 window_radius = 10):
        #configuration class with variables:
        #root_directory
        #window_radius
        
        if use_config:
            self.conf = led.load_config()
            
        else:
            self.config = ConfigData([root_directory, window_radius])
            
        #declarations of global variables
        self.search_areas = False
        self.line_indices = False

    """
    ------------------------------------
    LED area search
    ------------------------------------
    """
    
    """finds all LEDs in a single image file and defines the search areas, in
    which future LEDs will be searched"""
    def find_search_areas(self, img_filename):
        filename = "data/{}/{}".format(self.conf.root_directory, img_filename)
        out_filename = 'out/{}/led_search_areas.csv'.format(self.conf.root_directory)
        
        data = led.read_file(filename, channel=0)
        self.search_areas = led.find_search_areas(data, skip=1,
                                                  window_radius =
                                                  self.conf.window_radius)
      
        np.savetxt(out_filename, self.search_areas,
                   header='LED id, pixel position x, pixel position y', fmt='%d')


    """loads the search areas from the csv file"""    
    def load_search_areas(self):
        filename = 'out/{}/led_search_areas.csv'.format(self.conf.root_directory)
        self.search_areas = led.load_file(filename)


    """plots the search areas with their labels"""    
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

    """
    ------------------------------------
    LED array analysis
    ------------------------------------
    """

    """analyses, which LED belongs to which LED line array"""
    def analyse_positions(self):       
        if type(self.search_areas) == bool:
            self.load_search_areas() 
        
        led.analyse_position_man(self.search_areas, self.conf, self.line_indices)
                
        #plot the labeled LEDs
        
        #img_data = led.load_file('data/{}/{}'.format(self.conf.root_directory,img_filename))        
        #fig = plt.figure(dpi=900)        
        #plt.imshow(img_data, cmap='Greys')
        for i in range(len(line_indices)):
            plt.scatter(xs[line_indices[i]], ys[line_indices[i]], s=0.1, label='led strip {}'.format(i))
        
        plt.legend()
        plt.savefig('out/{}/led_lines.pdf'.format(self.conf.root_directory))
        
        #save the labeled LEDs
        for i in range(len(line_indices)):
            out_file = open('out/{}/line_indices_{:03}.csv'.format(self.conf.root_directory, i), 'w')
            for iled in line_indices[i]:
                out_file.write('{}\n'.format(iled))
            out_file.close()

    """
    -----------------------------------------
    usufull functions from the helper module
    -----------------------------------------
    """

    def shell_in_ignored_indices(self):
        led.shell_in_ignored_indices()
        
    def shell_in_line_edge_indices(self):
        led.shell_in_line_edge_indices()

"""
------------------------------------
Default script
------------------------------------
"""
    
if __name__=='main':
    None