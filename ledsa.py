import sys
import numpy as np
import matplotlib.pyplot as plt

import led_helper as led

class LEDSA:
    
    def __init__(self, use_config_file = True, root_directory = 'root', 
                 window_radius = 10):
        #configuration class with variables:
        #root_directory
        #window_radius
        
        if use_config_file:
            self.conf = led.load_config()
            
        else:
            self.config = ConfigData([root_directory, window_radius])
            
        #declarations of global variables
        #2D numpy array wit dimension (# of LEDs) x (LED_id, x, y)
        self.search_areas = False
        #2D list with dimension (# of LED arrays) x (# of LEDs per array)
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
      
        np.savetxt(out_filename, self.search_areas, delimiter=',',
                   header='LED id, pixel position x, pixel position y', fmt='%d')


    """loads the search areas from the csv file"""    
    def load_search_areas(self):
        filename = 'out/{}/led_search_areas.csv'.format(self.conf.root_directory)
        self.search_areas = led.load_file(filename, delim=',')


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
        
        self.line_indices = led.analyse_position_man(self.search_areas, self.conf)
                       
        #save the labeled LEDs
        for i in range(len(self.line_indices)):
            out_file = open('out/{}/line_indices_{:03}.csv'.format(self.conf.root_directory, i), 'w')
            for iled in self.line_indices[i]:
                out_file.write('{}\n'.format(iled))
            out_file.close()
            
    """loads the search areas from the csv file"""    
    def load_line_indices(self):
        self.line_indices = []
        for i in range(self.conf.num_of_arrays):
            filename = 'out/{}/line_indices_{:03}.csv'.format(self.conf.root_directory, i)
            self.line_indices.append(led.load_file(filename, type = 'int'))
            
    """plot the labeled LEDs"""        
    def plot_lines(self):
        #plot the labeled LEDs
        if type(self.line_indices) == bool:
            self.load_line_indices()
        if type(self.search_areas) == bool:
            self.load_search_areas()
        for i in range(len(self.line_indices)):
            plt.scatter(self.search_areas[self.line_indices[i],2], 
                        self.search_areas[self.line_indices[i],1], 
                        s=0.1, label='led strip {}'.format(i))
        
        plt.legend()
        plt.savefig('out/{}/led_lines.pdf'.format(self.conf.root_directory))
        
    """
    ------------------------------------
    LED smoke analysis
    ------------------------------------
    """
    
    """process the image data to find the changes in light intesity""" 
    def process_image_data(self):
        if type(self.search_areas) == bool:
            self.load_search_areas() 
        if type(self.line_indices) == bool:  
            self.load_line_indices()  
        data_indices = [7460 + 50*i for i in range(10)]           
    
        if self.conf.multicore_processing:
            from multiprocessing import Pool
    
            with Pool(4) as p:
                p.map(self.process_file, data_indices)
        else:
            for i in range(len(data_indices)):
                led.process_file(data_indices[i], self.search_areas, self.line_indices, self.conf)
                print('image ', i+1, '/', len(data_indices)+1, ' processed')

    """workaround for pool map"""
    def porcess_file(self,data_indices):
        led.process_file(data_indices, self.search_areas, self.line_indices, self.conf)
        

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