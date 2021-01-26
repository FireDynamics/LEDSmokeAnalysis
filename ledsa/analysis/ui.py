import ledsa.analysis.plot_functions as pf
import matplotlib.pyplot as plt
import matplotlib
import ledsa.core.led_helper as led
import tkinter as tk
import subprocess
import os

matplotlib.use('TkAgg')

LARGE_FONT = ('Verdana', 14)


class GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        self.title('LEDSA Analyser')

        container.pack(side='top', fill='both', expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        frame = MainPage(container, self)

        self.frames[MainPage] = frame

        frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(MainPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class MainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        plt.ion()

        self.led_id = 0
        self.img_id = 1
        self.channel = 0
        self.plot_par = 'A'
        self.lines = [3]
        self.time = 0
        self.use_time = tk.BooleanVar()
        self.use_time.set(1)
        self.fig = plt.figure()

        label = tk.Label(self, text='LEDSA Analyser', font=LARGE_FONT)
        label.grid(columnspan='4', sticky='N', padx=10, pady=10)

        new_l = tk.Label(self, text='Generate new Figure')
        new_b = tk.Button(self, text='New', command=self.new_figure)

        time_l = tk.Label(self, text='Time Plot')
        time_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindowPlots(self, 't'))

        time_av_l = tk.Label(self, text='Time Plot with Average')
        time_av_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindowPlots(self, 't_av'))

        height_l = tk.Label(self, text='Height Plot')
        height_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindowPlots(self, 'z'))

        height_av_l = tk.Label(self, text='Height Plot from Time Average')
        height_av_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindowPlots(self, 'z_t_av'))

        img_l = tk.Label(self, text='Show a whole Image')
        img_b = tk.Button(self, text='Open', command=lambda: InputArgumentWindowShowImg(self))

        led_ids_l = tk.Label(self, text='Show PDF with LED Labels')
        led_ids_b = tk.Button(self, text='Open',
                              command=lambda: subprocess.Popen('evince ' + os.sep.join([os.curdir, 'plots',
                                                                                        'led_search_areas.plot.pdf']),
                                                               shell=True))

        led_fit_l = tk.Label(self, text='Show LED with Fit')
        led_fit_b = tk.Button(self, text='Show', command=lambda: InputArgumentWindowShowLed(self))

        led_diff_l = tk.Label(self, text='Show Difference between LEDs')
        led_diff_b = tk.Button(self, text='Show', command=lambda: InputArgumentWindowShowLedDiff(self))

        new_l.grid(row=1)
        new_b.grid(row=1, column=1)
        time_l.grid(row=2)
        time_b.grid(row=2, column=1)
        time_av_l.grid(row=3)
        time_av_b.grid(row=3, column=1)
        height_l.grid(row=4)
        height_b.grid(row=4, column=1)
        height_av_l.grid(row=5)
        height_av_b.grid(row=5, column=1)
        img_l.grid(row=1, column=2)
        img_b.grid(row=1, column=3)
        led_ids_l.grid(row=2, column=2)
        led_ids_b.grid(row=2, column=3)
        led_fit_l.grid(row=3, column=2)
        led_fit_b.grid(row=3, column=3)
        led_diff_l.grid(row=4, column=2)
        led_diff_b.grid(row=4, column=3)

    def new_figure(self):
        self.fig = plt.figure()

    def update_form_presets(self, entries):
        for entry in entries:
            if entry[0] == 'Channel':
                self.channel = int(entry[1].get())
            if entry[0] == 'Parameter':
                self.plot_par = entry[1].get()
            if entry[0] == 'LED ID':
                self.led_id = int(entry[1].get())
            if entry[0] == 'Image ID':
                self.img_id = int(entry[1].get())
            if entry[0] == 'LED Arrays':
                lines_str = entry[1].get().split(sep=' ')
                self.lines = []
                for line in lines_str:
                    self.lines.append(int(line))
            if entry[0] == 'Time':
                self.time = int(entry[1].get())


class InputArgumentsWindow(tk.Toplevel):

    def __init__(self, master, fields, inserts):
        tk.Toplevel.__init__(self, master=master)
        self.inserts = inserts
        self.fields = fields

    def generate_form(self):
        entries = []
        for field, insert in zip(self.fields, self.inserts):
            row = tk.Frame(self)
            lab = tk.Label(row, width=15, text=field, anchor='w')
            ent = tk.Entry(row)
            ent.insert(10, insert)
            row.pack(side='top', fill='x', padx=5, pady=5)
            lab.pack(side='left')
            ent.pack(side='right', expand='YES', fill='x')
            entries.append((field, ent))
        return entries

    def switch_time_id(self):
        tk.Label(self, text='Which Variable should be used:').pack(side='top', fill='x', padx=5, pady=5)

        tk.Radiobutton(self, text='Image ID', variable=self.master.use_time, value=0).pack(side='left',
                                                                                           padx=5, pady=5)
        tk.Radiobutton(self, text='Time', variable=self.master.use_time, value=1).pack(side='left', padx=5, pady=5)

    def save_entries(self, entries):
        self.master.update_form_presets(entries)

    def add_show_close_buttons(self, entries, show_func):
        b1 = tk.Button(self, text='Show',
                       command=lambda: show_func(entries))
        b1.pack(side='left', padx=5, pady=5)

        b2 = tk.Button(self, text='Close', command=self.close)
        b2.pack(side='left', padx=5, pady=5)

    def close(self):
        self.destroy()


class InputArgumentsWindowPlots(InputArgumentsWindow):

    def __init__(self, master, plot_type):
        self.plot_type = plot_type
        if plot_type == 't' or plot_type == 't_av':
            self.form_type = 'time'
        else:
            self.form_type = 'height'

        if self.form_type == 'time':
            InputArgumentsWindow.__init__(self, master, ['Channel', 'Parameter', 'LED ID'],
                                          [master.channel, master.plot_par, master.led_id])
        else:
            InputArgumentsWindow.__init__(self, master, ['Channel', 'Parameter', 'Image ID', 'LED Arrays'],
                                          [master.channel, master.plot_par, master.img_id, master.lines])

        self.title(f'Arguments for {self.form_type} plot')

        entries = self.generate_form()

        b1 = tk.Button(self, text='Add',
                       command=lambda: self.save_and_plot(entries))
        b1.pack(side='left', padx=5, pady=5)

        b2 = tk.Button(self, text='Close', command=self.close)
        b2.pack(side='left', padx=5, pady=5)

    def save_and_plot(self, entries):
        self.save_entries(entries)
        self.plot()

    def plot(self):
        if self.plot_type == 't':
            pf.plot_t_fitpar(self.master.fig, self.master.led_id, self.master.plot_par, self.master.channel, 1,
                             led.get_last_img_id())
        if self.plot_type == 't_av':
            pf.plot_t_fitpar_with_moving_average(self.master.fig, self.master.led_id, self.master.plot_par,
                                                 self.master.channel, 1, led.get_last_img_id())
        if self.plot_type == 'z':
            pf.plot_z_fitpar(self.master.fig, self.master.plot_par, self.master.img_id, self.master.channel,
                             self.master.lines)
        if self.plot_type == 'z_t_av':
            pf.plot_z_fitpar_from_average(self.master.fig, self.master.plot_par, self.master.img_id,
                                          self.master.channel, self.master.lines)

        plt.draw()


class InputArgumentWindowShowImg(InputArgumentsWindow):

    def __init__(self, master):
        InputArgumentsWindow.__init__(self, master=master, fields=['Image ID', 'Time'],
                                      inserts=[master.img_id, master.time])

        self.title('Show Image')

        entries = self.generate_form()
        self.switch_time_id()
        self.add_show_close_buttons(entries, self.show)

    def show(self, entries):
        self.save_entries(entries)
        if self.master.use_time.get():
            pf.show_img(time=self.master.time)
        else:
            pf.show_img(img_id=self.master.img_id)
        self.close()


class InputArgumentWindowShowLed(InputArgumentsWindow):

    def __init__(self, master):
        InputArgumentsWindow.__init__(self, master, ['Channel', 'Image ID', 'Time', 'LED ID'],
                                      [master.channel, master.img_id, master.time, master.led_id])

        entries = self.generate_form()
        self.switch_time_id()
        self.add_show_close_buttons(entries, self.show)

    def show(self, entries):
        self.save_entries(entries)
        if self.master.use_time.get():
            time = self.master.time
        else:
            time = led.get_time_from_img_id(self.master.img_id)
        pf.plot_led_with_fit(self.master.channel, time, self.master.led_id)
        self.close()


class InputArgumentWindowShowLedDiff(InputArgumentsWindow):

    def __init__(self, master):
        InputArgumentsWindow.__init__(self, master, ['Channel', 'Image ID', 'Image ID 2', 'Time', 'Time 2', 'LED ID'],
                                      [master.channel, master.img_id, master.img_id + 1, master.time, master.time + 1,
                                       master.led_id])

        entries = self.generate_form()
        self.switch_time_id()
        self.add_show_close_buttons(entries, self.show)

    def show(self, entries):
        self.save_entries(entries)
        if self.master.use_time.get():
            time1 = self.master.time
            time2 = int(entries[4][1].get())
        else:
            time1 = led.get_time_from_img_id(self.master.img_id)
            time2 = led.get_time_from_img_id(int(entries[2][1].get()))
        pf.show_led_diff(self.master.channel, self.master.led_id, time1, time2)
        self.close()
