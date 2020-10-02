import ledsa.analysis.plot_functions as pf
import matplotlib.pyplot as plt
import matplotlib
import ledsa.core._led_helper as led
import tkinter as tk

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
        self.fig = plt.figure()

        label = tk.Label(self, text='LEDSA Analyser', font=LARGE_FONT)
        label.grid(columnspan=2, sticky='N', padx=10, pady=10)

        new_l = tk.Label(self, text='Generate new Figure')
        new_b = tk.Button(self, text='New', command=self.new_figure)

        time_l = tk.Label(self, text='Time Plot')
        time_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindow(self, 't'))

        time_av_l = tk.Label(self, text='Time Plot with Average')
        time_av_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindow(self, 't_av'))

        height_l = tk.Label(self, text='Height Plot')
        height_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindow(self, 'z'))

        height_av_l = tk.Label(self, text='Height Plot from Time Average')
        height_av_b = tk.Button(self, text='Add', command=lambda: InputArgumentsWindow(self, 'z_t_av'))

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

    def new_figure(self):
        self.fig = plt.figure()


class InputArgumentsWindow(tk.Toplevel):

    def __init__(self, master, plot_type):
        tk.Toplevel.__init__(self, master=master)

        self.plot_type = plot_type
        if plot_type == 't' or plot_type == 't_av':
            self.form_type = 'time'
        else:
            self.form_type = 'height'

        self.title(f'Arguments for {self.form_type} plot')

        if self.form_type == 'time':
            entries = self.generate_form(['Channel', 'Parameter', 'LED ID'],
                                         [master.channel, master.plot_par, master.led_id])
        else:
            entries = self.generate_form(['Channel', 'Parameter', 'Image ID', 'LED Arrays'],
                                         [master.channel, master.plot_par, master.img_id, master.lines])

        b1 = tk.Button(self, text='Add',
                       command=lambda: self.save_and_plot(entries))
        b1.pack(side='left', padx=5, pady=5)

        b2 = tk.Button(self, text='Close', command=self.close)
        b2.pack(side='left', padx=5, pady=5)

    def generate_form(self, fields, inserts):
        entries = []
        for field, insert in zip(fields, inserts):
            row = tk.Frame(self)
            lab = tk.Label(row, width=15, text=field, anchor='w')
            ent = tk.Entry(row)
            ent.insert(10, insert)
            row.pack(side='top', fill='x', padx=5, pady=5)
            lab.pack(side='left')
            ent.pack(side='right', expand='YES', fill='x')
            entries.append(ent)
        return entries

    def save_and_plot(self, entries):
        self.master.channel = int(entries[0].get())
        self.master.plot_par = entries[1].get()

        if self.form_type == 'time':
            self.master.led_id = int(entries[2].get())

        if self.form_type == 'height':
            self.master.img_id = int(entries[2].get())
            self.master.lines = []
            for line in entries[3].get().split(sep=' '):
                self.master.lines.append(int(line))

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

    def close(self):
        self.destroy()
