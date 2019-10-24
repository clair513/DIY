# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:50:06 2019

@author: alokj
"""

import numpy as np
import pandas as pd
from tkinter import *
from tkinter.ttk import *

root = Tk()

# Sample DataFrame:
t = np.arange(0.0,3.0,0.01)
data = pd.DataFrame({'Value':t, 'Sin Value':np.sin(2*np.pi*t)})
data['Choices'] = np.random.choice(['Alpha','Gamma','Beta','Delta'], len(data))


# To visualize input DataFrame:
def display_plot(gui_root=root, df=data, x_axis='t', y_axis='s',
                 plot={'type':None, 'hue':None},
                 aesthetics={'style':'whitegrid', 'palette':'hsv',
                             'size':(12,7), 'dpi':100}):
    """
    DESCRIPTION:
    """

    # Importing dependencies:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import seaborn as sns
    sns.set(style=aesthetics['style'], palette=aesthetics['palette'])

    # Setting up Tkinter Frame:
    lf = Labelframe(gui_root,
                    text = plot['type'].capitalize() + ' of ' + x_axis + ' v/s ' + y_axis)
    lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)

    # Setting up Canvas backed by Matplotlib:
    fig = Figure(figsize=aesthetics['size'], dpi=aesthetics['dpi'])
    ax = fig.add_subplot(111)

    # Drawing various plots with Seaborn (w/o validation):
    if plot['type']=='lineplot':  # Lineplot
        g = sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)
        g.set_ylabels("Sinular Wave")

    elif plot['type']=='barplot': # Grouped Barplot
        g = sns.catplot(x=y_axis, y=x_axis, hue=plot['hue'], data=df,
                        kind="bar", ax=ax)
        g.despine(left=True)

    else:
        # More to be added later
        pass

    # Displaying plot on Canvas:
    canvas = FigureCanvasTkAgg(fig, master=lf)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)

display_plot()

root.mainloop()
