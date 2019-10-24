# Importing required packages:
import pandas as pd
from tkinter import *
from tkinter.ttk import *

root = Tk()

# Publicly available sample DataFrame:
import seaborn as sns
tips = sns.load_dataset('tips')


# To visualize input DataFrame:
def generate_plot(gui_root, df, x_axis, y_axis=None,
                 plot={'type':None, 'hue':None},
                 aesthetics={'style':'whitegrid', 'palette':'hsv',
                             'size':(10,7), 'dpi':100}):
    """
    DESCRIPTION: Reads input Pandas DataFrame and returns a plot based on selected parameters.

    PARAMETERS:
        > gui_root : [default None] Accepts Tkinter application base class (Tk) initialized variable/instance.
        > df : [default None] Accepts Pandas DataFrame.
    """


    # Importing external dependencies:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import seaborn as sns
    sns.set(style=aesthetics['style'], palette=aesthetics['palette'])
    import warnings
    warnings.filterwarnings('ignore')

    # Defining Tableau colors:
    tableau_20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199,
                  199),(188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # Scaling over RGB values to [0,1] range (Matplotlib acceptable format):
    for i in range(len(tableau_20)):
        r,g,b = tableau_20[i]
        tableau_20[i] = (r/255., g/255., b/255.)

    # Setting up Tkinter Frame:
    lf = Labelframe(gui_root)
    lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)

    # Setting up Canvas backed by Matplotlib:
    fig = Figure(figsize=aesthetics['size'], dpi=aesthetics['dpi'])
    ax = fig.add_subplot(111)

    # Drawing various plots with Seaborn:
    if plot['type']=='lineplot':  # Lineplot
        g = sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)
    elif plot['type']=='regplot':  # Regplot
        g = sns.regplot(x=x_axis, y=y_axis, data=df, color=tableau_20[16], ax=ax)
    elif plot['type']=='distplot':  # Distplot
        g = sns.distplot(a=df[x_axis].dropna(), color=tableau_20[7],
                         hist_kws=dict(edgecolor='k', linewidth=0.5), ax=ax)
    elif plot['type']=='barplot': # Grouped Barplot
        g = sns.catplot(x=x_axis, y=y_axis, hue=plot['hue'], data=df,
                        kind="bar", palette='rocket', ax=ax)
        g.despine(left=True)
    else:
        # More to be added later
        pass

    # Displaying plot on Canvas:
    canvas = FigureCanvasTkAgg(fig, master=lf)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)

generate_plot()

root.mainloop()