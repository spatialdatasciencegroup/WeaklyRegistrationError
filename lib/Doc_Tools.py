import os, sys, csv

# Math
import numpy as np
import matplotlib.pyplot as plt

# Dates and Time
from datetime import datetime as dt
from datetime import timedelta
import pytz

from lib.envtools import gettime
"""
Module for documenting tests in the GeoErrors Project.

Functional successor to ModuleTools.
"""

# Set Timezone
tz = pytz.timezone("US/Central")

# Default Plot configurations
LOSS_PLOT = [
    { # Training Loss Plot info
        'name': 'Training Loss',
        'color_char': 'b'
    },
    { # Validation Loss Plot info
        'name': 'Validation Loss',
        'color_char': 'g'
    }
]
F1_PLOT = [
    { # Training F1 Plot info
        'name': 'Training F1',
        'color_char': 'b'
    },
    { # Validation F1 Plot info
        'name': 'Validation F1',
        'color_char': 'g'
    }
]
LR_PLOT = [{ # Learning Rate Plot
        'name': 'Learning Rate',
        'color_char': 'm',
        'line_char': ':'
        }]


def mkdir_safe(path: str):
    """ Make dir with duplicate awareness. """
    try:
        os.mkdir(path)
        return path
    except FileExistsError:
        i = 1
        while True:
            indexed_path = path + ' ({})'.format(i)
            if not os.path.exists(indexed_path):
                os.mkdir(indexed_path)
                return indexed_path
            else:
                i += 1
    

def InitTest(root_dir, test_name: str='em_results'):
    """ Initializes test folder for saving config and parameters. 
    
    Args:
        root_dir: parent folder that holds testing data
        config: dict holding info on EM iteration hyperparameters, etc.
    Returns:
        (int) test_idx: index of EM test.
        (str) test_dir: directory for test output
    """    
    if not os.path.exists(root_dir):
        print(f"Initializing root test dir: {root_dir}")
        os.mkdir(root_dir)
    test_idx = len(os.listdir(root_dir))       
    
    # Format test name
    if test_name[-1] != '_':
        test_name = f"{test_name}_"
    test_name = f"{test_name}{test_idx:02}"
    
    # make directory for test output
    test_fp = os.path.join(root_dir, test_name)
    if os.path.exists(test_fp):
        sel = input(f"Output folder already exists: '{test_fp}'  Overwrite? [y]/n")
        if ('y' in sel) or (sel.replace(' ', '') == ''):
            os.rmdir(test_fp)
        else:
            print("Exiting.")
            sys.exit(0) #! This is not the best way to exit
    os.mkdir(test_fp)
    return test_idx, test_fp


def plot_axis(ax, data, name, color_char='r', symbol_char='o', y_off=1., x_label=None, label_delta=False, label_base=True):
    """ Creates a pyplot chart of data by em step for different keys in the passed dict. 
    
    Args:
        ax (plt.axis): Axis to plot data over.
        data (list): List of values to plot.
        name (str): Name of data to be plotted.
        color_char (str[1] ['r']): character defining color of plot. Defaults to black.
        symbol_char (str[1] ['o']): character defining symbol of plot. Defaults to circle.
        y_off (float): Verticle offset for datapoint labeling
        x_label (str): optinal string to label x_axis of plot.
        label_delta (bool [True]): Optionally label delta from baseline at each point
    
    Returns:
        plt.figure: Pyplot figure of all plots.
    """

    if not (isinstance(data, list) or isinstance(data, np.ndarray)):
        raise RuntimeError("doc_tools.plot_axis: Must pass data for plot as 'list' or 'np.ndarray'. Passed: '{}'".format(type(data).__name__))

    fmt_str = symbol_char + color_char + '-'

    # list of steps from 1 to len(data)
    i = list(range(1, len(data)+1)) 

    if label_base:
        # Plot Source Horizontal and data line
        source_line, plot_line, = ax.plot(i, [data[0]]*len(data), 'k:', i, data, fmt_str)
        plot_line.set_label("Step {}".format(name))
        source_line.set_label("Base {}".format(name))

        # Annotate Baseline point 
        ax.annotate('{:.2f} (base)'.format(data[0]), (0.75, data[0]+0.75))
    else:
        # Plot Source Horizontal and data line
        plot_line, = ax.plot(i, data, fmt_str)
        plot_line.set_label("{}".format(name))


    # Annotate points 
    for idx in range(1, len(data)):

        # Adjust label offset based on pos
        if data[idx] <= data[0]:
            l_off = -y_off
        else:
            l_off = y_off
        
        if label_delta:
            ax.annotate('{:.2f} ({:+.2f})'.format(data[idx], (data[idx] - data[0])), 
                            xy=(idx+0.75, data[idx]+l_off),
                            c=color_char)
        else:
            ax.annotate('{:.2f}'.format(data[idx]), 
                            xy=(idx+0.75, data[idx]+l_off),
                            c=color_char)
    
    if label_delta:
        ax.set_title("{} (source delta)".format(name))
    else:
        ax.set_title("{}".format(name))
    
    
    if x_label:
        ax.set(ylabel=name, xlabel=x_label)
    else:
        ax.set(ylabel=name)

    ax.legend(loc=3)
    ax.margins(0.1, 0.3)

    return

def plot_multi_axis(ax, plot_info, plot_title=None, y_label=None, x_label=None, y_off=1., x_margin=0.1, y_margin=0.3):
    """ 
    Takes a pyplot axis and dict of plot data to create a plot with multiple sources.
    
    Notes:
        Designed for usage with keras history plots, hence the name epochs.
    
    Args:
        ax (plt.axis): Pyplot axis to draw data over.
        plot_info (dict): list of dicts, each corresponds to one line to be drawn.
            - 'data' (list):       list of data points to plot over axis
            - 'name' (str):        string used to label y-axis and legend for this series
            - 'color_char' (str):  Optional character used to set plot color
            - 'symbol_char' (str): Optional character used to set line marker. If not passed, line is not marked
            - 'line_char' (str):   Optional character used to set line style.
            - 'annotate' (bool):   Option to annotate points with their data.
        plot_title (str): Title of plot (Also used as y_label)
        x_label (str):    Optional label for x-axis.
        y_off (float):    Verticle offset for annotations
        x_margin (float):  Horizontal plot margin (default: 0.1) 
        y_margin (float):  Verticle plot margin (default: 0.3)
    Returns:
        plt.axis: Axis with drawn data.
    """
    
    # Defualts for plot formatting
    default_dict = {
        'color_char': 'k',
        'symbol_char': '',
        'line_char': '-',
        'annotate': False,
        'label_final': True
    }
    
    # Index 'epochs' to label plot values
    epochs = list(range(1, len(plot_info[0]['data'])+1))
    
    # Plot each line in the passed data
    for line_idx, line_data in enumerate(plot_info):
        
        # Preparation
        ## Check for missing critical keys
        if ('data' not in line_data.keys()):
            raise RuntimeError("doc.plot_multi_axis: Plot info dict ({}) Missing data series for plot.".format(line_idx))
        if ('name' not in line_data.keys()):
            raise RuntimeError("doc.plot_multi_axis: Plot info dict ({}) Missing data series for plot.".format(line_idx))

        ## Fill missing style keys with default values
        for default_key, default_item in default_dict.items():
            if default_key not in line_data:
                line_data.update({default_key: default_item})
        
        ## Create format/style string from dict
        fmt_str = line_data['symbol_char'] + line_data['color_char'] + line_data['line_char']
        
        
        # Plot and label line
        plot_line, = ax.plot(epochs, line_data['data'], fmt_str)
        if line_data['label_final']:
            plot_line.set_label("{} ({:.2f})".format(line_data['name'], line_data['data'][-1]))
        else:
            plot_line.set_label(line_data['name'])           
            
            
        # Conditionally Annotate points 
        if line_data['annotate']:
            
            for idx in epochs:
                # Adjust label offset based on pos
                if line_data['data'][idx] <= line_data['data'][0]:
                    l_off = -y_off
                else:
                    l_off = y_off

                # Add annotation
                ax.annotate('{:.2f}'.format(line_data['data'][idx-1]), 
                            xy = ( (idx+0.75), (line_data['data'][idx-1]+l_off) ),
                            c = line_data['color_char'])


    # Label plot and y-axis if passed a title.
    if plot_title:
        ax.set_title(plot_title)
        
        # If no y label passed, use plot title
        if y_label:
            ax.set(ylabel=y_label)
        else:
            ax.set(ylabel=plot_title)


    # Add x axis label
    if x_label:
        ax.set(xlabel=x_label)

    # Lastly, add legend and margins
    ax.legend(loc=3)
    ax.margins(x_margin, y_margin)
    
    return ax

def print_report(model_report: dict, spaces: int = 4):
    """ Prints a model report dict from ktools.ModelReport() """
    KEY_BLACKLIST = ['False_Positives', 'False_Negatives', 'Precision', 'Recall']
    for key, item in model_report.items():
        if key not in KEY_BLACKLIST:
            print(f"{(' '*spaces)}- {key}: {np.round((item*100), 3)}")

def plot_history(training_history, loss_plot=LOSS_PLOT, f1_plot=F1_PLOT, lr_plot=LR_PLOT, test_dir=None, config_idx=0, write_csv=False):
    """ Plots history from keras.history. Used for MassTesting currently. 
    
    Notes:
        Wraps plot_multi_axis, Used for MassTesting currently.
        
    Args:
        training_history (keras.history): Raw keras history from model.fit().
        loss_plot, f1_plot, lr_plot (list(dict)): plot info dict defined in notebook.
        test_dir (str): directory to save plot.
        config_idx (int): int for labeling plot.
        
    Returns:
        None
    """
    
    ## Format idx tag
    if isinstance(config_idx, int):
        config_idx = '{:02}'.format(config_idx)
    
    ## Update Data for Keras History Plots
    ### Update training loss data
    train_loss = np.array(training_history.history['loss'])
    loss_plot[0].update({'data': train_loss})
    ### Update validation loss data
    val_loss = np.array(training_history.history['val_loss'])
    loss_plot[1].update({'data': val_loss})

    ### Update training F1 Score data
    train_f1 = np.array(training_history.history['f1_score']) * 100
    f1_plot[0].update({'data': train_f1})
    ### Update validation F1 Score data
    val_f1 = np.array(training_history.history['val_f1_score']) * 100
    f1_plot[1].update({'data': val_f1})

    ### Update Learning Rate data
    lr_hist = np.array(training_history.history['lr'])
    lr_plot[0].update({'data': lr_hist})

    ## Create Figure for history plots
    hist_fig, (f1_ax, loss_ax, lr_ax) = plt.subplots(3, sharex=True, figsize=(6, 10))

    ### Plot F1 Score History
    plot_multi_axis(ax=f1_ax, 
                    plot_info=f1_plot,
                    plot_title='F1 Score', 
                    y_off=2,
                    x_margin=0.02,
                    y_margin=0.02)

    ### Plot Loss History
    plot_multi_axis(ax=loss_ax, 
                    plot_info=loss_plot,
                    plot_title='Loss', 
                    y_off=2,
                    x_margin=0.02,
                    y_margin=0.02)

    ### Plot Learning Rate History
    plot_multi_axis(ax=lr_ax, 
                    plot_info=lr_plot,
                    plot_title='Learning Rate', 
                    x_label='Epochs',
                    y_off=2,
                    x_margin=0.02,
                    y_margin=0.02)

    ### Title and save figure
    hist_fig.suptitle(f"History {config_idx}", x=0.2, y=1, fontsize='x-large', fontweight='bold')
    hist_fig.tight_layout()
    hist_fig_path = os.path.join(test_dir, f'training_history_{config_idx}.png')
    hist_fig.savefig(hist_fig_path)
    hist_fig.show()
    
    if write_csv:
        # Save to CSV file 
        csv_fp = os.path.join(test_dir, f'training_history_{config_idx}.csv')
        with open(csv_fp, 'w') as csvfile:
            hist_csv = csv.writer(csvfile, delimiter=',')
            hist_csv.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train F1', 'Val F1', 'Learning_Rate'])
            for idx, (t_l, v_l, t_f1, v_f1, lr) in enumerate(zip(train_loss, val_loss, train_f1, val_f1, lr_hist)):
                hist_csv.writerow([idx, t_l, v_l, t_f1, v_f1, lr])    
    return


def get_model_dict(em_dict: dict): 
    """ Reformats model-related contents of em_dict for plotting """
    model_dict = {'Test_Data': {}, 'Train_Data': {}, 'Val_Data': {}}
    for em_key in model_dict.keys():
        for report in em_dict[em_key]:
            for rpt_key, rpt_value in [(key, item) for key, item in report.items()]:
                if rpt_key not in model_dict[em_key].keys():
                    model_dict[em_key].update({rpt_key: np.array([rpt_value])})
                else:
                    model_dict[em_key][rpt_key] = np.append(model_dict[em_key][rpt_key], report[rpt_key])
    return model_dict


def print_s(idx: int=None, *args):
    """ Prints args with em step and time """
    TIME_FORMAT = '%b %d | %I:%M:%S%p'
    ALT_FORMAT = '%a at %I:%M:%S%p'
    
    print(*args)
    if idx or (idx==0):
        print(f"[{gettime(TIME_FORMAT)}] (Step {idx:02})\n")
    else:
        print(f"[{gettime(TIME_FORMAT)}]\n")