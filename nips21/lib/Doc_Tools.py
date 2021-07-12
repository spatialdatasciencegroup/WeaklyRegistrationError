import os, sys, csv, time, math

# Math
import numpy as np
import matplotlib.pyplot as plt

# Dates and Time
from datetime import datetime as dt
from datetime import timedelta
import pytz

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


def mkdir_safe(path):
    """ Make dir with duplicate awareness. """
    try:
        os.mkdir(path)
        return path
    except FileExistsError:
        i = 1
        while True:
            if not os.path.exists(path + '({})'.format(i)):
                os.mkdir(path + '({})'.format(i))
                return path + '({})'.format(i)
            else:
                i += 1
    

def InitTest(root_dir, prompt=False, **kwargs):
    """ Initializes test folder for saving config and parameters. 
    
    Args:
        root_dir: parent folder that holds testing data
        config: dict holding info on EM iteration hyperparameters, etc.
        prompt: optionally prompts user for test notes
            False: Silent, no prompts will be displayed
            True:  Prompts test notes
        Pass test parameters as kwargs

    Returns:
        (int) test_idx: index of EM test.
    """
    
    root_name = os.path.split(root_dir)[1]
    csv_fp = os.path.join(root_dir, '{}.csv'.format(root_name))
    
    if not os.path.exists(root_dir):
        print("Initializing root test dir: {}".format(root_dir))
        os.mkdir(root_dir)

        with open(csv_fp, 'a+', newline='\n') as csvfile:
            main_csv = csv.writer(csvfile, delimiter=',')
            header = [root_name, 'date']
            header.extend([key for key in kwargs.keys()])
            main_csv.writerow(header)


    test_idx = len(os.listdir(root_dir))        
    
    with open(csv_fp, 'a+', newline='\n') as csvfile:
            main_csv = csv.writer(csvfile, delimiter=',')
            row = ['Test_{:02}'.format(test_idx), dt.now().strftime('%m/%d/%Y')]
            row.extend([item for _, item in kwargs.items()])
            main_csv.writerow(row)

    if 'test_name' in kwargs.keys():
        test_name = kwargs['test_name']
    else:
        test_name = 'Test_'

    test_fp = '{}/{}{:02}'.format(root_dir, test_name, test_idx)
    
    if os.path.exists(test_fp):
        raise RuntimeError("Test folder already exists: '{}'".format(test_fp))
    else: 
        os.mkdir(test_fp)

    return test_idx, test_fp


def plot_axis(ax, data, name, color_char='r', symbol_char='o', y_off=1., x_label=None, label_delta=True, label_base=True):
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
        label_delta (bool [True]): Optionally label baseline point and line
    
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


def plot_history(training_history, loss_plot=LOSS_PLOT, f1_plot=F1_PLOT, lr_plot=LR_PLOT, test_dir=None, config_idx=0):
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
    hist_fig.suptitle("History {}".format(config_idx), x=0.2, y=1, fontsize='x-large', fontweight='bold')
    hist_fig.tight_layout()
    hist_fig_path = os.path.join(test_dir, 'history_plot_{}.png'.format(config_idx))
    hist_fig.savefig(hist_fig_path)
    hist_fig.show()
    
    ### Save to CSV file 
    csv_fp = os.path.join(test_dir, 'history_{}.csv'.format(config_idx))
    with open(csv_fp, 'w') as csvfile:
        hist_csv = csv.writer(csvfile, delimiter=',')
        hist_csv.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train F1', 'Val F1', 'Learning_Rate'])
        for idx, (t_l, v_l, t_f1, v_f1, lr) in enumerate(zip(train_loss, val_loss, train_f1, val_f1, lr_hist)):
            hist_csv.writerow([idx, t_l, v_l, t_f1, v_f1, lr])    
    return

# Mass testing tools

def init_mass_baseline(root_dir):
    """ Create folder for mass baseline test.
    
    Args:
        root_dir (str): Root test folder to hold all mass tests.
        
    Returns:
        int (test_idx),    Index of this test 
        str (test_dir)     Directory for this test
    """
    
    # Validate root folder and get test index
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        test_idx = 0
    else:
        test_idx = len(os.listdir(root_dir))
    
    # Create test folder
    test_dir = os.path.join(root_dir, 'Mass_Baseline_{:02}'.format(test_idx))
    
    return test_idx, mkdir_safe(test_dir)
    

def parameter_permutations(parameters: dict, expected_step_sec: int):
    """ 
    Prepare all permutations of the parameters and return list of dicts for each permutation.
    
    Also, print the permutations and estimated time required for test. 
    
    Args:
        parameters (dict): Dict with keyed parameters.
        expected_step_time (int): Time in seconds that each permutation will require.
    
    Returns:
        list(dict): All parameter permutation configurations as list of keyed values. 
    """

    # Get permutations of all parameters
    permutations = []
    for param_indices in np.array(np.meshgrid(*[list(range(len(item))) for key, item in parameters.items()])).T.reshape(-1,len(parameters.keys())):
        permutations.append({key: item[idx] for idx, (key, item) in zip(param_indices, parameters.items())})
  
    ## calculate expected time for test
    expected_sec = expected_step_sec * len(permutations)

    
    # Print preparation analysis
    print("CONFIGURATION DETAILS:")
    print("------------------")
    print("Total Configurations: {}".format(len(permutations)))
    for key, item in parameters.items():
        print("- {} ({}): {} ".format(key, len(item), item))
    print("------------------")
    
    ## Print estimate processing time 
    print("Estimated time to process:\n- {}h {}m {}s".format((expected_sec//3600), ((expected_sec%3600)// 60), (expected_sec%60)))
    
    ## Print estimated completion time
    finish_time = tz.localize(dt.now()) + timedelta(seconds=expected_sec)
    print("Estimated time of completion:", finish_time.strftime('\n- %a at %I:%M:%S%p'))

    return permutations



def plot_keras_history():
    """ Plots data from a keras model's history object. """