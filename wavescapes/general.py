import matplotlib.pyplot as plt

from .pcv import produce_pitch_class_matrix_from_filename
from .dft import apply_dft_to_pitch_class_matrix
from .color import complex_utm_to_ws_utm
from .draw import Wavescape, compute_height

def generate_single_wavescape(filepath, coefficient, pixel_width, save_label=None,\
                            aw_size=1, remove_percussions=True, trim_extremities=True,\
                            dpi=96, drawing_primitive=Wavescape.RHOMBUS_STR,\
                            tick_ratio=0, start_offset=0,\
                            plot_indicators=True, add_line=False, subparts_highlighted=None):
    
    '''
    Given a fourier coefficient, generates the wavescape from 
    the file path of a musical piece in MIDI or XML format.
    
    The smaller the analysis window's size, the longer this function will
    take to produce the plot.

    Parameters
    ---------
    filepath: str
        path to the MIDI or XML file that gets visualized.
    
    pixel_width: int
        the width in pixel of the wavescape. The height is dependant on both the width and
        the drawing primitive used, and as such, cannot be decided by the user of this function
    
    coefficient: int, between 1 and 6 (included)
        Index of the Fourier coefficient that is visualised in the wavescape plot. For more details,
        see the doc of ''
        
    save_label: str, optional
        if provided, save the resulting plot in the label indicated. Internally will call matplotlib.pyplot.savefig
        with this parameter, so the file format can (and must) be specified as an extension in this parameter
        Default value is None
    
    aw_size: int, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is 1
        
    remove_percussions: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    trim_extremities: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    dpi: int, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is 96 (most common dpi values for computers' screen)
    
    drawing_primitive: str, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is Wavescape.RHOMBUS_STR (i.e. 'rhombus')
    
    tick_ratio: int, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 1
    
    start_offset: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 0
    
    plot_indicators: boolean, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is True
    
    add_line: boolean, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is False
    
    '''
    
    pc_mat = produce_pitch_class_matrix_from_filename(filepath, aw_size=aw_size, trim_extremities=trim_extremities, remove_percussions=remove_percussions)
    fourier_mat = apply_dft_to_pitch_class_matrix(pc_mat, build_utm=True)
    color_mat = complex_utm_to_ws_utm(fourier_mat, coeff=coefficient)
    ws = Wavescape(color_mat, pixel_width=pixel_width, drawing_primitive=drawing_primitive, subparts_highlighted=subparts_highlighted)
    ws.draw(plot_indicators=plot_indicators, tick_ratio=tick_ratio, start_offset=start_offset,add_line=add_line)
    if save_label:
        plt.savefig(save_label)



def generate_all_wavescapes(filepath,individual_width, save_label=None,\
                            aw_size=1, remove_percussions=True, trim_extremities=True,\
                            dpi=96, drawing_primitive=Wavescape.RHOMBUS_STR,\
                            tick_ratio=0, start_offset=0,\
                            plot_indicators=True, add_line=False):

    '''
    Generates the wavescapes for all six unique Fourier coefficients given
    the path of a musical piece in MIDI or WAV format. 
    Can output all 6 coefficients in a single figure, or output and save
    each coefficient separately.
    
    For small analysis window's size, this function will take some time to
    render all the individual figures.

    Parameters
    ---------
    filepath: str
        path to the MIDI or WAV file that gets visualized.
    
    individual_width: int
        the width in pixel of each individual wavescapes. If no save label is provided,
        then the resulting plot holds all 6 plots and consequently has 3*individual_width
        as width, and a height of two individual wavescapes (the hieght of a wavescape is
        dependent on the width and drawing primitive used)
    
    save_label: str, optional
        The prefix of the filepath to save each individual plot. If it has the (default)
        value of `None`, then the function produces all six plots into a single 3 by 2 figure
        and don't save it in PNG format (but this can be easily achieved by calling the "saveFig" 
        function of matplotlib.pyplot after this one)
        The path can be absolute or relative, however, it should not hold any file extensions at the end, 
        as it is generated by this function.
        For example, if the value "bach" is given for this parameter, then the following files will be created:
        bach1.png, bach2.png, bach3.png, bach4.png, bach5.png and bach6.png 
        Each number preceding the PNG extension indicates which coefficient is vizualized in the file.
        Default value is None.
    
    aw_size: int, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is 1
        
    remove_percussions: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    trim_extremities: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    dpi: int, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is 96 (most common dpi values for computers' screen)
    
    drawing_primitive: str, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is Wavescape.RHOMBUS_STR (i.e. 'rhombus')
    
    tick_ratio: int, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 1
    
    start_offset: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 0
    
    plot_indicators: boolean, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is True
    
    add_line: boolean, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is False
    
    '''

    pc_mat = produce_pitch_class_matrix_from_filename(filepath, aw_size=aw_size)
    fourier_mat = apply_dft_to_pitch_class_matrix(pc_mat)
    
    total_width = (3.1*individual_width)/dpi
    total_height = (2.1*compute_height(individual_width, fourier_mat.shape[0], drawing_primitive))/dpi
    if not save_label:
        fig = plt.figure(figsize=(total_width, total_height), dpi=dpi)
    
    for i in range(1, 7):
        color_utm = complex_utm_to_ws_utm(fourier_mat, coeff=i)
        w = Wavescape(color_utm, pixel_width=individual_width, drawing_primitive=drawing_primitive)
        if save_label:
            w.draw(plot_indicators=plot_indicators, add_line=add_line,\
               tick_ratio=tick_ratio, start_offset=start_offset)
            plt.tight_layout()
            plt.savefig(save_label+str(i)+'.png', bbox_inches='tight')
        else:
            ax = fig.add_subplot(2, 3, i)
            w.draw(ax=ax, plot_indicators=plot_indicators, add_line=add_line,\
               tick_ratio=tick_ratio,start_offset=start_offset,  label=coeff_nbr_to_label(i)+' coeff.')
            
    plt.tight_layout()