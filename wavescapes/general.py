import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np

from .pcv import produce_pitch_class_matrix_from_filename
from .dft import apply_dft_to_pitch_class_matrix
from .color import complex_utm_to_ws_utm, circular_hue
from .draw import Wavescape, compute_plot_height, rgb_to_hex, coeff_nbr_to_label


def single_wavescape(filepath, width, coefficient, aw_size=1, save_label=None,
                     remove_unpitched_tracks=False, deep_chroma=False, trim_extremities=True,
                     magn_stra='0c', output_rgba=False, primitive=Wavescape.RHOMBUS_STR,
                     aw_per_tick=None, tick_offset=0, tick_start=0, tick_factor=1, ignore_magnitude=False, ignore_phase=False,
                     indicator_size=None, add_line=False, subparts_highlighted=None, label=None, label_size=None,
                     ax=None):
    """
    Given a fourier coefficient, generates the wavescape from 
    the file path of a musical piece in MIDI or XML format.
    
    The smaller the analysis window's size, the longer this function will
    take to produce the plot.

    Parameters
    ---------
    filepath: str
        path to the MIDI or XML file that gets visualized.
    
    width: int
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
        
    remove_unpitched_tracks: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is False
        
    deep_chroma: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is False
        
    trim_extremities: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    magn_stra: str, optional
        see the doc 'complex_utm_to_ws_utm' for information on this parameter
        Default value is '0c'
    
    output_rgba: 
        see the doc 'complex_utm_to_ws_utm' for information on this parameter
        Default value is '0c'
    
    primitive: str, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is Wavescape.RHOMBUS_STR (i.e. 'rhombus')
    
    aw_per_tick: int, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (meaning no ticks are drawn)
    
    tick_offset: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 0

    tick_start: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 0
        
    tick_factor: float, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 1.0
    
    indicator_size: float, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None
    
    add_line: numeric value, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is False
        
    subparts_highlighted: array of tuples, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None
        
    label: str, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None
    
    label_size: float, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (in which case the default 
        size of the labels is the width of the plot divided by 30)

    ax: matplotlib figure, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None.
    
    """
    pc_mat = produce_pitch_class_matrix_from_filename(filepath, aw_size=aw_size, deep_chroma=deep_chroma,
                                                      trim_extremities=trim_extremities,
                                                      remove_unpitched_tracks=remove_unpitched_tracks)
    fourier_mat = apply_dft_to_pitch_class_matrix(pc_mat, build_utm=True)
    color_mat = complex_utm_to_ws_utm(fourier_mat, coeff=coefficient, magn_stra=magn_stra, output_rgba=output_rgba,
                                      ignore_magnitude=ignore_magnitude, ignore_phase=ignore_phase)
    ws = Wavescape(color_mat, width=width, primitive=primitive)
    ws.draw(indicator_size=indicator_size, aw_per_tick=aw_per_tick, tick_start=tick_start, tick_offset=tick_offset, tick_factor=tick_factor,
            add_line=add_line, subparts_highlighted=subparts_highlighted, label=label, label_size=label_size, ax=ax)
    if save_label:
        plt.savefig(save_label, transparent=output_rgba)


# generate all plots in one image
def all_wavescapes(filepath,individual_width, save_label=None,
                   aw_size=1, remove_unpitched_tracks=False, deep_chroma=False, trim_extremities=True,
                   magn_stra = '0c', output_rgba = False, primitive=Wavescape.RHOMBUS_STR,
                   aw_per_tick=None, tick_offset=0, tick_start=0, tick_factor=1.,ignore_magnitude=False, ignore_phase=False,
                   indicator_size=None, add_line=False, subparts_highlighted = None, label_size=None):

    """
    Generates the wavescapes for all six unique Fourier coefficients given
    the path of a musical piece in the format supported. 
    Can output all 6 coefficients in a single figure, or output and save
    each coefficient separately.
    
    For small analysis window's size (aw_size argument), this function will take some time to
    render all the individual figures, so be patient, nothing crashed ;) 

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
        
    remove_unpitched_tracks: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is False
        
    deep_chroma: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is False
        
    trim_extremities: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    magn_stra: str, optional
        see the doc 'complex_utm_to_ws_utm' for information on this parameter
        Default value is '0c'
    
    output_rgba: 
        see the doc 'complex_utm_to_ws_utm' for information on this parameter
        Default value is '0c'
    
    primitive: str, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is Wavescape.RHOMBUS_STR (i.e. 'rhombus')
    
    aw_per_tick: int, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (meaning no ticks are drawn)
    
    tick_offset: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (meaning ticks numbers start at 0)

    tick_start: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 0
        
    tick_factor: float, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 1.0
    
    indicator_size: boolean, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is True
    
    add_line: numeric value, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is False
        
    subparts_highlighted: array of tuples, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None
    
    label_size: float, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (in which case the default 
        size of the labels is the width of one individual plot divided by 30)
    
    """
    pc_mat = produce_pitch_class_matrix_from_filename(filepath, aw_size=aw_size,
                                                      remove_unpitched_tracks=remove_unpitched_tracks,
                                                      trim_extremities=trim_extremities, deep_chroma=deep_chroma)
    fourier_mat = apply_dft_to_pitch_class_matrix(pc_mat)

    dpi = 96  # (most common dpi values for computers' screen)
    total_width = (3.1*individual_width)/dpi
    total_height = (2.1*compute_plot_height(individual_width, fourier_mat.shape[0], primitive))/dpi
    if not save_label:
        fig = plt.figure(figsize=(total_width, total_height), dpi=dpi)
        
    for i in range(1, 7):
        color_utm = complex_utm_to_ws_utm(fourier_mat, coeff=i, magn_stra=magn_stra, output_rgba=output_rgba,
                                          ignore_magnitude=ignore_magnitude, ignore_phase=ignore_phase)
        w = Wavescape(color_utm, width=individual_width, primitive=primitive)
        if save_label:
            w.draw(indicator_size=indicator_size, add_line=add_line,
                   aw_per_tick=aw_per_tick, tick_offset=tick_offset, tick_start=tick_start, tick_factor=tick_factor,
                   subparts_highlighted=subparts_highlighted, label_size=label_size)
            plt.tight_layout()
            plt.savefig(save_label+str(i)+'.png', bbox_inches='tight', transparent=output_rgba)
        else:
            ax = fig.add_subplot(2, 3, i, aspect='equal')  # TODO: what if fig was not initialised above?
            w.draw(ax=ax, indicator_size=indicator_size, add_line=add_line,
                   aw_per_tick=aw_per_tick, tick_offset=tick_offset, tick_start=tick_start, tick_factor=tick_factor,
                   label=coeff_nbr_to_label(i) + ' coeff.', subparts_highlighted=subparts_highlighted,
                   label_size=label_size)
            
    plt.tight_layout()


def legend_decomposition(pcv_dict, width = 13, single_img_coeff = None, ignore_magnitude=False, ignore_phase=False):
    """
    Draw the circle color space defined by the color mapping used in wavescapes.
    Given a dict of labels/pitch-class vector, and list of coefficient to visualize,
    this function will plot the position of each of the PCV on the coefficient selected.
    This function helps visualising which color of the wavescape correspond to which musical 
    structure with respect to the coefficient number.
    
    Parameters
    ----------
    
    pcv_dict: dict, type of key is str, type of value is an array of array dimension (2,N) (0<= N, <=12)
        defines the label and pitch-class vector to be drawn, as well as the list of coefficients on which
        the pitch-class vector position needs to be drawn. For example, consider this dict is given to the
        function:
        {'CMaj':([1,0,1,0,1,1,0,1,0,1,0,1], [5]),
         'D#aug': ([0,0,0,2,0,0,0,1,0,0,0,5], [3,4]),
         'E': ([0,0,0,0,6,0,0,0,0,0,0,0], [0])}
         The position of the C Major diatonic scale will be drawn on the color space of the fifth coefficient,
         while the position of the D augmented triad will be drawn on the color space of both the third and
         sixth coefficient. Finally, the value 0 associated to the single pitch PCV 'E' indicates its position
         will be drawn on all of the 6 coefficients. The label support LateX math mode.
    
    width: int, optional
        plot's width in inches.
        Default value is 13.
        
    single_img_coeff: int, optional
        Indicates which coefficient's color space will be drawn. If no number or "None" is provided for the value
        of this parameter, then the resulting plots will feature all 6 color space, one per coefficient. The coefficient
        number contained in the dict 'pcv_dict' still apply if a single coefficient is selected with this parameter.
        Default value is None.

    ignore_magnitude: bool, optional
        Indicates whether the resulting plot needs to display the magnitude as opacity mapping from the center to
        the outward part of the circle. If only the phase, and thus the hue, of a certain musical structure
        is needed to be seen, this parameter needs to be set to False. 
        Default value is False (meaning the resulting plot displays opacity values)
        
    """
    phivals = np.arange(0, 2*np.pi, 0.01)
    mu_step = .025
    muvals = np.arange(0, 1. + mu_step, mu_step)
    
    #powerset of all phis and mus.
    cartesian_polar = np.array(np.meshgrid(phivals, muvals)).T.reshape(-1, 2)
    
    #generating the color corresponding to each point.
    color_arr = []
    for phi, mu in cartesian_polar:
        if ignore_phase:
            if ignore_magnitude:
                hexa = '#ffffff'
            else:
                stand = lambda v: int(0xff * (1-v))
                g = stand(mu)
                hexa = rgb_to_hex([g,g,g])
        else:
            hexa = rgb_to_hex(circular_hue(phi, magnitude=mu, output_rgba=ignore_magnitude))
            
        color_arr.append(hexa)
        
    xvals = cartesian_polar[:,0]
    yvals = cartesian_polar[:,1]

    norm = mpl.colors.Normalize(0.0, 2*np.pi)
    fig = plt.figure(figsize= (width,width) if single_img_coeff else (width, 8*width/5) )
    
    def single_circle(ax, i, pcv_dict, marker_width, display_title=True):
        label_size = (marker_width/10.)
        ax.scatter(xvals, yvals, c=color_arr, s=marker_width, norm=norm, linewidths=1, marker='.')
        if display_title:
            ax.set_title(coeff_nbr_to_label(i)+' coefficient', fontdict={'fontsize': label_size+6}, y=1.08)
        for k,v in pcv_dict.items():
            for coeff in v[1]:
                if coeff == i or coeff == 0:
                    comp = np.fft.fft(v[0])
                    angle = np.angle(comp[i])
                    magn = np.abs(comp[i])/np.abs(comp[0])
                    ax.scatter(angle, magn, s=marker_width, facecolors='none', edgecolors='#777777')
                    pos_magn = np.abs(magn-0.125)
                    ax.annotate(k, (angle, pos_magn), size=(marker_width/10.)+2, annotation_clip=False, horizontalalignment='center', verticalalignment='center')
        
        ax.tick_params(axis='both', which='major', labelsize=(marker_width/10.)+6)
        ax.tick_params(axis='both', which='minor', labelsize=(marker_width/10.)+4)
        ax.set_xticklabels(['$0$', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$', ''])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.xaxis.grid(False)
    
    if single_img_coeff:
        ax = plt.subplot(1, 1, 1, polar=True)
        single_circle(ax=ax, i=single_img_coeff, pcv_dict=pcv_dict, marker_width=20*width, display_title=False)
    else:
        for i in range(1, 7):
            ax = fig.add_subplot(3, 2, i, polar=True)
            single_circle(ax=ax, i=i, pcv_dict= pcv_dict, marker_width=10*width)
        plt.tight_layout() #needs to be before subplot_adjust, otherwise subplot_adjust is useless.
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.3)
