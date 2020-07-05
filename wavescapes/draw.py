import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.ticker import MultipleLocator, IndexLocator,FuncFormatter

SQRT_OF_THREE = math.sqrt(3)


def rgb_to_hex(rgb):
    if type(rgb) is str and rgb[0] == '#' and len(rgb) > 6:
        # we already have an hex value given let's just return it back.
        return rgb 
    elif len(rgb) == 3:
        return '#%02x%02x%02x' % (rgb[0],rgb[1],rgb[2])
    elif len(rgb) == 4:
        return '#%02x%02x%02x%02x' % (rgb[0],rgb[1],rgb[2], rgb[3])
    else:
        raise Exception('Cannot convert RGB tuple to hex value if the value given is neither in the RGB or the RGBA format.')

def coeff_nbr_to_label(k):
    if type(k) == str:
        k = int(k)
    if k == 1:
        return '%dst'%k
    elif k == 2:
        return '%dnd'%k
    elif k == 3:
        return '%drd'%k
    else:
        return '%dth'%k
    

def compute_height(width, mat_dim, drawing_primitive):
    if drawing_primitive == Wavescape.HEXAGON_STR:
        return Wavescape.HEXAGON_PLOT_HEIGHT(width, mat_dim)
    elif drawing_primitive ==  Wavescape.RHOMBUS_STR:
        return (width/2.) * SQRT_OF_THREE
    elif drawing_primitive ==  Wavescape.DIAMOND_STR:
        return width
    else:
        raise Exception('Unknown drawing primitive: %s'%drawing_primitive)

class DiamondPrimitive(object):
    def __init__(self, x, y, width, height, color, bottom_diamond):
        self.half_width = width/2.
        self.half_height = height/2.
        self.x = x
        self.y = y
        self.color = color
        self.bottom_diamond = bottom_diamond
        
    def draw(self, new_color=None, stroke=None):
        curr_color = new_color if new_color else self.color
        x = self.x
        y = self.y
        # this is to treat the bottom diamond that needs to be drawn as a triangle
        last_coord = (x,y if self.bottom_diamond else y-self.half_height)
        return Polygon(((x-self.half_width, y),
                               (x, y+self.half_height),
                               (x+self.half_width, y),
                               last_coord),
                         alpha=1,
                         facecolor = curr_color,
                         edgecolor=stroke if stroke else curr_color,
                         linewidth=self.half_width/10. if stroke else None)

class HexagonPrimitive(object):
    def __init__(self, x, y, width, color):
        self.half_width = width/2.
        self.h = SQRT_OF_THREE*self.half_width/3.
        self.x = x
        self.y = y
        self.color = color

    def draw(self, new_color=None, stroke=None):
        w = self.half_width
        h = self.h
        d_x = self.x
        d_y = self.y
        curr_color = new_color if new_color else self.color
        return Polygon(((d_x+w, d_y+h),
                        (d_x, d_y+2*h),
                        (d_x-w, d_y+h),
                        (d_x-w, d_y-h),
                        (d_x, d_y-2*h),
                        (d_x+w, d_y-h)),
                         alpha=1,
                         facecolor = curr_color,
                         edgecolor=stroke if stroke else curr_color,
                         linewidth=w/20. if stroke else None)
class Wavescape(object):
    '''
    This class represent an object that holds the attributes 
    and methods needed to effectively draw the wavescapes plot.

    Attributes
    ----------
    utm : NxNx3 or NxNx4 matrix (numpy.ndarray of numpy.uint8), 
        upper triangle matrix holding color values as tuples of 3 (RGB) or 4 (RGBA) 8 bit integers. 
        Holds the color information and their relevant informations to draw the plot.
        
    width : int
        the width in pixels of the plot. It needs to be at least twice as big as the shape of the 
        upper triangle matrix. The height of the plot is defined by the drawing primitive chosen.
        
    drawing_primitive : {'diamond', 'rhombus', 'hexagon'} , optional 
        the drawing shape that forms a single colored element from the plot. Three primitives are 
        currently available:
          -'diamond': diamond whose height is twice its width
          -'rhombus': diamond formed by two equilateral triangles. Each side is the same size
          -'hexagon': a hexagon, i.e. a 6 sides polygonal shape, each side being the same size.
        default value is 'rhombus'
        
    subparts_highlighted: array of tuples of int, optional
        list of subsection that needs to be drawn with black outlines on the wavescape. Units
        are expressed in number of analysis windows. For example, if a musical piece has a 4/4 
        time signature, an analysis window of 1 quarter note and a total of 10 bars, the
        value [[20,28],[32,36]] for 'subparts_highlighted' will draw black outlines on th region
        of the wavescape corresponding to bars 5 until 7 and 8 until 9.
    '''
    #Formula derived with a bit of algebra in order to determine the height of the wavescape hexagon plot 
    #based on the just the given plot's width (wi) and the number of layer (n). The SQRT_OF_THREE*wi was broadcasted
    #to the two parts of the addition to mitigate the numeric error caused by the division by 6 times the number
    #of layer (n).
    HEXAGON_PLOT_HEIGHT = lambda wi, n: (SQRT_OF_THREE*wi)*(0.5) + ((SQRT_OF_THREE/6.)*(wi/n))
    
    #constants 
    #fun fact that would please anyone with OCD: all drawing primitives' name have the same amount of letters.
    DIAMOND_STR = 'diamond'
    RHOMBUS_STR = 'rhombus'
    HEXAGON_STR = 'hexagon'
    
    def __init__(self, utm, pixel_width, drawing_primitive='rhombus', subparts_highlighted=None):
        self.utm = utm
        self.width = pixel_width
        self.drawing_primitive = drawing_primitive
        
        mat_dim, mat_dim_other_axis, mat_depth = utm.shape
        if mat_dim != mat_dim_other_axis:
            raise Exception("The upper triangle matrix is not a square matrix")
        if mat_dim > self.width/2:
            raise Exception("The number of elements to be drawn exceeds the wavescape's resolution.(%d elements out of %d allowed by the resolution) Increase the width of the plot to solve this issue" % (mat_dim, self.width/2))
        if (mat_depth < 3 or mat_depth > 4):
            raise Exception("The upper triangle matrix given as argument does not hold either RGB or RGBA values")
        self.mat_dim = mat_dim
        
        self.subparts = subparts_highlighted
        
        #building a matrix with None to hold the element object for drawing them later.
        self.matrix_primitive = np.full((mat_dim, mat_dim), None, object)
        
        self.height = compute_height(self.width, mat_dim, drawing_primitive)
        if drawing_primitive == self.HEXAGON_STR:
            self.generate_hexagons(subparts_highlighted)
        elif drawing_primitive == self.RHOMBUS_STR or drawing_primitive == self.DIAMOND_STR:
            self.generate_diamonds(subparts_highlighted)
        else:
            raise Exception('Unkown drawing primitive: %s'%drawing_primitive)
            
    def generate_hightlights(self, unit_width):
        '''
        Helper method, is called by the other helper functions 'generate_diamonds/hexagons'. 
        Take care of generating the drawing primitive corresponding to the 
        highlights given as arguments to the constructor of the Wavescape class. 
        '''
        triangles = []
        for tup in self.subparts:
            lo = min(tup)
            hi = max(tup)
            if lo == hi:
                raise Exception('Highlight\'s start index (%s) should not be equal to its end index'%(str(lo)))
            if lo > self.mat_dim or hi > self.mat_dim:
                raise Exception('Subpart highlights\' indices cannot be above the number of element at the base of the wavescape (%d)'%self.mat_dim)
            tri_width = (hi-lo) * unit_width
            tri_height = compute_height(tri_width, hi-lo, self.drawing_primitive)
            xl = (lo-.5)*unit_width - self.width/2.
            yb = -self.height/2.
            xr = (hi-.5)*unit_width - self.width/2.
            yt = tri_height-self.height/2. 
            xt = (lo+hi-1)/2.*unit_width - self.width/2.
            triangles.append(Polygon(((xl, yb),
                               (xt, yt),
                               (xr, yb)),
                         alpha=1,
                         facecolor = None,
                         fill = None,
                         linewidth=1))
        self.subparts = triangles
                
    
    def generate_hexagons(self, subparts_highlighted):
        '''
        Helper method, is called by the constructor of the class. 
        This method takes care of generating the Hexagon drawing primitives in case such
        drawing primitive was chosen. One matplotlib.patches.Polygon is generated per element 
        of the plot. The draw method takes care of drawing those patches on the final figure.
        '''
        hexagon_width = self.width/float(self.mat_dim)
        hexagon_height = 2*SQRT_OF_THREE*hexagon_width/3.
        half_width_shift = self.width/2.
        half_height_shift = self.height/2.
        
        for y in range(self.mat_dim):
            for x in range(y, self.mat_dim):
                curr_color = rgb_to_hex(rgba_to_rgb(self.utm[y][x], background=(0xff,0xff,0xff)))
                #Useless to draw if there is nothing but blank to draw
                if curr_color != '#FFFFFF':
                    #classic x-axis placement taking into account the half width of the hexagon
                    d_x = hexagon_width*x
                    #Now shifting all of this to the left to go from utm placement to pyramid placement
                    d_x = d_x - hexagon_width*y/2.
                    #And finally shifting this to take into account drawSvg center placement I posed
                    d_x = d_x - half_width_shift
                    
                    d_y = hexagon_height/2.+(0.75*hexagon_height)*y
                    d_y = d_y - half_height_shift
                    
                    #self.matrix_primitive[y][x] = Hexagon(d_x, d_y, hexagon_width, curr_color)
                    self.matrix_primitive[y][x] = HexagonPrimitive(d_x, d_y, hexagon_width, curr_color)
        
        if subparts_highlighted:
            self.generate_hightlights(hexagon_width)
        else:
            self.subparts = None
    
    def generate_diamonds(self, subparts_highlighted):
        '''
        Helper method, is called by the constructor of the class. 
        This method takes care of generating the Diamond drawing primitives in case such
        drawing primitive was chosen. One matplotlib.patches.Polygon is generated per element 
        of the plot. The draw method takes care of drawing those patches on the final figure.
        '''
        diamond_width = self.width/float(self.mat_dim)
        diamond_height = diamond_width*2 if self.drawing_primitive != 'rhombus' else diamond_width * SQRT_OF_THREE
        
        half_width_shift = self.width/2.
        half_height_shift = self.height/2.
        
        for y in range(self.mat_dim):
            for x in range(y, self.mat_dim):
                
                curr_color = rgb_to_hex(rgba_to_rgb(self.utm[y][x], background=(0xff,0xff,0xff)))
                #Useless to draw if there is nothing but blank to draw, duh.
                if curr_color != '#FFFFFF':
                    #classic x-axis placement taking into account the edge from the diamond 
                    d_x = diamond_width*x
                    #Now shifting all of this to the left to go from utm placement to pyramid placement
                    d_x = d_x - diamond_width*y/2.
                    #And finally shifting this to take into account drawSvg center placement I posed
                    d_x = d_x - half_width_shift
                    
                    d_y = diamond_height/2.*y
                    d_y = d_y - half_height_shift
                    self.matrix_primitive[y][x] = DiamondPrimitive(d_x, d_y, \
                                          diamond_width, diamond_height, curr_color, y == 0)
        
        if subparts_highlighted:
            self.generate_hightlights(diamond_width)
        else:
            self.subparts = None

    def draw(self, ax=None, dpi=96, plot_indicators = True, add_line = False, tick_ratio = None, start_offset=0, label=None):
        '''
        After being called on a properly initialised instance of a Wavescape object,
        this method draws the visual plot known as "wavescape" and generate a 
        matplotlib.pyplot figure of it. This means any of the method from this
        library can be used after this method has been called in order to
        save or alter the figure produced. 

        Parameters
        ----------
        ax: matplotlib figure, optional
            Default value is None.
        
        plot_indicators: bool, optional 
            indicates whether rounded indicators on the lateral edges of the plot need to 
            be drawn. A rounded indicator is drawn at each eight of the height of the plot
            Default value is True
            
        dpi: int, optional
            dot per inch (dpi) of the figure. N
            Default value is 96, which is normally the dpi on windows machine. The dpi 
            
        add_line: bool, optional
            indicates whether all element of the plot (single drawing primitives) need to be
            outlined with a black line.
            Default value is False.
            
            
        tick_ratio: int, optional
            Ratio of tick per elements of the lowest level in the wavescape. If tick_ratio has value 1,
            one horizontal axis tick will be drawn per element at the lowest hierarchical level of the wavescape, 
            if it has value 2, then a tick will be drawn each two elements. For the ticks to represent the bar numbers,
            a preexisting knowledge of the time signature of the musical piece is required. (if a piece is in 4/4,
            and the analysis window has a size of one quarter note, then tick_ratio needs to have value 4 for the
            ticks to correspond to bar numbers)
            Default value is None (meaning no ticks are drawn)

        Returns
        -------
        Nothing, but a matplotlib.pyplot figure is produced by this method, and any method of that library
        can be used to alter the resulting figure (notably matplotlib.pyplot.savefig can be used to save the
        resulting figure in an Image format)
        '''
        
        utm_w = self.matrix_primitive.shape[0]
        utm_h = self.matrix_primitive.shape[1]
        
        if self.matrix_primitive is None or utm_w < 1 or utm_h < 1:
            raise Exception("cannot draw when there is nothing to draw. Don't forget to generate diamonds in the correct mode before drawing.")
        
        if tick_ratio: 
            
            if tick_ratio < 1 or type(tick_ratio) is not int:
                raise Exception("Tick ratio must be an integer greater or equal to 1")
            
            #argument start_offseet is only meaningless if there is tick ratio involved in the plotting
            if type(start_offset) is not int or start_offset < 0 or start_offset > tick_ratio:
                raise Exception("Stat offset needs to be a positive integer that is smaller or equal to the tick ratio")
        
        height = self.height
        width = self.width

        
        black_stroke_or_none = 'black' if add_line else None
        if not ax:
            fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            ax = fig.add_subplot(111, aspect='equal')
        primitive_half_width = None
        

        for y in range(self.matrix_primitive.shape[0]):
            for x in range(y, self.matrix_primitive.shape[1]):
                element = self.matrix_primitive[y][x]
                if x == 1 and x == y:
                    primitive_half_width = element.half_width
                ax.add_patch(element.draw(stroke=black_stroke_or_none))
                             

        if plot_indicators:
            ind_width = width if self.drawing_primitive != self.HEXAGON_STR else width + 2
            mid_size = int(self.width / 40.)
            eigth_size = int(mid_size /4.)
            quart_size = eigth_size * 3

            white_fill = (1, 1, 1, 0)
            middle_gray= (.398, .398, .398, 1)

            params = [
                {'size': mid_size,   'facecolor': white_fill, 'edgecolor': 'black' },
                {'size': quart_size, 'facecolor': white_fill, 'edgecolor': middle_gray},
                {'size': eigth_size, 'facecolor': middle_gray,'edgecolor': middle_gray}
            ]

            stroke_width = int(self.width / 1000.)+1

            
            # Code to draw the indicators using circles.
            # This is probably the most far fetched discrete mathematical formula I ever made.
            # Basically I found the coordinates relative to the height and width of the plot by trial 
            # and error using negative power of 2, and then I derived a discrete formula
            # depending on two parameters n and m (the second one depending on the first)
            # which give me automatically the right x and y coordinates. It works, just trust me.
            for n in range(1,4):
                p = params[n-1]
                for m in range(2**(n-1)):
                    x = 1/float(2**(n+1)) + m/float(2**n)
                    y = (2**n - 1)/float(2**n) - m/float(2**(n-1)) - 1/2.
                    for i in [-1, 1]:
                        ax.add_patch(Circle((i*x*width-primitive_half_width, y*height), radius=p['size'], facecolor=p['facecolor'], \
                                                  edgecolor=p['edgecolor'], linewidth=stroke_width))

        plt.autoscale(enable = True)
        
        labelsize = self.width/30.
        
        if tick_ratio:
            indiv_w = self.width/utm_w
            scale_x = indiv_w * tick_ratio
            ticks_x = FuncFormatter(lambda x, pos: '{0:g}'.format(math.ceil((x+ self.width/2.)/scale_x) + (1 if start_offset == 0 else 0)))
            
            ax.tick_params(which='major', length=self.width/50., labelsize=labelsize)
            ax.tick_params(which='minor', length=self.width/100.)
            
            ax.xaxis.set_major_formatter(ticks_x)
            number_of_ticks = self.width/scale_x
            eight_major_tick_base = scale_x*round(number_of_ticks/8.)
            ax.xaxis.set_major_locator(IndexLocator(base=eight_major_tick_base, offset=start_offset*indiv_w))
            
            #display minor indicators
            ax.xaxis.set_minor_locator(IndexLocator(base=scale_x, offset=start_offset*indiv_w))
            
            #make all the other border invisible
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.yticks([])
        else:
            plt.axis('off')
            
        if self.subparts:
            for pat in self.subparts:
                ax.add_patch(pat)

        #remove top and bottom margins 
        if label:
            x_pos = -self.width/2. + self.width/10.
            y_pos = self.height/2. - self.width/10.
            ax.annotate(label, (x_pos, y_pos), size=labelsize, annotation_clip=False, horizontalalignment='left', verticalalignment='top')
        ax.set_ylim(bottom=-self.height/2., top=self.height/2.)
        ax.set_xlim(left=-self.width/2.-primitive_half_width, right=self.width/2.-primitive_half_width)
        plt.tight_layout()

def legend_decomposition(pcv_dict, width = 13, single_img_coeff = None):
    '''
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
        {'CMaj':[[1,0,1,0,1,1,0,1,0,1,0,1], [5]],
         'Daug':[[0,0,1,0,0,0,1,0,0,0,1,0], [3,6]],
         'E': [0,0,0,0,1,0,0,0,0,0,0,0], [0]}
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
        number contain in the dict 'pcv_dict' still apply if a single coefficient is selected with this parameter.
        Default value is None.
        
    '''
    phivals = np.arange(0, 2*np.pi, 0.01)
    mu_step = .025
    muvals = np.arange(0, 1. + mu_step, mu_step)
    
    #powerset of all phis and mus.
    cartesian_polar = np.array(np.meshgrid(phivals, muvals)).T.reshape(-1, 2)
    
    #generating the color corresponding to each point.
    color_arr = []
    for phi, mu in cartesian_polar:
        hexa = rgb_to_hex(circular_hue(phi, magnitude=mu, opacity_mapping=True))
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

