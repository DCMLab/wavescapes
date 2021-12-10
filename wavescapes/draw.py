from warnings import warn
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.ticker import IndexLocator,FuncFormatter
SQRT_OF_THREE = math.sqrt(3)

def rgb_to_hex(rgb):
    if type(rgb) is str and rgb[0] == '#' and len(rgb) > 6:
        # we already have an hex value given let's just return it back.
        return rgb
    elif rgb[0] == (0xff+1):
        #not an element that has to be drawn
        return None
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


def compute_plot_height(width, mat_dim, primitive):
    primitive = primitive.lower()
    if primitive == Wavescape.HEXAGON_STR:
        return Wavescape.HEXAGON_PLOT_HEIGHT(width, mat_dim)
    elif primitive ==  Wavescape.RHOMBUS_STR:
        return (width/2.) * SQRT_OF_THREE
    elif primitive ==  Wavescape.DIAMOND_STR:
        return width
    else:
        raise Exception('Unknown drawing primitive: %s'%primitive)


def compute_bounding_box_limits(mat_dim, start, end, width, height, primitive_half_width):
    #this is invariant.
    bottom = -height/2.
    if start == 0 and mat_dim == end:
        #base case.
        return (-width/2.-primitive_half_width, width/2.-primitive_half_width, height/2., bottom)
    else:
        #resizing needed because of subwavescaping
        scaling_factor = (end - start)/mat_dim
        subzone_width = width * scaling_factor
        subzone_height = height * scaling_factor + primitive_half_width
        left = (width*(start/mat_dim)) - width/2. -primitive_half_width
        right = left + subzone_width
        return (left,right,bottom+subzone_height,bottom)


def get_primitive_height(primitive_name, primitive_width):
    primitive_name = primitive_name.lower()
    if primitive_name == Wavescape.HEXAGON_STR:
        return 2*SQRT_OF_THREE*primitive_width/3.
    elif primitive_name == Wavescape.DIAMOND_STR:
        return primitive_width*2
    elif primitive_name == Wavescape.RHOMBUS_STR:
        return primitive_width * SQRT_OF_THREE
    else:
        raise Exception('Unknown drawing primitive: %s'%primitive_name)


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
        is_rgba = len(curr_color) > 8
        edgecolor = 'black' if stroke else None if is_rgba else curr_color
        # this is to treat the bottom diamond that needs to be drawn as a triangle
        last_coord = (x,y if self.bottom_diamond else y-self.half_height)
        return Polygon(((x-self.half_width, y),
                               (x, y+self.half_height),
                               (x+self.half_width, y),
                               last_coord),
                         facecolor = curr_color,
                         edgecolor=edgecolor,
                         linewidth=stroke if stroke else None)


class HexagonPrimitive(object):
    def __init__(self, x, y, width, color):
        self.half_width = width/2.
        self.h = SQRT_OF_THREE*self.half_width/3.
        self.half_height = self.h/2.
        self.x = x
        self.y = y
        self.color = color

    def draw(self, new_color=None, stroke=None):
        w = self.half_width
        h = self.h
        d_x = self.x
        d_y = self.y
        curr_color = new_color if new_color else self.color
        is_rgba = len(curr_color) > 8
        edgecolor = 'black' if stroke else None if is_rgba else curr_color
        return Polygon(((d_x+w, d_y+h),
                        (d_x, d_y+2*h),
                        (d_x-w, d_y+h),
                        (d_x-w, d_y-h),
                        (d_x, d_y-2*h),
                        (d_x+w, d_y-h)),
                         facecolor = curr_color,
                         edgecolor= edgecolor,
                         linewidth= stroke if stroke else None)


def new_primitive_with_coords(curr_color, x, y, hws, hhs, primitive_name, primitive_width, primitive_height):
    '''
    Generates the primitive at the right place in the final plot according to the parameter chosen.
    '''
    #classic x-axis placement taking into account the half width of the hexagon
    d_x = primitive_width*x
    #Now shifting all of this to the left to go from utm placement to pyramid placement
    d_x = d_x - primitive_width*y/2.
    #And finally shifting this to take into account center placement of the figure
    d_x = d_x - hws
    #accounting center placement in the y axis before adding the actual element y-position
    d_y = -hhs
    
    primitive_name = primitive_name.lower()
    if primitive_name == Wavescape.HEXAGON_STR:
        d_y = d_y + primitive_height/2.+(0.75*primitive_height)*y
        return HexagonPrimitive(d_x, d_y, primitive_width, curr_color)
    elif primitive_name == Wavescape.RHOMBUS_STR or primitive_name == Wavescape.DIAMOND_STR:
        d_y = d_y + primitive_height/2.*y
        return DiamondPrimitive(d_x, d_y, \
                          primitive_width, primitive_height, curr_color, y == 0)
    else:
        raise Exception('Unknown drawing primitive: %s'%primitive_name)


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
        
    primitive : {'diamond', 'rhombus', 'hexagon'} , optional 
        the drawing shape that forms a single colored element from the plot. Three primitives are 
        currently available:
          -'diamond': diamond whose height is twice its width
          -'rhombus': diamond formed by two equilateral triangles. Each side is the same size
          -'hexagon': a hexagon, i.e. a 6 sides polygonal shape, each side being the same size.
        default value is 'rhombus'
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
    ALL_PRIMITIVE_SUPPORTED = [DIAMOND_STR, RHOMBUS_STR, HEXAGON_STR]
    
    def __init__(self, utm, width, primitive='rhombus'):
        self.utm = utm
        self.width = width
        self.primitive = primitive
        
        mat_dim, mat_dim_other_axis, mat_depth = utm.shape
        if mat_dim != mat_dim_other_axis:
            raise Exception("The upper triangle matrix is not a square matrix")
        if mat_dim > self.width/2:
            raise Exception("The number of elements to be drawn exceeds the wavescape's resolution.(%d elements out of %d allowed by the resolution) Increase the width of the plot to solve this issue" % (mat_dim, self.width/2))
        if (mat_depth < 3 or mat_depth > 4):
            raise Exception("The upper triangle matrix given as argument does not hold either RGB or RGBA values")
        self.mat_dim = mat_dim
        
        #building a matrix with None to hold the element object for drawing them later.
        self.matrix_primitive = np.full((mat_dim, mat_dim), None, object)
        
        if primitive.lower() in Wavescape.ALL_PRIMITIVE_SUPPORTED:
            self.height = compute_plot_height(self.width, mat_dim, primitive)
            self.generate_primitives()
        else:
            raise Exception('Unkown drawing primitive: %s'%primitive)
            
    def generate_highlights(self, unit_width, linewidth):
        '''
        Helper method
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
                raise Exception('Subpart highlights\' indices cannot be above the number of element at the base of the '
                                'wavescape (%d)'%self.mat_dim)
            tri_width = (hi-lo) * unit_width
            tri_height = compute_plot_height(tri_width, hi-lo, self.primitive)
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
                         linewidth=linewidth))
        self.subparts = triangles
                
    def generate_primitives(self):
        '''
        Helper method, called by the constructor of the class. 
        This method takes care of generating a "primitive" class instance per element forming the wavescape.  
        One matplotlib.patches.Polygon is generated per element of the plot. 
        The draw method takes care of drawing those patches on the final figure.
        '''
        half_width_shift = self.width/2.
        half_height_shift = self.height/2.
        primitive_width = self.width/float(self.mat_dim)
        primitive_height = get_primitive_height(self.primitive, primitive_width)
        
        for y in range(self.mat_dim):
            for x in range(y, self.mat_dim):
                curr_color = rgb_to_hex(self.utm[y][x])
                if curr_color:
                    self.matrix_primitive[y][x] = new_primitive_with_coords(curr_color, x, y, half_width_shift,
                                                                            half_height_shift, self.primitive,
                                                                            primitive_width, primitive_height) 

    def draw(self, ax=None, aw_per_tick = None, tick_offset=0, tick_start=0, tick_factor=1, subparts_highlighted = None,
             indicator_size = None, add_line = None, label=None, label_size=None):
        '''
        After being called on a properly initialised instance of a Wavescape object,
        this method draws the visual plot known as "wavescape" and generate a 
        matplotlib.pyplot figure of it. This means any of the method from this
        library can be used after this method has been called in order to
        save or alter the figure produced. 

        Parameters
        ----------
        
        ax: matplotlib figure, optional
            if provided, will draw the wavescape on it. Useful if many wavescapes need to
            be drawn in the same figure, or if the plot needs to be combined to others.
            Default value is None.
        
        aw_per_tick: numeric value, optional
            Ratio of tick per elements of the lowest level in the wavescape. If aw_per_tick has value 1,
            one horizontal axis tick will be drawn per element at the lowest hierarchical level of the wavescape, 
            if it has value 2, then a tick will be drawn each two elements and so forth. For the ticks to represent the bar numbers,
            a pre-existing knowledge of the time signature of the musical piece is required. (for instance, if a piece is in 4/4,
            and the analysis window has a size of one quarter note, then aw_per_tick needs to have value 4 for the
            ticks to correspond to bar numbers)
            Default value is None (meaning no ticks are drawn)
            
        tick_offset: int, optional
            offset value for the tick drawn according to the 'aw_per_tick' parameter. This is done
            so that musical pieces with 0th measure can have tick accurately representing the source
            material's bar numbers. Like the tick ratio, this number is relative to the analysis window's
            size and requires a pre-existing knowledge of the score.
            Its value must be higher or equal to 0 but strictly lower than aw_per_tick.
            If it has value "0", the first tick of the plot will be set to the value of 1. 
            For having the first tick of the plot set to the value of 0, leave that parameter to
            None and have a coherent value for aw_per_tick
            Default value is 0 (meaning no tick offset). 
        
        tick_start: int, optional
            Indicates at which number to start the tick numbering. We recommand, for
            classical score, to put the value "1", as most scores starts numbering their bars
            with 1 instead of 0.
            Default value is 0.
        
        tick_factor: float, optional
            Multiply the major ticks numbers displayed on the x-axis by a constant.
            Can be useful for very large pieces, where displaying a single tick on each
            bottom row element would make the x-axis hard to read. By increasing the aw_per_tick,
            and giving this parameter a certain value, it is possible to display less ticks while 
            still keeping the right indicative numbers with respect to the unit system chosen. 
            Default value is 1.0, (meaning the value displayed is consistent with the number of ticks, with 
            respect to tick_offset of course.)
        
        subparts_highlighted: tuple of numeric values, OR array of tuples of numeric values, optional
            List of subsections that needs to be highlighted with black outlines on the wavescape. Units
            are expressed in number of analysis windows. For example, if a musical piece has a 4/4 
            time signature, an analysis window of 1 quarter note and a total of 10 bars, the
            value [[20,28],[32,36]] for 'subparts_highlighted' will draw black outlines on the region
            of the wavescape corresponding to bars 5 until 7 and 8 until 9. 
            This parameter is interpreted differently if it has the shape of a single tuple. In such case,
            a subpplot corresponding to the delimitation in the tuple will be drawn instead. For instance,
            'subparts_highlighted=[20,28]' will only draw the wavescape corresponding to piece from bar
            5 to 7 (if the musical piece is in 4/4). Be careful not to write 'subparts_highlighted=[[20,28]]' 
            which is interpreted as drawing a highlight of bar 5 to 7 on the full wavescape.
            Default value is None (meaning no highlighting and no subsection of wavescape drawn)

        indicator_size: float, optional 
            Determine the factor by which to increment the size of the rounded indicators on the lateral edges of the plot that need. 
            A rounded indicator is drawn at each eight of the height of the plot if a value is provided for this argument.
            Enter the value "1.0" for the default size.
            Default value is None (meaning no vertical indicators)
        
        add_line: numeric value, optional
            if provided, this parameter represents the thickness of the black line outlining
            all element of the plot (drawing primitives).
            Default value is None.
        
        label: str, optional
            If provided, add this string as a textual label on the top left corner of the resulting plot.
            Can be used to specify the Fourier coefficient visualised on the wavescape for example.
            Default value is None
            
        label_size: float, optional
            Determine the size of the top-left label and the tick number labels if provided
            Default value is None (in which case the default 
            size of the labels is the width of the plot divided by 30)

        Returns
        -------
        Nothing, but a matplotlib.pyplot figure is produced by this method, and any method of pyplot
        can be used to alter the resulting figure (notably matplotlib.pyplot.savefig can be used to save the
        resulting figure in a file)
        '''
        
        start_primitive_idx = 0
        utm_w = self.matrix_primitive.shape[0]
        utm_h = self.matrix_primitive.shape[1]
        
        if self.matrix_primitive is None or utm_w < 1 or utm_h < 1:
            raise Exception("Cannot draw when there is nothing to draw.")
        
        if aw_per_tick is not None: 
            
            if aw_per_tick < 1 or type(aw_per_tick) is not int:
                raise Exception("'aw_per_tick' must be an integer greater or equal to 1")
            
            if tick_factor <= 0 or (type(tick_factor) is not int and type(tick_factor) is not float):
                raise Exception("'tick_factor' must be a numeric value greater than 0")

            if tick_start < 0 or type(tick_start) is not int:
                raise Exception("'tick_start' must be a numeric value greater than or equal to 0")
            
            #argument start_offseet is only meaningless if there is tick ratio involved in the plotting
            if type(tick_offset) is not int or tick_offset < 0 or tick_offset > aw_per_tick:
                raise Exception("Stat offset needs to be a positive integer that is smaller or equal to the tick ratio")
        
        
        idx = 0
        primitive_half_width = 0
        while primitive_half_width == 0 and idx < utm_w:
            elem = self.matrix_primitive[0][idx]
            if elem:
                #needed for highlights and some tick ofsetting later.
                primitive_half_width = elem.half_width
                #needed for outlines
                primitive_half_height = elem.half_height
            idx += 1
            
        if primitive_half_width == 0 and idx == utm_w:
            raise Exception('No primitive were generated for the drawing of the wavescape')
        
        subpart_offset = 0
        if subparts_highlighted is not None:
            hl_dimensions = len(np.shape(subparts_highlighted))
            if hl_dimensions == 1 and len(subparts_highlighted) == 2:
                #restraining dimensions of what'll be drawn
                if subparts_highlighted[0] >= subparts_highlighted[1]:
                    raise Exception('subparts_highlighted coordinates should be ordered and not the same')
                elif subparts_highlighted[0] > utm_w or subparts_highlighted[1] > utm_w:
                    raise Exception('subparts_highlighted coordinates should not exceed the matrix size (%d)'%utm_w)
                    
                start_primitive_idx = subparts_highlighted[0]
                utm_h = subparts_highlighted[1]
                utm_w = subparts_highlighted[1]
                
                #cannot work with this "mode"
                if indicator_size:
                    msg = "Vertical indicators cannot be drawn when subparts are being produced."
                    warn(msg)
                    indicator_size = None
                self.subparts=None
                
                #need to adapt start offset
                if aw_per_tick:
                    subpart_offset = -(start_primitive_idx % aw_per_tick)
                
            elif hl_dimensions == 2:
                #wavescape fullpart conserved, only has to draw highlights on it.
                self.subparts = subparts_highlighted
                #so the highlights are thicker than the delimiting lines
                linewidth = 2.5*add_line if add_line else 1
                self.generate_highlights(primitive_half_width * 2, linewidth)
            else:
                raise Exception('subparts_highlighted should be a matrix of numeric values or a tuple of numbers')
        else:
            self.subparts = None
        
        height = self.height
        width = self.width
    
        dpi = 96 #(most common dpi values for computers' screen)
        if not ax:
            fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            ax = fig.add_subplot(111, aspect='equal')
        

        for y in range(utm_h):
            for x in range(start_primitive_idx+y, utm_w):
                element = self.matrix_primitive[y][x]
                #array of primitive is by default filled with None value, this avoid that.
                if element:
                    ax.add_patch(element.draw(stroke=add_line))

        if indicator_size:
            ind_width = (width if self.primitive != self.HEXAGON_STR else width + 2) * indicator_size
            mid_size = int(ind_width / 60.)
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
                        ax.add_patch(Circle((i*x*width-primitive_half_width, y*height), radius=p['size'],
                                            facecolor=p['facecolor'],
                                            edgecolor=p['edgecolor'], linewidth=stroke_width))

        plt.autoscale(enable = True)
        
        if not aw_per_tick and not label and label_size:
            msg = "'label_size' argument provided when nothing needs to be labeled on the figure."
            warn(msg)
            
        labelsize = label_size if label_size else self.width/30.
        
        if aw_per_tick:
            indiv_w = primitive_half_width*2
            
            scale_x = indiv_w * aw_per_tick
            maj_loc_offset = (tick_offset+subpart_offset)*indiv_w

            major_ticks_formatter = lambda x, pos: '{0:g}'.format((math.ceil((x + self.width / 2.) / scale_x) * tick_factor + (0 if tick_offset < 1 else -1))+tick_start)
            
            ticks_x = FuncFormatter(major_ticks_formatter)
            ax.tick_params(which='major', length=self.width/50., labelsize=labelsize)
            ax.tick_params(which='minor', length=self.width/100.)
            
            ax.xaxis.set_major_formatter(ticks_x)
            number_of_ticks = self.width/scale_x
            number_of_bars = (utm_w-start_primitive_idx)/aw_per_tick
            major_tick_base = scale_x*round(number_of_ticks/(8 if number_of_bars > 8 else number_of_ticks))
            ax.xaxis.set_major_locator(IndexLocator(base=major_tick_base, offset=maj_loc_offset))
            
            #display minor indicators
            ax.xaxis.set_minor_locator(IndexLocator(base=scale_x, offset=maj_loc_offset))
            
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

        #whether a subwavescape is drawn will influence the values returned by this function.
        bb_l, bb_r, bb_t, bb_b = compute_bounding_box_limits(self.matrix_primitive.shape[0], start_primitive_idx, utm_w,
                                                             self.width, self.height, primitive_half_width)
        
        #needed to account for line_width in the plot's size
        if add_line:
            bb_l += -add_line
            bb_r += add_line
        
        if label:
            new_width = np.abs(bb_l - bb_r)
            new_height = np.abs(bb_b - bb_t)
            x_pos = (new_width/20.) + bb_l
            y_pos =  bb_t - (new_height/20.) 
            ax.annotate(label, (x_pos, y_pos), size=labelsize, annotation_clip=False, horizontalalignment='left',
                        verticalalignment='top')
            
        #remove top and bottom margins 
        ax.set_ylim(bottom=bb_b, top=bb_t)
        ax.set_xlim(left=bb_l, right=bb_r)
        plt.tight_layout()
