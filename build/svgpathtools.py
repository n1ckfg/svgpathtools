'''
The MIT License (MIT)

Copyright (c) 2015 Andrew Allan Port
Copyright (c) 2013-2014 Lennart Regebro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from .bezier import (bezier_point, bezier2polynomial,
                     polynomial2bezier, split_bezier,
                     bezier_bounding_box, bezier_intersections,
                     bezier_by_line_intersections)
from .path import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                   bezier_segment, is_bezier_segment, is_path_segment,
                   is_bezier_path, concatpaths, poly2bez, bpoints2bezier,
                   closest_point_in_path, farthest_point_in_path,
                   path_encloses_pt, bbox2path, polygon, polyline)
from .parser import parse_path
from .paths2svg import disvg, wsvg, paths2Drawing
from .polytools import polyroots, polyroots01, rational_limit, real, imag
from .misctools import hex2rgb, rgb2hex
from .smoothing import smoothed_path, smoothed_joint, is_differentiable, kinks
from .document import Document, CONVERSIONS, CONVERT_ONLY_PATHS, SVG_GROUP_TAG, SVG_NAMESPACE
from .svg_io_sax import SaxDocument

try:
    from .svg_to_paths import svg2paths, svg2paths2
except ImportError:
    pass

"""This submodule contains tools that deal with generic, degree n, Bezier
curves.
Note:  Bezier curves here are always represented by the tuple of their control
points given by their standard representation."""

# External dependencies:
from __future__ import division, absolute_import, print_function
from math import factorial as fac, ceil, log, sqrt
from numpy import poly1d

# Internal dependencies
from .polytools import real, imag, polyroots, polyroots01


# Evaluation ##################################################################

def n_choose_k(n, k):
    return fac(n)//fac(k)//fac(n-k)


def bernstein(n, t):
    """returns a list of the Bernstein basis polynomials b_{i, n} evaluated at
    t, for i =0...n"""
    t1 = 1-t
    return [n_choose_k(n, k) * t1**(n-k) * t**k for k in range(n+1)]


def bezier_point(p, t):
    """Evaluates the Bezier curve given by it's control points, p, at t.
    Note: Uses Horner's rule for cubic and lower order Bezier curves.
    Warning:  Be concerned about numerical stability when using this function
    with high order curves."""

    # begin arc support block ########################
    try:
        p.large_arc
        return p.point(t)
    except:
        pass
    # end arc support block ##########################

    deg = len(p) - 1
    if deg == 3:
        return p[0] + t*(
            3*(p[1] - p[0]) + t*(
                3*(p[0] + p[2]) - 6*p[1] + t*(
                    -p[0] + 3*(p[1] - p[2]) + p[3])))
    elif deg == 2:
        return p[0] + t*(
            2*(p[1] - p[0]) + t*(
                p[0] - 2*p[1] + p[2]))
    elif deg == 1:
        return p[0] + t*(p[1] - p[0])
    elif deg == 0:
        return p[0]
    else:
        bern = bernstein(deg, t)
        return sum(bern[k]*p[k] for k in range(deg+1))


# Conversion ##################################################################

def bezier2polynomial(p, numpy_ordering=True, return_poly1d=False):
    """Converts a tuple of Bezier control points to a tuple of coefficients
    of the expanded polynomial.
    return_poly1d : returns a numpy.poly1d object.  This makes computations
    of derivatives/anti-derivatives and many other operations quite quick.
    numpy_ordering : By default (to accommodate numpy) the coefficients will
    be output in reverse standard order."""
    if len(p) == 4:
        coeffs = (-p[0] + 3*(p[1] - p[2]) + p[3],
                  3*(p[0] - 2*p[1] + p[2]),
                  3*(p[1]-p[0]),
                  p[0])
    elif len(p) == 3:
        coeffs = (p[0] - 2*p[1] + p[2],
                  2*(p[1] - p[0]),
                  p[0])
    elif len(p) == 2:
        coeffs = (p[1]-p[0],
                  p[0])
    elif len(p) == 1:
        coeffs = p
    else:
        # https://en.wikipedia.org/wiki/Bezier_curve#Polynomial_form
        n = len(p) - 1
        coeffs = [fac(n)//fac(n-j) * sum(
            (-1)**(i+j) * p[i] / (fac(i) * fac(j-i)) for i in range(j+1))
            for j in range(n+1)]
        coeffs.reverse()
    if not numpy_ordering:
        coeffs = coeffs[::-1]  # can't use .reverse() as might be tuple
    if return_poly1d:
        return poly1d(coeffs)
    return coeffs


def polynomial2bezier(poly):
    """Converts a cubic or lower order Polynomial object (or a sequence of
    coefficients) to a CubicBezier, QuadraticBezier, or Line object as
    appropriate."""
    if isinstance(poly, poly1d):
        c = poly.coeffs
    else:
        c = poly
    order = len(c)-1
    if order == 3:
        bpoints = (c[3], c[2]/3 + c[3], (c[1] + 2*c[2])/3 + c[3],
                   c[0] + c[1] + c[2] + c[3])
    elif order == 2:
        bpoints = (c[2], c[1]/2 + c[2], c[0] + c[1] + c[2])
    elif order == 1:
        bpoints = (c[1], c[0] + c[1])
    else:
        raise AssertionError("This function is only implemented for linear, "
                             "quadratic, and cubic polynomials.")
    return bpoints


# Curve Splitting #############################################################

def split_bezier(bpoints, t):
    """Uses deCasteljau's recursion to split the Bezier curve at t into two
    Bezier curves of the same order."""
    def split_bezier_recursion(bpoints_left_, bpoints_right_, bpoints_, t_):
        if len(bpoints_) == 1:
            bpoints_left_.append(bpoints_[0])
            bpoints_right_.append(bpoints_[0])
        else:
            new_points = [None]*(len(bpoints_) - 1)
            bpoints_left_.append(bpoints_[0])
            bpoints_right_.append(bpoints_[-1])
            for i in range(len(bpoints_) - 1):
                new_points[i] = (1 - t_)*bpoints_[i] + t_*bpoints_[i + 1]
            bpoints_left_, bpoints_right_ = split_bezier_recursion(
                bpoints_left_, bpoints_right_, new_points, t_)
        return bpoints_left_, bpoints_right_

    bpoints_left = []
    bpoints_right = []
    bpoints_left, bpoints_right = \
        split_bezier_recursion(bpoints_left, bpoints_right, bpoints, t)
    bpoints_right.reverse()
    return bpoints_left, bpoints_right


def halve_bezier(p):

    # begin arc support block ########################
    try:
        p.large_arc
        return p.split(0.5)
    except:
        pass
    # end arc support block ##########################

    if len(p) == 4:
        return ([p[0], (p[0] + p[1])/2, (p[0] + 2*p[1] + p[2])/4,
                 (p[0] + 3*p[1] + 3*p[2] + p[3])/8],
                [(p[0] + 3*p[1] + 3*p[2] + p[3])/8,
                 (p[1] + 2*p[2] + p[3])/4, (p[2] + p[3])/2, p[3]])
    else:
        return split_bezier(p, 0.5)


# Bounding Boxes ##############################################################

def bezier_real_minmax(p):
    """returns the minimum and maximum for any real cubic bezier"""
    local_extremizers = [0, 1]
    if len(p) == 4:  # cubic case
        a = [p.real for p in p]
        denom = a[0] - 3*a[1] + 3*a[2] - a[3]
        if denom != 0:
            delta = a[1]**2 - (a[0] + a[1])*a[2] + a[2]**2 + (a[0] - a[1])*a[3]
            if delta >= 0:  # otherwise no local extrema
                sqdelta = sqrt(delta)
                tau = a[0] - 2*a[1] + a[2]
                r1 = (tau + sqdelta)/denom
                r2 = (tau - sqdelta)/denom
                if 0 < r1 < 1:
                    local_extremizers.append(r1)
                if 0 < r2 < 1:
                    local_extremizers.append(r2)
            local_extrema = [bezier_point(a, t) for t in local_extremizers]
            return min(local_extrema), max(local_extrema)

    # find reverse standard coefficients of the derivative
    dcoeffs = bezier2polynomial(a, return_poly1d=True).deriv().coeffs

    # find real roots, r, such that 0 <= r <= 1
    local_extremizers += polyroots01(dcoeffs)
    local_extrema = [bezier_point(a, t) for t in local_extremizers]
    return min(local_extrema), max(local_extrema)


def bezier_bounding_box(bez):
    """returns the bounding box for the segment in the form
    (xmin, xmax, ymin, ymax).
    Warning: For the non-cubic case this is not particularly efficient."""

    # begin arc support block ########################
    try:
        bla = bez.large_arc
        return bez.bbox()  # added to support Arc objects
    except:
        pass
    # end arc support block ##########################

    if len(bez) == 4:
        xmin, xmax = bezier_real_minmax([p.real for p in bez])
        ymin, ymax = bezier_real_minmax([p.imag for p in bez])
        return xmin, xmax, ymin, ymax
    poly = bezier2polynomial(bez, return_poly1d=True)
    x = real(poly)
    y = imag(poly)
    dx = x.deriv()
    dy = y.deriv()
    x_extremizers = [0, 1] + polyroots(dx, realroots=True,
                                    condition=lambda r: 0 < r < 1)
    y_extremizers = [0, 1] + polyroots(dy, realroots=True,
                                    condition=lambda r: 0 < r < 1)
    x_extrema = [x(t) for t in x_extremizers]
    y_extrema = [y(t) for t in y_extremizers]
    return min(x_extrema), max(x_extrema), min(y_extrema), max(y_extrema)


def box_area(xmin, xmax, ymin, ymax):
    """
    INPUT: 2-tuple of cubics (given by control points)
    OUTPUT: boolean
    """
    return (xmax - xmin)*(ymax - ymin)


def interval_intersection_width(a, b, c, d):
    """returns the width of the intersection of intervals [a,b] and [c,d]
    (thinking of these as intervals on the real number line)"""
    return max(0, min(b, d) - max(a, c))


def boxes_intersect(box1, box2):
    """Determines if two rectangles, each input as a tuple
        (xmin, xmax, ymin, ymax), intersect."""
    xmin1, xmax1, ymin1, ymax1 = box1
    xmin2, xmax2, ymin2, ymax2 = box2
    if interval_intersection_width(xmin1, xmax1, xmin2, xmax2) and \
            interval_intersection_width(ymin1, ymax1, ymin2, ymax2):
        return True
    else:
        return False


# Intersections ###############################################################

class ApproxSolutionSet(list):
    """A class that behaves like a set but treats two elements , x and y, as
    equivalent if abs(x-y) < self.tol"""
    def __init__(self, tol):
        self.tol = tol

    def __contains__(self, x):
        for y in self:
            if abs(x - y) < self.tol:
                return True
        return False

    def appadd(self, pt):
        if pt not in self:
            self.append(pt)


class BPair(object):
    def __init__(self, bez1, bez2, t1, t2):
        self.bez1 = bez1
        self.bez2 = bez2
        self.t1 = t1  # t value to get the mid point of this curve from cub1
        self.t2 = t2  # t value to get the mid point of this curve from cub2


def bezier_intersections(bez1, bez2, longer_length, tol=1e-8, tol_deC=1e-8):
    """INPUT:
    bez1, bez2 = [P0,P1,P2,...PN], [Q0,Q1,Q2,...,PN] defining the two
    Bezier curves to check for intersections between.
    longer_length - the length (or an upper bound) on the longer of the two
    Bezier curves.  Determines the maximum iterations needed together with tol.
    tol - is the smallest distance that two solutions can differ by and still
    be considered distinct solutions.
    OUTPUT: a list of tuples (t,s) in [0,1]x[0,1] such that
        abs(bezier_point(bez1[0],t) - bezier_point(bez2[1],s)) < tol_deC
    Note: This will return exactly one such tuple for each intersection
    (assuming tol_deC is small enough)."""
    maxits = int(ceil(1-log(tol_deC/longer_length)/log(2)))
    pair_list = [BPair(bez1, bez2, 0.5, 0.5)]
    intersection_list = []
    k = 0
    approx_point_set = ApproxSolutionSet(tol)
    while pair_list and k < maxits:
        new_pairs = []
        delta = 0.5**(k + 2)
        for pair in pair_list:
            bbox1 = bezier_bounding_box(pair.bez1)
            bbox2 = bezier_bounding_box(pair.bez2)
            if boxes_intersect(bbox1, bbox2):
                if box_area(*bbox1) < tol_deC and box_area(*bbox2) < tol_deC:
                    point = bezier_point(bez1, pair.t1)
                    if point not in approx_point_set:
                        approx_point_set.append(point)
                        # this is the point in the middle of the pair
                        intersection_list.append((pair.t1, pair.t2))

                    # this prevents the output of redundant intersection points
                    for otherPair in pair_list:
                        if pair.bez1 == otherPair.bez1 or \
                                pair.bez2 == otherPair.bez2 or \
                                pair.bez1 == otherPair.bez2 or \
                                pair.bez2 == otherPair.bez1:
                            pair_list.remove(otherPair)
                else:
                    (c11, c12) = halve_bezier(pair.bez1)
                    (t11, t12) = (pair.t1 - delta, pair.t1 + delta)
                    (c21, c22) = halve_bezier(pair.bez2)
                    (t21, t22) = (pair.t2 - delta, pair.t2 + delta)
                    new_pairs += [BPair(c11, c21, t11, t21),
                                  BPair(c11, c22, t11, t22),
                                  BPair(c12, c21, t12, t21),
                                  BPair(c12, c22, t12, t22)]
        pair_list = new_pairs
        k += 1
    if k >= maxits:
        raise Exception("bezier_intersections has reached maximum "
                        "iterations without terminating... "
                        "either there's a problem/bug or you can fix by "
                        "raising the max iterations or lowering tol_deC")
    return intersection_list


def bezier_by_line_intersections(bezier, line):
    """Returns tuples (t1,t2) such that bezier.point(t1) ~= line.point(t2)."""
    # The method here is to translate (shift) then rotate the complex plane so
    # that line starts at the origin and proceeds along the positive real axis.
    # After this transformation, the intersection points are the real roots of
    # the imaginary component of the bezier for which the real component is
    # between 0 and abs(line[1]-line[0])].
    assert len(line[:]) == 2
    assert line[0] != line[1]
    if not any(p != bezier[0] for p in bezier):
        raise ValueError("bezier is nodal, use "
                         "bezier_by_line_intersection(bezier[0], line) "
                         "instead for a bool to be returned.")

    # First let's shift the complex plane so that line starts at the origin
    shifted_bezier = [z - line[0] for z in bezier]
    shifted_line_end = line[1] - line[0]
    line_length = abs(shifted_line_end)

    # Now let's rotate the complex plane so that line falls on the x-axis
    rotation_matrix = line_length/shifted_line_end
    transformed_bezier = [rotation_matrix*z for z in shifted_bezier]

    # Now all intersections should be roots of the imaginary component of
    # the transformed bezier
    transformed_bezier_imag = [p.imag for p in transformed_bezier]
    coeffs_y = bezier2polynomial(transformed_bezier_imag)
    roots_y = list(polyroots01(coeffs_y))  # returns real roots 0 <= r <= 1

    transformed_bezier_real = [p.real for p in transformed_bezier]
    intersection_list = []
    for bez_t in set(roots_y):
        xval = bezier_point(transformed_bezier_real, bez_t)
        if 0 <= xval <= line_length:
            line_t = xval/line_length
            intersection_list.append((bez_t, line_t))
    return intersection_list

"""(Experimental) replacement for import/export functionality.

This module contains the `Document` class, a container for a DOM-style 
document (e.g. svg, html, xml, etc.) designed to replace and improve 
upon the IO functionality of svgpathtools (i.e. the svg2paths and 
disvg/wsvg functions). 

An Historic Note:
     The functionality in this module is meant to replace and improve 
     upon the IO functionality previously provided by the the 
     `svg2paths` and `disvg`/`wsvg` functions. 

Example:
    Typical usage looks something like the following.

        >> from svgpathtools import *
        >> doc = Document('my_file.html')
        >> results = doc.flatten_all_paths()
        >> for result in results:
        >>     path = result.path
        >>     # Do something with the transformed Path object.
        >>     element = result.element
        >>     # Inspect the raw SVG element. This gives access to the
        >>     # path's attributes
        >>     transform = result.transform
        >>     # Use the transform that was applied to the path.
        >> foo(doc.tree)  # do stuff using ElementTree's functionality
        >> doc.display()  # display doc in OS's default application
        >> doc.save('my_new_file.html')

A Big Problem:  
    Derivatives and other functions may be messed up by 
    transforms unless transforms are flattened (and not included in 
    css)
"""

# External dependencies
from __future__ import division, absolute_import, print_function
import os
import collections
import xml.etree.ElementTree as etree
from xml.etree.ElementTree import Element, SubElement, register_namespace
import warnings

# Internal dependencies
from .parser import parse_path
from .parser import parse_transform
from .svg_to_paths import (path2pathd, ellipse2pathd, line2pathd,
                           polyline2pathd, polygon2pathd, rect2pathd)
from .misctools import open_in_browser
from .path import *

# To maintain forward/backward compatibility
try:
    str = basestring
except NameError:
    pass

# Let xml.etree.ElementTree know about the SVG namespace
SVG_NAMESPACE = {'svg': 'http://www.w3.org/2000/svg'}
register_namespace('svg', 'http://www.w3.org/2000/svg')

# THESE MUST BE WRAPPED TO OUTPUT ElementTree.element objects
CONVERSIONS = {'path': path2pathd,
               'circle': ellipse2pathd,
               'ellipse': ellipse2pathd,
               'line': line2pathd,
               'polyline': polyline2pathd,
               'polygon': polygon2pathd,
               'rect': rect2pathd}

CONVERT_ONLY_PATHS = {'path': path2pathd}

SVG_GROUP_TAG = 'svg:g'


def flatten_all_paths(group, group_filter=lambda x: True,
                      path_filter=lambda x: True, path_conversions=CONVERSIONS,
                      group_search_xpath=SVG_GROUP_TAG):
    """Returns the paths inside a group (recursively), expressing the
    paths in the base coordinates.

    Note that if the group being passed in is nested inside some parent
    group(s), we cannot take the parent group(s) into account, because
    xml.etree.Element has no pointer to its parent. You should use
    Document.flatten_group(group) to flatten a specific nested group into
    the root coordinates.

    Args:
        group is an Element
        path_conversions (dict):
            A dictionary to convert from an SVG element to a path data
            string. Any element tags that are not included in this
            dictionary will be ignored (including the `path` tag). To
            only convert explicit path elements, pass in
            `path_conversions=CONVERT_ONLY_PATHS`.
    """
    if not isinstance(group, Element):
        raise TypeError('Must provide an xml.etree.Element object. '
                        'Instead you provided {0}'.format(type(group)))

    # Stop right away if the group_selector rejects this group
    if not group_filter(group):
        return []

    # To handle the transforms efficiently, we'll traverse the tree of
    # groups depth-first using a stack of tuples.
    # The first entry in the tuple is a group element and the second
    # entry is its transform. As we pop each entry in the stack, we
    # will add all its child group elements to the stack.
    StackElement = collections.namedtuple('StackElement',
                                          ['group', 'transform'])

    def new_stack_element(element, last_tf):
        return StackElement(element, last_tf.dot(
            parse_transform(element.get('transform'))))

    def get_relevant_children(parent, last_tf):
        children = []
        for elem in filter(group_filter,
                           parent.iterfind(group_search_xpath, SVG_NAMESPACE)):
            children.append(new_stack_element(elem, last_tf))
        return children

    stack = [new_stack_element(group, np.identity(3))]

    FlattenedPath = collections.namedtuple('FlattenedPath',
                                           ['path', 'element', 'transform'])
    paths = []

    while stack:
        top = stack.pop()

        # For each element type that we know how to convert into path
        # data, parse the element after confirming that the path_filter
        # accepts it.
        for key, converter in path_conversions.items():
            for path_elem in filter(path_filter, top.group.iterfind(
                    'svg:'+key, SVG_NAMESPACE)):
                path_tf = top.transform.dot(
                    parse_transform(path_elem.get('transform')))
                path = transform(parse_path(converter(path_elem)), path_tf)
                paths.append(FlattenedPath(path, path_elem, path_tf))

        stack.extend(get_relevant_children(top.group, top.transform))

    return paths


def flatten_group(group_to_flatten, root, recursive=True,
                  group_filter=lambda x: True, path_filter=lambda x: True,
                  path_conversions=CONVERSIONS,
                  group_search_xpath=SVG_GROUP_TAG):
    """Flatten all the paths in a specific group.

    The paths will be flattened into the 'root' frame. Note that root
    needs to be an ancestor of the group that is being flattened.
    Otherwise, no paths will be returned."""

    if not any(group_to_flatten is descendant for descendant in root.iter()):
        warnings.warn('The requested group_to_flatten is not a '
                      'descendant of root')
        # We will shortcut here, because it is impossible for any paths
        # to be returned anyhow.
        return []

    # We create a set of the unique IDs of each element that we wish to
    # flatten, if those elements are groups. Any groups outside of this
    # set will be skipped while we flatten the paths.
    desired_groups = set()
    if recursive:
        for group in group_to_flatten.iter():
            desired_groups.add(id(group))
    else:
        desired_groups.add(id(group_to_flatten))

    def desired_group_filter(x):
        return (id(x) in desired_groups) and group_filter(x)

    return flatten_all_paths(root, desired_group_filter, path_filter,
                             path_conversions, group_search_xpath)


class Document:
    def __init__(self, filename):
        """A container for a DOM-style SVG document.

        The `Document` class provides a simple interface to modify and analyze 
        the path elements in a DOM-style document.  The DOM-style document is 
        parsed into an ElementTree object (stored in the `tree` attribute).

        This class provides functions for extracting SVG data into Path objects.
        The Path output objects will be transformed based on their parent groups.
        
        Args:
            filename (str): The filename of the DOM-style object.
        """

        # remember location of original svg file
        if filename is not None and os.path.dirname(filename) == '':
            self.original_filename = os.path.join(os.getcwd(), filename)
        else:
            self.original_filename = filename

        if filename is not None:
            # parse svg to ElementTree object
            self.tree = etree.parse(filename)
        else:
            self.tree = etree.ElementTree(Element('svg'))

        self.root = self.tree.getroot()

    def flatten_all_paths(self, group_filter=lambda x: True,
                          path_filter=lambda x: True,
                          path_conversions=CONVERSIONS):
        """Forward the tree of this document into the more general
        flatten_all_paths function and return the result."""
        return flatten_all_paths(self.tree.getroot(), group_filter,
                                 path_filter, path_conversions)

    def flatten_group(self, group, recursive=True, group_filter=lambda x: True,
                      path_filter=lambda x: True, path_conversions=CONVERSIONS):
        if all(isinstance(s, str) for s in group):
            # If we're given a list of strings, assume it represents a
            # nested sequence
            group = self.get_or_add_group(group)
        elif not isinstance(group, Element):
            raise TypeError(
                'Must provide a list of strings that represent a nested '
                'group name, or provide an xml.etree.Element object. '
                'Instead you provided {0}'.format(group))

        return flatten_group(group, self.tree.getroot(), recursive,
                             group_filter, path_filter, path_conversions)

    def add_path(self, path, attribs=None, group=None):
        """Add a new path to the SVG."""

        # If not given a parent, assume that the path does not have a group
        if group is None:
            group = self.tree.getroot()

        # If given a list of strings (one or more), assume it represents
        # a sequence of nested group names
        elif all(isinstance(elem, str) for elem in group):
            group = self.get_or_add_group(group)

        elif not isinstance(group, Element):
            raise TypeError(
                'Must provide a list of strings or an xml.etree.Element '
                'object. Instead you provided {0}'.format(group))

        else:
            # Make sure that the group belongs to this Document object
            if not self.contains_group(group):
                warnings.warn('The requested group does not belong to '
                              'this Document')

        # TODO: It might be better to use duck-typing here with a try-except
        if isinstance(path, Path):
            path_svg = path.d()
        elif is_path_segment(path):
            path_svg = Path(path).d()
        elif isinstance(path, str):
            # Assume this is a valid d-string.
            # TODO: Should we sanity check the input string?
            path_svg = path
        else:
            raise TypeError(
                'Must provide a Path, a path segment type, or a valid '
                'SVG path d-string. Instead you provided {0}'.format(path))

        if attribs is None:
            attribs = {}
        else:
            attribs = attribs.copy()

        attribs['d'] = path_svg

        return SubElement(group, 'path', attribs)

    def contains_group(self, group):
        return any(group is owned for owned in self.tree.iter())

    def get_or_add_group(self, nested_names, name_attr='id'):
        """Get a group from the tree, or add a new one with the given
        name structure.

        `nested_names` is a list of strings which represent group names.
        Each group name will be nested inside of the previous group name.

        `name_attr` is the group attribute that is being used to
        represent the group's name. Default is 'id', but some SVGs may
        contain custom name labels, like 'inkscape:label'.

        Returns the requested group. If the requested group did not
        exist, this function will create it, as well as all parent
        groups that it requires. All created groups will be left with
        blank attributes.

        """
        group = self.tree.getroot()
        # Drill down through the names until we find the desired group
        while len(nested_names):
            prev_group = group
            next_name = nested_names.pop(0)
            for elem in group.iterfind(SVG_GROUP_TAG, SVG_NAMESPACE):
                if elem.get(name_attr) == next_name:
                    group = elem
                    break

            if prev_group is group:
                # The group we're looking for does not exist, so let's
                # create the group structure
                nested_names.insert(0, next_name)

                while nested_names:
                    next_name = nested_names.pop(0)
                    group = self.add_group({'id': next_name}, group)
                # Now nested_names will be empty, so the topmost
                # while-loop will end
        return group

    def add_group(self, group_attribs=None, parent=None):
        """Add an empty group element to the SVG."""
        if parent is None:
            parent = self.tree.getroot()
        elif not self.contains_group(parent):
            warnings.warn('The requested group {0} does not belong to '
                          'this Document'.format(parent))

        if group_attribs is None:
            group_attribs = {}
        else:
            group_attribs = group_attribs.copy()

        return SubElement(parent, '{{{0}}}g'.format(
            SVG_NAMESPACE['svg']), group_attribs)

    def save(self, filename):
        with open(filename, 'w') as output_svg:
            output_svg.write(etree.tostring(self.tree.getroot()))

    def display(self, filename=None):
        """Displays/opens the doc using the OS's default application."""

        if filename is None:
            filename = self.original_filename

        # write to a (by default temporary) file
        with open(filename, 'w') as output_svg:
            output_svg.write(etree.tostring(self.tree.getroot()))

        open_in_browser(filename)

"""This submodule contains miscellaneous tools that are used internally, but
aren't specific to SVGs or related mathematical objects."""

# External dependencies:
from __future__ import division, absolute_import, print_function
import os
import sys
import webbrowser


# stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
def hex2rgb(value):
    """Converts a hexadeximal color string to an RGB 3-tuple

    EXAMPLE
    -------
    >>> hex2rgb('#0000FF')
    (0, 0, 255)
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))


# stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
def rgb2hex(rgb):
    """Converts an RGB 3-tuple to a hexadeximal color string.

    EXAMPLE
    -------
    >>> rgb2hex((0,0,255))
    '#0000FF'
    """
    return ('#%02x%02x%02x' % tuple(rgb)).upper()


def isclose(a, b, rtol=1e-5, atol=1e-8):
    """This is essentially np.isclose, but slightly faster."""
    return abs(a - b) < (atol + rtol * abs(b))


def open_in_browser(file_location):
    """Attempt to open file located at file_location in the default web
    browser."""

    # If just the name of the file was given, check if it's in the Current
    # Working Directory.
    if not os.path.isfile(file_location):
        file_location = os.path.join(os.getcwd(), file_location)
    if not os.path.isfile(file_location):
        raise IOError("\n\nFile not found.")

    #  For some reason OSX requires this adjustment (tested on 10.10.4)
    if sys.platform == "darwin":
        file_location = "file:///"+file_location

    new = 2  # open in a new tab, if possible
    webbrowser.get().open(file_location, new=new)


BugException = Exception("This code should never be reached.  You've found a "
                         "bug.  Please submit an issue to \n"
                         "https://github.com/mathandy/svgpathtools/issues"
                         "\nwith an easily reproducible example.")

"""This submodule contains the path_parse() function used to convert SVG path
element d-strings into svgpathtools Path objects.
Note: This file was taken (nearly) as is from the svg.path module (v 2.0)."""

# External dependencies
from __future__ import division, absolute_import, print_function
import re
import numpy as np
import warnings

# Internal dependencies
from .path import Path, Line, QuadraticBezier, CubicBezier, Arc

# To maintain forward/backward compatibility
try:
    str = basestring
except NameError:
    pass

COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
UPPERCASE = set('MZLHVCSQTA')

COMMAND_RE = re.compile("([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile("[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def _tokenize_path(pathdef):
    for x in COMMAND_RE.split(pathdef):
        if x in COMMANDS:
            yield x
        for token in FLOAT_RE.findall(x):
            yield token


def parse_path(pathdef, current_pos=0j, tree_element=None):
    # In the SVG specs, initial movetos are absolute, even if
    # specified as 'm'. This is the default behavior here as well.
    # But if you pass in a current_pos variable, the initial moveto
    # will be relative to that current_pos. This is useful.
    elements = list(_tokenize_path(pathdef))
    # Reverse for easy use of .pop()
    elements.reverse()

    if tree_element is None:
        segments = Path()
    else:
        segments = Path(tree_element=tree_element)

    start_pos = None
    command = None

    while elements:

        if elements[-1] in COMMANDS:
            # New command.
            last_command = command  # Used by S and T
            command = elements.pop()
            absolute = command in UPPERCASE
            command = command.upper()
        else:
            # If this element starts with numbers, it is an implicit command
            # and we don't change the command. Check that it's allowed:
            if command is None:
                raise ValueError("Unallowed implicit command in %s, position %s" % (
                    pathdef, len(pathdef.split()) - len(elements)))

        if command == 'M':
            # Moveto command.
            x = elements.pop()
            y = elements.pop()
            pos = float(x) + float(y) * 1j
            if absolute:
                current_pos = pos
            else:
                current_pos += pos

            # when M is called, reset start_pos
            # This behavior of Z is defined in svg spec:
            # http://www.w3.org/TR/SVG/paths.html#PathDataClosePathCommand
            start_pos = current_pos

            # Implicit moveto commands are treated as lineto commands.
            # So we set command to lineto here, in case there are
            # further implicit commands after this moveto.
            command = 'L'

        elif command == 'Z':
            # Close path
            if not (current_pos == start_pos):
                segments.append(Line(current_pos, start_pos))
            segments.closed = True
            current_pos = start_pos
            command = None

        elif command == 'L':
            x = elements.pop()
            y = elements.pop()
            pos = float(x) + float(y) * 1j
            if not absolute:
                pos += current_pos
            segments.append(Line(current_pos, pos))
            current_pos = pos

        elif command == 'H':
            x = elements.pop()
            pos = float(x) + current_pos.imag * 1j
            if not absolute:
                pos += current_pos.real
            segments.append(Line(current_pos, pos))
            current_pos = pos

        elif command == 'V':
            y = elements.pop()
            pos = current_pos.real + float(y) * 1j
            if not absolute:
                pos += current_pos.imag * 1j
            segments.append(Line(current_pos, pos))
            current_pos = pos

        elif command == 'C':
            control1 = float(elements.pop()) + float(elements.pop()) * 1j
            control2 = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                control1 += current_pos
                control2 += current_pos
                end += current_pos

            segments.append(CubicBezier(current_pos, control1, control2, end))
            current_pos = end

        elif command == 'S':
            # Smooth curve. First control point is the "reflection" of
            # the second control point in the previous path.

            if last_command not in 'CS':
                # If there is no previous command or if the previous command
                # was not an C, c, S or s, assume the first control point is
                # coincident with the current point.
                control1 = current_pos
            else:
                # The first control point is assumed to be the reflection of
                # the second control point on the previous command relative
                # to the current point.
                control1 = current_pos + current_pos - segments[-1].control2

            control2 = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                control2 += current_pos
                end += current_pos

            segments.append(CubicBezier(current_pos, control1, control2, end))
            current_pos = end

        elif command == 'Q':
            control = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                control += current_pos
                end += current_pos

            segments.append(QuadraticBezier(current_pos, control, end))
            current_pos = end

        elif command == 'T':
            # Smooth curve. Control point is the "reflection" of
            # the second control point in the previous path.

            if last_command not in 'QT':
                # If there is no previous command or if the previous command
                # was not an Q, q, T or t, assume the first control point is
                # coincident with the current point.
                control = current_pos
            else:
                # The control point is assumed to be the reflection of
                # the control point on the previous command relative
                # to the current point.
                control = current_pos + current_pos - segments[-1].control

            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                end += current_pos

            segments.append(QuadraticBezier(current_pos, control, end))
            current_pos = end

        elif command == 'A':
            radius = float(elements.pop()) + float(elements.pop()) * 1j
            rotation = float(elements.pop())
            arc = float(elements.pop())
            sweep = float(elements.pop())
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                end += current_pos

            segments.append(Arc(current_pos, radius, rotation, arc, sweep, end))
            current_pos = end

    return segments


def _check_num_parsed_values(values, allowed):
    if not any(num == len(values) for num in allowed):
        if len(allowed) > 1:
            warnings.warn('Expected one of the following number of values {0}, but found {1} values instead: {2}'
                          .format(allowed, len(values), values))
        elif allowed[0] != 1:
            warnings.warn('Expected {0} values, found {1}: {2}'.format(allowed[0], len(values), values))
        else:
            warnings.warn('Expected 1 value, found {0}: {1}'.format(len(values), values))
        return False
    return True


def _parse_transform_substr(transform_substr):

    type_str, value_str = transform_substr.split('(')
    value_str = value_str.replace(',', ' ')
    values = list(map(float, filter(None, value_str.split(' '))))

    transform = np.identity(3)
    if 'matrix' in type_str:
        if not _check_num_parsed_values(values, [6]):
            return transform

        transform[0:2, 0:3] = np.array([values[0:6:2], values[1:6:2]])

    elif 'translate' in transform_substr:
        if not _check_num_parsed_values(values, [1, 2]):
            return transform

        transform[0, 2] = values[0]
        if len(values) > 1:
            transform[1, 2] = values[1]

    elif 'scale' in transform_substr:
        if not _check_num_parsed_values(values, [1, 2]):
            return transform

        x_scale = values[0]
        y_scale = values[1] if (len(values) > 1) else x_scale
        transform[0, 0] = x_scale
        transform[1, 1] = y_scale

    elif 'rotate' in transform_substr:
        if not _check_num_parsed_values(values, [1, 3]):
            return transform

        angle = values[0] * np.pi / 180.0
        if len(values) == 3:
            offset = values[1:3]
        else:
            offset = (0, 0)
        tf_offset = np.identity(3)
        tf_offset[0:2, 2:3] = np.array([[offset[0]], [offset[1]]])
        tf_rotate = np.identity(3)
        tf_rotate[0:2, 0:2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        tf_offset_neg = np.identity(3)
        tf_offset_neg[0:2, 2:3] = np.array([[-offset[0]], [-offset[1]]])

        transform = tf_offset.dot(tf_rotate).dot(tf_offset_neg)

    elif 'skewX' in transform_substr:
        if not _check_num_parsed_values(values, [1]):
            return transform

        transform[0, 1] = np.tan(values[0] * np.pi / 180.0)

    elif 'skewY' in transform_substr:
        if not _check_num_parsed_values(values, [1]):
            return transform

        transform[1, 0] = np.tan(values[0] * np.pi / 180.0)
    else:
        # Return an identity matrix if the type of transform is unknown, and warn the user
        warnings.warn('Unknown SVG transform type: {0}'.format(type_str))

    return transform


def parse_transform(transform_str):
    """Converts a valid SVG transformation string into a 3x3 matrix.
    If the string is empty or null, this returns a 3x3 identity matrix"""
    if not transform_str:
        return np.identity(3)
    elif not isinstance(transform_str, str):
        raise TypeError('Must provide a string to parse')

    total_transform = np.identity(3)
    transform_substrs = transform_str.split(')')[:-1]  # Skip the last element, because it should be empty
    for substr in transform_substrs:
        total_transform = total_transform.dot(_parse_transform_substr(substr))

    return total_transform

"""This submodule contains the class definitions of the the main five classes
svgpathtools is built around: Path, Line, QuadraticBezier, CubicBezier, and
Arc."""

# External dependencies
from __future__ import division, absolute_import, print_function
from math import sqrt, cos, sin, acos, asin, degrees, radians, log, pi, ceil
from cmath import exp, sqrt as csqrt, phase
from collections import MutableSequence
from warnings import warn
from operator import itemgetter
import numpy as np
try:
    from scipy.integrate import quad
    _quad_available = True
except:
    _quad_available = False

# Internal dependencies
from .bezier import (bezier_intersections, bezier_bounding_box, split_bezier,
                     bezier_by_line_intersections, polynomial2bezier,
                     bezier2polynomial)
from .misctools import BugException
from .polytools import rational_limit, polyroots, polyroots01, imag, real


# Default Parameters ##########################################################

# path segment .length() parameters for arc length computation
LENGTH_MIN_DEPTH = 5
LENGTH_ERROR = 1e-12
USE_SCIPY_QUAD = True  # for elliptic Arc segment arc length computation

# path segment .ilength() parameters for inverse arc length computation
ILENGTH_MIN_DEPTH = 5
ILENGTH_ERROR = 1e-12
ILENGTH_S_TOL = 1e-12
ILENGTH_MAXITS = 10000

# compatibility/implementation related warnings and parameters
CLOSED_WARNING_ON = True
_NotImplemented4ArcException = \
    Exception("This method has not yet been implemented for Arc objects.")
# _NotImplemented4QuadraticException = \
#     Exception("This method has not yet been implemented for QuadraticBezier "
#               "objects.")
_is_smooth_from_warning = \
    ("The name of this method is somewhat misleading (yet kept for "
     "compatibility with scripts created using svg.path 2.0).  This method "
     "is meant only for d-string creation and should NOT be used to check "
     "for kinks.  To check a segment for differentiability, use the "
     "joins_smoothly_with() method instead or the kinks() function (in "
     "smoothing.py).\nTo turn off this warning, set "
     "warning_on=False.")


# Miscellaneous ###############################################################

def bezier_segment(*bpoints):
    if len(bpoints) == 2:
        return Line(*bpoints)
    elif len(bpoints) == 4:
        return CubicBezier(*bpoints)
    elif len(bpoints) == 3:
        return QuadraticBezier(*bpoints)
    else:
        assert len(bpoints) in (2, 3, 4)


def is_bezier_segment(seg):
    return (isinstance(seg, Line) or
            isinstance(seg, QuadraticBezier) or
            isinstance(seg, CubicBezier))


def is_path_segment(seg):
    return is_bezier_segment(seg) or isinstance(seg, Arc)


def is_bezier_path(path):
    """Checks that all segments in path are a Line, QuadraticBezier, or
    CubicBezier object."""
    return isinstance(path, Path) and all(map(is_bezier_segment, path))


def concatpaths(list_of_paths):
    """Takes in a sequence of paths and returns their concatenations into a
    single path (following the order of the input sequence)."""
    return Path(*[seg for path in list_of_paths for seg in path])


def bbox2path(xmin, xmax, ymin, ymax):
    """Converts a bounding box 4-tuple to a Path object."""
    b = Line(xmin + 1j*ymin, xmax + 1j*ymin)
    t = Line(xmin + 1j*ymax, xmax + 1j*ymax)
    r = Line(xmax + 1j*ymin, xmax + 1j*ymax)
    l = Line(xmin + 1j*ymin, xmin + 1j*ymax)
    return Path(b, r, t.reversed(), l.reversed())


def polyline(*points):
    """Converts a list of points to a Path composed of lines connecting those 
    points (i.e. a linear spline or polyline).  See also `polygon()`."""
    return Path(*[Line(points[i], points[i+1])
                  for i in range(len(points) - 1)])


def polygon(*points):
    """Converts a list of points to a Path composed of lines connecting those 
    points, then closes the path by connecting the last point to the first.  
    See also `polyline()`."""
    return Path(*[Line(points[i], points[(i + 1) % len(points)])
                  for i in range(len(points))])


# Conversion###################################################################

def bpoints2bezier(bpoints):
    """Converts a list of length 2, 3, or 4 to a CubicBezier, QuadraticBezier,
    or Line object, respectively.
    See also: poly2bez."""
    order = len(bpoints) - 1
    if order == 3:
        return CubicBezier(*bpoints)
    elif order == 2:
        return QuadraticBezier(*bpoints)
    elif order == 1:
        return Line(*bpoints)
    else:
        assert len(bpoints) in {2, 3, 4}


def poly2bez(poly, return_bpoints=False):
    """Converts a cubic or lower order Polynomial object (or a sequence of
    coefficients) to a CubicBezier, QuadraticBezier, or Line object as
    appropriate.  If return_bpoints=True then this will instead only return
    the control points of the corresponding Bezier curve.
    Note: The inverse operation is available as a method of CubicBezier,
    QuadraticBezier and Line objects."""
    bpoints = polynomial2bezier(poly)
    if return_bpoints:
        return bpoints
    else:
        return bpoints2bezier(bpoints)


def bez2poly(bez, numpy_ordering=True, return_poly1d=False):
    """Converts a Bezier object or tuple of Bezier control points to a tuple
    of coefficients of the expanded polynomial.
    return_poly1d : returns a numpy.poly1d object.  This makes computations
    of derivatives/anti-derivatives and many other operations quite quick.
    numpy_ordering : By default (to accommodate numpy) the coefficients will
    be output in reverse standard order.
    Note:  This function is redundant thanks to the .poly() method included
    with all bezier segment classes."""
    if is_bezier_segment(bez):
        bez = bez.bpoints()
    return bezier2polynomial(bez,
                             numpy_ordering=numpy_ordering,
                             return_poly1d=return_poly1d)


# Geometric####################################################################

def rotate(curve, degs, origin=None):
    """Returns curve rotated by `degs` degrees (CCW) around the point `origin`
    (a complex number).  By default origin is either `curve.point(0.5)`, or in
    the case that curve is an Arc object, `origin` defaults to `curve.center`.
    """
    def transform(z):
        return exp(1j*radians(degs))*(z - origin) + origin

    if origin is None:
        if isinstance(curve, Arc):
            origin = curve.center
        else:
            origin = curve.point(0.5)

    if isinstance(curve, Path):
        return Path(*[rotate(seg, degs, origin=origin) for seg in curve])
    elif is_bezier_segment(curve):
        return bpoints2bezier([transform(bpt) for bpt in curve.bpoints()])
    elif isinstance(curve, Arc):
        new_start = transform(curve.start)
        new_end = transform(curve.end)
        new_rotation = curve.rotation + degs
        return Arc(new_start, radius=curve.radius, rotation=new_rotation,
                   large_arc=curve.large_arc, sweep=curve.sweep, end=new_end)
    else:
        raise TypeError("Input `curve` should be a Path, Line, "
                        "QuadraticBezier, CubicBezier, or Arc object.")


def translate(curve, z0):
    """Shifts the curve by the complex quantity z such that
    translate(curve, z0).point(t) = curve.point(t) + z0"""
    if isinstance(curve, Path):
        return Path(*[translate(seg, z0) for seg in curve])
    elif is_bezier_segment(curve):
        return bpoints2bezier([bpt + z0 for bpt in curve.bpoints()])
    elif isinstance(curve, Arc):
        new_start = curve.start + z0
        new_end = curve.end + z0
        return Arc(new_start, radius=curve.radius, rotation=curve.rotation,
                   large_arc=curve.large_arc, sweep=curve.sweep, end=new_end)
    else:
        raise TypeError("Input `curve` should be a Path, Line, "
                        "QuadraticBezier, CubicBezier, or Arc object.")


def scale(curve, sx, sy=None, origin=0j):
    """Scales `curve`, about `origin`, by diagonal matrix `[[sx,0],[0,sy]]`.

    Notes:
    ------
    * If `sy` is not specified, it is assumed to be equal to `sx` and 
    a scalar transformation of `curve` about `origin` will be returned.
    I.e.
        scale(curve, sx, origin).point(t) == 
            ((curve.point(t) - origin) * sx) + origin
    """

    if sy is None:
        isy = 1j*sx
    else:
        isy = 1j*sy

    def _scale(z):
        if sy is None:
            return sx*z
        return sx*z.real + isy*z.imag          

    def scale_bezier(bez):
        p = [_scale(c) for c in bez2poly(bez)]
        p[-1] += origin - _scale(origin)
        return poly2bez(p)

    if isinstance(curve, Path):
        return Path(*[scale(seg, sx, sy, origin) for seg in curve])
    elif is_bezier_segment(curve):
        return scale_bezier(curve)
    elif isinstance(curve, Arc):
        if sy is None or sy == sx:
            return Arc(start=sx*(curve.start - origin) + origin,
                       radius=sx*curve.radius,
                       rotation=curve.rotation, 
                       large_arc=curve.large_arc, 
                       sweep=curve.sweep, 
                       end=sx*(curve.end - origin) + origin)
        else:
            raise Exception("\nFor `Arc` objects, only scale transforms "
                            "with sx==sy are implemented.\n")
    else:
        raise TypeError("Input `curve` should be a Path, Line, "
                        "QuadraticBezier, CubicBezier, or Arc object.")


def transform(curve, tf):
    """Transforms the curve by the homogeneous transformation matrix tf"""
    def to_point(p):
        return np.array([[p.real], [p.imag], [1.0]])

    def to_vector(z):
        return np.array([[z.real], [z.imag], [0.0]])

    def to_complex(v):
        return v.item(0) + 1j * v.item(1)

    if isinstance(curve, Path):
        return Path(*[transform(segment, tf) for segment in curve])
    elif is_bezier_segment(curve):
        return bpoints2bezier([to_complex(tf.dot(to_point(p)))
                               for p in curve.bpoints()])
    elif isinstance(curve, Arc):
        new_start = to_complex(tf.dot(to_point(curve.start)))
        new_end = to_complex(tf.dot(to_point(curve.end)))
        new_radius = to_complex(tf.dot(to_vector(curve.radius)))
        return Arc(new_start, radius=new_radius, rotation=curve.rotation,
                   large_arc=curve.large_arc, sweep=curve.sweep, end=new_end)
    else:
        raise TypeError("Input `curve` should be a Path, Line, "
                        "QuadraticBezier, CubicBezier, or Arc object.")


def bezier_unit_tangent(seg, t):
    """Returns the unit tangent of the segment at t.

    Notes
    -----
    If you receive a RuntimeWarning, try the following:
    >>> import numpy
    >>> old_numpy_error_settings = numpy.seterr(invalid='raise')
    This can be undone with:
    >>> numpy.seterr(**old_numpy_error_settings)
    """
    assert 0 <= t <= 1
    dseg = seg.derivative(t)

    # Note: dseg might be numpy value, use np.seterr(invalid='raise')
    try:
        unit_tangent = dseg/abs(dseg)
    except (ZeroDivisionError, FloatingPointError):
        # This may be a removable singularity, if so we just need to compute
        # the limit.
        # Note: limit{{dseg / abs(dseg)} = sqrt(limit{dseg**2 / abs(dseg)**2})
        dseg_poly = seg.poly().deriv()
        dseg_abs_squared_poly = (real(dseg_poly) ** 2 +
                                 imag(dseg_poly) ** 2)
        try:
            unit_tangent = csqrt(rational_limit(dseg_poly**2,
                                            dseg_abs_squared_poly, t))
        except ValueError:
            bef = seg.poly().deriv()(t - 1e-4)
            aft = seg.poly().deriv()(t + 1e-4)
            mes = ("Unit tangent appears to not be well-defined at "
                   "t = {}, \n".format(t) +
                   "seg.poly().deriv()(t - 1e-4) = {}\n".format(bef) +
                   "seg.poly().deriv()(t + 1e-4) = {}".format(aft))
            raise ValueError(mes)
    return unit_tangent


def segment_curvature(self, t, use_inf=False):
    """returns the curvature of the segment at t.

    Notes
    -----
    If you receive a RuntimeWarning, run command
    >>> old = np.seterr(invalid='raise')
    This can be undone with
    >>> np.seterr(**old)
    """

    dz = self.derivative(t)
    ddz = self.derivative(t, n=2)
    dx, dy = dz.real, dz.imag
    ddx, ddy = ddz.real, ddz.imag
    old_np_seterr = np.seterr(invalid='raise')
    try:
        kappa = abs(dx*ddy - dy*ddx)/sqrt(dx*dx + dy*dy)**3
    except (ZeroDivisionError, FloatingPointError):
        # tangent vector is zero at t, use polytools to find limit
        p = self.poly()
        dp = p.deriv()
        ddp = dp.deriv()
        dx, dy = real(dp), imag(dp)
        ddx, ddy = real(ddp), imag(ddp)
        f2 = (dx*ddy - dy*ddx)**2
        g2 = (dx*dx + dy*dy)**3
        lim2 = rational_limit(f2, g2, t)
        if lim2 < 0:  # impossible, must be numerical error
            return 0
        kappa = sqrt(lim2)
    finally:
        np.seterr(**old_np_seterr)
    return kappa


def bezier_radialrange(seg, origin, return_all_global_extrema=False):
    """returns the tuples (d_min, t_min) and (d_max, t_max) which minimize and
    maximize, respectively, the distance d = |self.point(t)-origin|.
    return_all_global_extrema:  Multiple such t_min or t_max values can exist.
    By default, this will only return one. Set return_all_global_extrema=True
    to return all such global extrema."""

    def _radius(tau):
        return abs(seg.point(tau) - origin)

    shifted_seg_poly = seg.poly() - origin
    r_squared = real(shifted_seg_poly) ** 2 + \
                imag(shifted_seg_poly) ** 2
    extremizers = [0, 1] + polyroots01(r_squared.deriv())
    extrema = [(_radius(t), t) for t in extremizers]

    if return_all_global_extrema:
        raise NotImplementedError
    else:
        seg_global_min = min(extrema, key=itemgetter(0))
        seg_global_max = max(extrema, key=itemgetter(0))
        return seg_global_min, seg_global_max


def closest_point_in_path(pt, path):
    """returns (|path.seg.point(t)-pt|, t, seg_idx) where t and seg_idx
    minimize the distance between pt and curve path[idx].point(t) for 0<=t<=1
    and any seg_idx.
    Warning:  Multiple such global minima can exist.  This will only return
    one."""
    return path.radialrange(pt)[0]


def farthest_point_in_path(pt, path):
    """returns (|path.seg.point(t)-pt|, t, seg_idx) where t and seg_idx
    maximize the distance between pt and curve path[idx].point(t) for 0<=t<=1
    and any seg_idx.
    :rtype : object
    :param pt:
    :param path:
    Warning:  Multiple such global maxima can exist.  This will only return
    one."""
    return path.radialrange(pt)[1]


def path_encloses_pt(pt, opt, path):
    """returns true if pt is a point enclosed by path (which must be a Path
    object satisfying path.isclosed==True).  opt is a point you know is
    NOT enclosed by path."""
    assert path.isclosed()
    intersections = Path(Line(pt, opt)).intersect(path)
    if len(intersections) % 2:
        return True
    else:
        return False


def segment_length(curve, start, end, start_point, end_point,
                   error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH, depth=0):
    """Recursively approximates the length by straight lines"""
    mid = (start + end)/2
    mid_point = curve.point(mid)
    length = abs(end_point - start_point)
    first_half = abs(mid_point - start_point)
    second_half = abs(end_point - mid_point)

    length2 = first_half + second_half
    if (length2 - length > error) or (depth < min_depth):
        # Calculate the length of each segment:
        depth += 1
        return (segment_length(curve, start, mid, start_point, mid_point,
                               error, min_depth, depth) +
                segment_length(curve, mid, end, mid_point, end_point,
                               error, min_depth, depth))
    # This is accurate enough.
    return length2


def inv_arclength(curve, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                  error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
    """INPUT: curve should be a CubicBezier, Line, of Path of CubicBezier
    and/or Line objects.
    OUTPUT: Returns a float, t, such that the arc length of curve from 0 to
    t is approximately s.
    s_tol - exit when |s(t) - s| < s_tol where
        s(t) = seg.length(0, t, error, min_depth) and seg is either curve or,
        if curve is a Path object, then seg is a segment in curve.
    error - used to compute lengths of cubics and arcs
    min_depth - used to compute lengths of cubics and arcs
    Note:  This function is not designed to be efficient, but if it's slower
    than you need, make sure you have scipy installed."""

    curve_length = curve.length(error=error, min_depth=min_depth)
    assert curve_length > 0
    if not 0 <= s <= curve_length:
        raise ValueError("s is not in interval [0, curve.length()].")

    if s == 0:
        return 0
    if s == curve_length:
        return 1

    if isinstance(curve, Path):
        seg_lengths = [seg.length(error=error, min_depth=min_depth)
                       for seg in curve]
        lsum = 0
        # Find which segment the point we search for is located on
        for k, len_k in enumerate(seg_lengths):
            if lsum <= s <= lsum + len_k:
                t = inv_arclength(curve[k], s - lsum, s_tol=s_tol,
                                  maxits=maxits, error=error,
                                  min_depth=min_depth)
                return curve.t2T(k, t)
            lsum += len_k
        return 1

    elif isinstance(curve, Line):
        return s / curve.length(error=error, min_depth=min_depth)

    elif (isinstance(curve, QuadraticBezier) or
          isinstance(curve, CubicBezier) or
          isinstance(curve, Arc)):
        t_upper = 1
        t_lower = 0
        iteration = 0
        while iteration < maxits:
            iteration += 1
            t = (t_lower + t_upper)/2
            s_t = curve.length(t1=t, error=error, min_depth=min_depth)
            if abs(s_t - s) < s_tol:
                return t
            elif s_t < s:  # t too small
                t_lower = t
            else:  # s < s_t, t too big
                t_upper = t
            if t_upper == t_lower:
                warn("t is as close as a float can be to the correct value, "
                     "but |s(t) - s| = {} > s_tol".format(abs(s_t-s)))
                return t
        raise Exception("Maximum iterations reached with s(t) - s = {}."
                        "".format(s_t - s))
    else:
        raise TypeError("First argument must be a Line, QuadraticBezier, "
                        "CubicBezier, Arc, or Path object.")

# Operations###################################################################


def crop_bezier(seg, t0, t1):
    """returns a cropped copy of this segment which starts at self.point(t0)
    and ends at self.point(t1)."""
    assert t0 < t1
    if t0 == 0:
        cropped_seg = seg.split(t1)[0]
    elif t1 == 1:
        cropped_seg = seg.split(t0)[1]
    else:
        pt1 = seg.point(t1)

        # trim off the 0 <= t < t0 part
        trimmed_seg = crop_bezier(seg, t0, 1)

        # find the adjusted t1 (i.e. the t1 such that
        # trimmed_seg.point(t1) ~= pt))and trim off the t1 < t <= 1 part
        t1_adj = trimmed_seg.radialrange(pt1)[0][1]
        cropped_seg = crop_bezier(trimmed_seg, 0, t1_adj)
    return cropped_seg


# Main Classes ################################################################


class Line(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return 'Line(start=%s, end=%s)' % (self.start, self.end)

    def __eq__(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        return not self == other

    def __getitem__(self, item):
        return self.bpoints()[item]

    def __len__(self):
        return 2

    def joins_smoothly_with(self, previous, wrt_parameterization=False):
        """Checks if this segment joins smoothly with previous segment.  By
        default, this only checks that this segment starts moving (at t=0) in
        the same direction (and from the same positive) as previous stopped
        moving (at t=1).  To check if the tangent magnitudes also match, set
        wrt_parameterization=True."""
        if wrt_parameterization:
            return self.start == previous.end and np.isclose(
                self.derivative(0), previous.derivative(1))
        else:
            return self.start == previous.end and np.isclose(
                self.unit_tangent(0), previous.unit_tangent(1))

    def point(self, t):
        """returns the coordinates of the Bezier curve evaluated at t."""
        distance = self.end - self.start
        return self.start + distance*t

    def length(self, t0=0, t1=1, error=None, min_depth=None):
        """returns the length of the line segment between t0 and t1."""
        return abs(self.end - self.start)*(t1-t0)

    def ilength(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns a float, t, such that self.length(0, t) is approximately s.
        See the inv_arclength() docstring for more details."""
        return inv_arclength(self, s, s_tol=s_tol, maxits=maxits, error=error,
                             min_depth=min_depth)

    def bpoints(self):
        """returns the Bezier control points of the segment."""
        return self.start, self.end

    def poly(self, return_coeffs=False):
        """returns the line as a Polynomial object."""
        p = self.bpoints()
        coeffs = ([p[1] - p[0], p[0]])
        if return_coeffs:
            return coeffs
        else:
            return np.poly1d(coeffs)

    def derivative(self, t=None, n=1):
        """returns the nth derivative of the segment at t."""
        assert self.end != self.start
        if n == 1:
            return self.end - self.start
        elif n > 1:
            return 0
        else:
            raise ValueError("n should be a positive integer.")

    def unit_tangent(self, t=None):
        """returns the unit tangent of the segment at t."""
        assert self.end != self.start
        dseg = self.end - self.start
        return dseg/abs(dseg)

    def normal(self, t=None):
        """returns the (right hand rule) unit normal vector to self at t."""
        return -1j*self.unit_tangent(t)

    def curvature(self, t):
        """returns the curvature of the line, which is always zero."""
        return 0

    # def icurvature(self, kappa):
    #     """returns a list of t-values such that 0 <= t<= 1 and
    #     seg.curvature(t) = kappa."""
    #     if kappa:
    #         raise ValueError("The .icurvature() method for Line elements will "
    #                          "return an empty list if kappa is nonzero and "
    #                          "will raise this exception when kappa is zero as "
    #                          "this is true at every point on the line.")
    #     return []

    def reversed(self):
        """returns a copy of the Line object with its orientation reversed."""
        return Line(self.end, self.start)

    def intersect(self, other_seg, tol=None):
        """Finds the intersections of two segments.
        returns a list of tuples (t1, t2) such that
        self.point(t1) == other_seg.point(t2).
        Note: This will fail if the two segments coincide for more than a
        finite collection of points.
        tol is not used."""
        if isinstance(other_seg, Line):
            assert other_seg.end != other_seg.start and self.end != self.start
            assert self != other_seg
            # Solve the system [p1-p0, q1-q0]*[t1, t2]^T = q0 - p0
            # where self == Line(p0, p1) and other_seg == Line(q0, q1)
            a = (self.start.real, self.end.real)
            b = (self.start.imag, self.end.imag)
            c = (other_seg.start.real, other_seg.end.real)
            d = (other_seg.start.imag, other_seg.end.imag)
            denom = ((a[1] - a[0])*(d[0] - d[1]) -
                     (b[1] - b[0])*(c[0] - c[1]))
            if np.isclose(denom, 0):
                return []
            t1 = (c[0]*(b[0] - d[1]) -
                  c[1]*(b[0] - d[0]) -
                  a[0]*(d[0] - d[1]))/denom
            t2 = -(a[1]*(b[0] - d[0]) -
                   a[0]*(b[1] - d[0]) -
                   c[0]*(b[0] - b[1]))/denom
            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                return [(t1, t2)]
            return []
        elif isinstance(other_seg, QuadraticBezier):
            t2t1s = bezier_by_line_intersections(other_seg, self)
            return [(t1, t2) for t2, t1 in t2t1s]
        elif isinstance(other_seg, CubicBezier):
            t2t1s = bezier_by_line_intersections(other_seg, self)
            return [(t1, t2) for t2, t1 in t2t1s]
        elif isinstance(other_seg, Arc):
            t2t1s = other_seg.intersect(self)
            return [(t1, t2) for t2, t1 in t2t1s]
        elif isinstance(other_seg, Path):
            raise TypeError(
                "other_seg must be a path segment, not a Path object, use "
                "Path.intersect().")
        else:
            raise TypeError("other_seg must be a path segment.")

    def bbox(self):
        """returns the bounding box for the segment in the form
        (xmin, xmax, ymin, ymax)."""
        xmin = min(self.start.real, self.end.real)
        xmax = max(self.start.real, self.end.real)
        ymin = min(self.start.imag, self.end.imag)
        ymax = max(self.start.imag, self.end.imag)
        return xmin, xmax, ymin, ymax

    def point_to_t(self, point):
        """If the point lies on the Line, returns its `t` parameter.
        If the point does not lie on the Line, returns None."""

        # Single-precision floats have only 7 significant figures of
        # resolution, so test that we're within 6 sig figs.
        if np.isclose(point, self.start, rtol=0, atol=1e-6):
            return 0.0
        elif np.isclose(point, self.end, rtol=0, atol=1e-6):
            return 1.0

        # Finding the point "by hand" here is much faster than calling
        # radialrange(), see the discussion on PR #40:
        # https://github.com/mathandy/svgpathtools/pull/40#issuecomment-358134261

        p = self.poly()
        # p(t) = (p_1 * t) + p_0 = point
        # t = (point - p_0) / p_1
        t = (point - p[0]) / p[1]
        if np.isclose(t.imag, 0) and (t.real >= 0.0) and (t.real <= 1.0):
            return t.real
        return None

    def cropped(self, t0, t1):
        """returns a cropped copy of this segment which starts at
        self.point(t0) and ends at self.point(t1)."""
        return Line(self.point(t0), self.point(t1))

    def split(self, t):
        """returns two segments, whose union is this segment and which join at
        self.point(t)."""
        pt = self.point(t)
        return Line(self.start, pt), Line(pt, self.end)

    def radialrange(self, origin, return_all_global_extrema=False):
        """returns the tuples (d_min, t_min) and (d_max, t_max) which minimize
        and maximize, respectively, the distance d = |self.point(t)-origin|."""
        return bezier_radialrange(self, origin,
                return_all_global_extrema=return_all_global_extrema)

    def rotated(self, degs, origin=None):
        """Returns a copy of self rotated by `degs` degrees (CCW) around the
        point `origin` (a complex number).  By default `origin` is either
        `self.point(0.5)`, or in the case that self is an Arc object,
        `origin` defaults to `self.center`."""
        return rotate(self, degs, origin=origin)

    def translated(self, z0):
        """Returns a copy of self shifted by the complex quantity `z0` such
        that self.translated(z0).point(t) = self.point(t) + z0 for any t."""
        return translate(self, z0)

    def scaled(self, sx, sy=None, origin=0j):
        """Scale transform.  See `scale` function for further explanation."""
        return scale(self, sx=sx, sy=sy, origin=origin)


class QuadraticBezier(object):
    # For compatibility with old pickle files.
    _length_info = {'length': None, 'bpoints': None}

    def __init__(self, start, control, end):
        self.start = start
        self.end = end
        self.control = control

        # used to know if self._length needs to be updated
        self._length_info = {'length': None, 'bpoints': None}

    def __repr__(self):
        return 'QuadraticBezier(start=%s, control=%s, end=%s)' % (
            self.start, self.control, self.end)

    def __eq__(self, other):
        if not isinstance(other, QuadraticBezier):
            return NotImplemented
        return self.start == other.start and self.end == other.end \
            and self.control == other.control

    def __ne__(self, other):
        if not isinstance(other, QuadraticBezier):
            return NotImplemented
        return not self == other

    def __getitem__(self, item):
        return self.bpoints()[item]

    def __len__(self):
        return 3

    def is_smooth_from(self, previous, warning_on=True):
        """[Warning: The name of this method is somewhat misleading (yet kept
        for compatibility with scripts created using svg.path 2.0).  This
        method is meant only for d string creation and should not be used to
        check for kinks.  To check a segment for differentiability, use the
        joins_smoothly_with() method instead.]"""
        if warning_on:
            warn(_is_smooth_from_warning)
        if isinstance(previous, QuadraticBezier):
            return (self.start == previous.end and
                    (self.control - self.start) == (
                        previous.end - previous.control))
        else:
            return self.control == self.start

    def joins_smoothly_with(self, previous, wrt_parameterization=False,
                            error=0):
        """Checks if this segment joins smoothly with previous segment.  By
        default, this only checks that this segment starts moving (at t=0) in
        the same direction (and from the same positive) as previous stopped
        moving (at t=1).  To check if the tangent magnitudes also match, set
        wrt_parameterization=True."""
        if wrt_parameterization:
            return self.start == previous.end and abs(
                self.derivative(0) - previous.derivative(1)) <= error
        else:
            return self.start == previous.end and abs(
                self.unit_tangent(0) - previous.unit_tangent(1)) <= error

    def point(self, t):
        """returns the coordinates of the Bezier curve evaluated at t."""
        return (1 - t)**2*self.start + 2*(1 - t)*t*self.control + t**2*self.end

    def length(self, t0=0, t1=1, error=None, min_depth=None):
        if t0 == 1 and t1 == 0:
            if self._length_info['bpoints'] == self.bpoints():
                return self._length_info['length']
        a = self.start - 2*self.control + self.end
        b = 2*(self.control - self.start)
        a_dot_b = a.real*b.real + a.imag*b.imag

        if abs(a) < 1e-12:
            s = abs(b)*(t1 - t0)
        elif abs(a_dot_b + abs(a)*abs(b)) < 1e-12:
            tstar = abs(b)/(2*abs(a))
            if t1 < tstar:
                return abs(a)*(t0**2 - t1**2) - abs(b)*(t0 - t1)
            elif tstar < t0:
                return abs(a)*(t1**2 - t0**2) - abs(b)*(t1 - t0)
            else:
                return abs(a)*(t1**2 + t0**2) - abs(b)*(t1 + t0) + \
                    abs(b)**2/(2*abs(a))
        else:
            c2 = 4*(a.real**2 + a.imag**2)
            c1 = 4*a_dot_b
            c0 = b.real**2 + b.imag**2

            beta = c1/(2*c2)
            gamma = c0/c2 - beta**2

            dq1_mag = sqrt(c2*t1**2 + c1*t1 + c0)
            dq0_mag = sqrt(c2*t0**2 + c1*t0 + c0)
            logarand = (sqrt(c2)*(t1 + beta) + dq1_mag) / \
                       (sqrt(c2)*(t0 + beta) + dq0_mag)

            s = (t1 + beta)*dq1_mag - (t0 + beta)*dq0_mag + \
                gamma*sqrt(c2)*log(logarand)
            s /= 2

        if t0 == 1 and t1 == 0:
            self._length_info['length'] = s
            self._length_info['bpoints'] = self.bpoints()
            return self._length_info['length']
        else:
            return s

    def ilength(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns a float, t, such that self.length(0, t) is approximately s.
        See the inv_arclength() docstring for more details."""
        return inv_arclength(self, s, s_tol=s_tol, maxits=maxits, error=error,
                             min_depth=min_depth)

    def bpoints(self):
        """returns the Bezier control points of the segment."""
        return self.start, self.control, self.end

    def poly(self, return_coeffs=False):
        """returns the quadratic as a Polynomial object."""
        p = self.bpoints()
        coeffs = (p[0] - 2*p[1] + p[2], 2*(p[1] - p[0]), p[0])
        if return_coeffs:
            return coeffs
        else:
            return np.poly1d(coeffs)

    def derivative(self, t, n=1):
        """returns the nth derivative of the segment at t.
        Note: Bezier curves can have points where their derivative vanishes.
        If you are interested in the tangent direction, use the unit_tangent()
        method instead."""
        p = self.bpoints()
        if n == 1:
            return 2*((p[1] - p[0])*(1 - t) + (p[2] - p[1])*t)
        elif n == 2:
            return 2*(p[2] - 2*p[1] + p[0])
        elif n > 2:
            return 0
        else:
            raise ValueError("n should be a positive integer.")

    def unit_tangent(self, t):
        """returns the unit tangent vector of the segment at t (centered at
        the origin and expressed as a complex number).  If the tangent
        vector's magnitude is zero, this method will find the limit of
        self.derivative(tau)/abs(self.derivative(tau)) as tau approaches t."""
        return bezier_unit_tangent(self, t)

    def normal(self, t):
        """returns the (right hand rule) unit normal vector to self at t."""
        return -1j*self.unit_tangent(t)

    def curvature(self, t):
        """returns the curvature of the segment at t."""
        return segment_curvature(self, t)

    # def icurvature(self, kappa):
    #     """returns a list of t-values such that 0 <= t<= 1 and
    #     seg.curvature(t) = kappa."""
    #     z = self.poly()
    #     x, y = real(z), imag(z)
    #     dx, dy = x.deriv(), y.deriv()
    #     ddx, ddy = dx.deriv(), dy.deriv()
    #
    #     p = kappa**2*(dx**2 + dy**2)**3 - (dx*ddy - ddx*dy)**2
    #     return polyroots01(p)

    def reversed(self):
        """returns a copy of the QuadraticBezier object with its orientation
        reversed."""
        new_quad = QuadraticBezier(self.end, self.control, self.start)
        if self._length_info['length']:
            new_quad._length_info = self._length_info
            new_quad._length_info['bpoints'] = (
                self.end, self.control, self.start)
        return new_quad

    def intersect(self, other_seg, tol=1e-12):
        """Finds the intersections of two segments.
        returns a list of tuples (t1, t2) such that
        self.point(t1) == other_seg.point(t2).
        Note: This will fail if the two segments coincide for more than a
        finite collection of points."""
        if isinstance(other_seg, Line):
            return bezier_by_line_intersections(self, other_seg)
        elif isinstance(other_seg, QuadraticBezier):
            assert self != other_seg
            longer_length = max(self.length(), other_seg.length())
            return bezier_intersections(self, other_seg,
                                        longer_length=longer_length,
                                        tol=tol, tol_deC=tol)
        elif isinstance(other_seg, CubicBezier):
            longer_length = max(self.length(), other_seg.length())
            return bezier_intersections(self, other_seg,
                                        longer_length=longer_length,
                                        tol=tol, tol_deC=tol)
        elif isinstance(other_seg, Arc):
            t2t1s = other_seg.intersect(self)
            return [(t1, t2) for t2, t1 in t2t1s]
        elif isinstance(other_seg, Path):
            raise TypeError(
                "other_seg must be a path segment, not a Path object, use "
                "Path.intersect().")
        else:
            raise TypeError("other_seg must be a path segment.")

    def bbox(self):
        """returns the bounding box for the segment in the form
        (xmin, xmax, ymin, ymax)."""
        return bezier_bounding_box(self)

    def split(self, t):
        """returns two segments, whose union is this segment and which join at
        self.point(t)."""
        bpoints1, bpoints2 = split_bezier(self.bpoints(), t)
        return QuadraticBezier(*bpoints1), QuadraticBezier(*bpoints2)

    def cropped(self, t0, t1):
        """returns a cropped copy of this segment which starts at
        self.point(t0) and ends at self.point(t1)."""
        return QuadraticBezier(*crop_bezier(self, t0, t1))

    def radialrange(self, origin, return_all_global_extrema=False):
        """returns the tuples (d_min, t_min) and (d_max, t_max) which minimize
        and maximize, respectively, the distance d = |self.point(t)-origin|."""
        return bezier_radialrange(self, origin,
                return_all_global_extrema=return_all_global_extrema)

    def rotated(self, degs, origin=None):
        """Returns a copy of self rotated by `degs` degrees (CCW) around the
        point `origin` (a complex number).  By default `origin` is either
        `self.point(0.5)`, or in the case that self is an Arc object,
        `origin` defaults to `self.center`."""
        return rotate(self, degs, origin=origin)

    def translated(self, z0):
        """Returns a copy of self shifted by the complex quantity `z0` such
        that self.translated(z0).point(t) = self.point(t) + z0 for any t."""
        return translate(self, z0)

    def scaled(self, sx, sy=None, origin=0j):
        """Scale transform.  See `scale` function for further explanation."""
        return scale(self, sx=sx, sy=sy, origin=origin)


class CubicBezier(object):
    # For compatibility with old pickle files.
    _length_info = {'length': None, 'bpoints': None, 'error': None,
                    'min_depth': None}

    def __init__(self, start, control1, control2, end):
        self.start = start
        self.control1 = control1
        self.control2 = control2
        self.end = end

        # used to know if self._length needs to be updated
        self._length_info = {'length': None, 'bpoints': None, 'error': None,
                             'min_depth': None}

    def __repr__(self):
        return 'CubicBezier(start=%s, control1=%s, control2=%s, end=%s)' % (
            self.start, self.control1, self.control2, self.end)

    def __eq__(self, other):
        if not isinstance(other, CubicBezier):
            return NotImplemented
        return self.start == other.start and self.end == other.end \
            and self.control1 == other.control1 \
            and self.control2 == other.control2

    def __ne__(self, other):
        if not isinstance(other, CubicBezier):
            return NotImplemented
        return not self == other

    def __getitem__(self, item):
        return self.bpoints()[item]

    def __len__(self):
        return 4

    def is_smooth_from(self, previous, warning_on=True):
        """[Warning: The name of this method is somewhat misleading (yet kept
        for compatibility with scripts created using svg.path 2.0).  This
        method is meant only for d string creation and should not be used to
        check for kinks.  To check a segment for differentiability, use the
        joins_smoothly_with() method instead.]"""
        if warning_on:
            warn(_is_smooth_from_warning)
        if isinstance(previous, CubicBezier):
            return (self.start == previous.end and
                    (self.control1 - self.start) == (
                        previous.end - previous.control2))
        else:
            return self.control1 == self.start

    def joins_smoothly_with(self, previous, wrt_parameterization=False):
        """Checks if this segment joins smoothly with previous segment.  By
        default, this only checks that this segment starts moving (at t=0) in
        the same direction (and from the same positive) as previous stopped
        moving (at t=1).  To check if the tangent magnitudes also match, set
        wrt_parameterization=True."""
        if wrt_parameterization:
            return self.start == previous.end and np.isclose(
                self.derivative(0), previous.derivative(1))
        else:
            return self.start == previous.end and np.isclose(
                self.unit_tangent(0), previous.unit_tangent(1))

    def point(self, t):
        """Evaluate the cubic Bezier curve at t using Horner's rule."""
        # algebraically equivalent to
        # P0*(1-t)**3 + 3*P1*t*(1-t)**2 + 3*P2*(1-t)*t**2 + P3*t**3
        # for (P0, P1, P2, P3) = self.bpoints()
        return self.start + t*(
            3*(self.control1 - self.start) + t*(
                3*(self.start + self.control2) - 6*self.control1 + t*(
                    -self.start + 3*(self.control1 - self.control2) + self.end
                )))

    def length(self, t0=0, t1=1, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        """Calculate the length of the path up to a certain position"""
        if t0 == 0 and t1 == 1:
            if self._length_info['bpoints'] == self.bpoints() \
                    and self._length_info['error'] >= error \
                    and self._length_info['min_depth'] >= min_depth:
                return self._length_info['length']

        # using scipy.integrate.quad is quick
        if _quad_available:
            s = quad(lambda tau: abs(self.derivative(tau)), t0, t1,
                            epsabs=error, limit=1000)[0]
        else:
            s = segment_length(self, t0, t1, self.point(t0), self.point(t1),
                               error, min_depth, 0)

        if t0 == 0 and t1 == 1:
            self._length_info['length'] = s
            self._length_info['bpoints'] = self.bpoints()
            self._length_info['error'] = error
            self._length_info['min_depth'] = min_depth
            return self._length_info['length']
        else:
            return s

    def ilength(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns a float, t, such that self.length(0, t) is approximately s.
        See the inv_arclength() docstring for more details."""
        return inv_arclength(self, s, s_tol=s_tol, maxits=maxits, error=error,
                             min_depth=min_depth)

    def bpoints(self):
        """returns the Bezier control points of the segment."""
        return self.start, self.control1, self.control2, self.end

    def poly(self, return_coeffs=False):
        """Returns a the cubic as a Polynomial object."""
        p = self.bpoints()
        coeffs = (-p[0] + 3*(p[1] - p[2]) + p[3],
                  3*(p[0] - 2*p[1] + p[2]),
                  3*(-p[0] + p[1]),
                  p[0])
        if return_coeffs:
            return coeffs
        else:
            return np.poly1d(coeffs)

    def derivative(self, t, n=1):
        """returns the nth derivative of the segment at t.
        Note: Bezier curves can have points where their derivative vanishes.
        If you are interested in the tangent direction, use the unit_tangent()
        method instead."""
        p = self.bpoints()
        if n == 1:
            return 3*(p[1] - p[0])*(1 - t)**2 + 6*(p[2] - p[1])*(1 - t)*t + 3*(
                p[3] - p[2])*t**2
        elif n == 2:
            return 6*(
                (1 - t)*(p[2] - 2*p[1] + p[0]) + t*(p[3] - 2*p[2] + p[1]))
        elif n == 3:
            return 6*(p[3] - 3*(p[2] - p[1]) - p[0])
        elif n > 3:
            return 0
        else:
            raise ValueError("n should be a positive integer.")

    def unit_tangent(self, t):
        """returns the unit tangent vector of the segment at t (centered at
        the origin and expressed as a complex number).  If the tangent
        vector's magnitude is zero, this method will find the limit of
        self.derivative(tau)/abs(self.derivative(tau)) as tau approaches t."""
        return bezier_unit_tangent(self, t)

    def normal(self, t):
        """returns the (right hand rule) unit normal vector to self at t."""
        return -1j * self.unit_tangent(t)

    def curvature(self, t):
        """returns the curvature of the segment at t."""
        return segment_curvature(self, t)

    # def icurvature(self, kappa):
    #     """returns a list of t-values such that 0 <= t<= 1 and
    #     seg.curvature(t) = kappa."""
    #     z = self.poly()
    #     x, y = real(z), imag(z)
    #     dx, dy = x.deriv(), y.deriv()
    #     ddx, ddy = dx.deriv(), dy.deriv()
    #
    #     p = kappa**2*(dx**2 + dy**2)**3 - (dx*ddy - ddx*dy)**2
    #     return polyroots01(p)

    def reversed(self):
        """returns a copy of the CubicBezier object with its orientation
        reversed."""
        new_cub = CubicBezier(self.end, self.control2, self.control1,
                              self.start)
        if self._length_info['length']:
            new_cub._length_info = self._length_info
            new_cub._length_info['bpoints'] = (
                self.end, self.control2, self.control1, self.start)
        return new_cub

    def intersect(self, other_seg, tol=1e-12):
        """Finds the intersections of two segments.
        returns a list of tuples (t1, t2) such that
        self.point(t1) == other_seg.point(t2).
        Note: This will fail if the two segments coincide for more than a
        finite collection of points."""
        if isinstance(other_seg, Line):
            return bezier_by_line_intersections(self, other_seg)
        elif (isinstance(other_seg, QuadraticBezier) or
              isinstance(other_seg, CubicBezier)):
            assert self != other_seg
            longer_length = max(self.length(), other_seg.length())
            return bezier_intersections(self, other_seg,
                                        longer_length=longer_length,
                                        tol=tol, tol_deC=tol)
        elif isinstance(other_seg, Arc):
            t2t1s = other_seg.intersect(self)
            return [(t1, t2) for t2, t1 in t2t1s]
        elif isinstance(other_seg, Path):
            raise TypeError(
                "other_seg must be a path segment, not a Path object, use "
                "Path.intersect().")
        else:
            raise TypeError("other_seg must be a path segment.")

    def bbox(self):
        """returns the bounding box for the segment in the form
        (xmin, xmax, ymin, ymax)."""
        return bezier_bounding_box(self)

    def split(self, t):
        """returns two segments, whose union is this segment and which join at
        self.point(t)."""
        bpoints1, bpoints2 = split_bezier(self.bpoints(), t)
        return CubicBezier(*bpoints1), CubicBezier(*bpoints2)

    def cropped(self, t0, t1):
        """returns a cropped copy of this segment which starts at
        self.point(t0) and ends at self.point(t1)."""
        return CubicBezier(*crop_bezier(self, t0, t1))

    def radialrange(self, origin, return_all_global_extrema=False):
        """returns the tuples (d_min, t_min) and (d_max, t_max) which minimize
        and maximize, respectively, the distance d = |self.point(t)-origin|."""
        return bezier_radialrange(self, origin,
                return_all_global_extrema=return_all_global_extrema)

    def rotated(self, degs, origin=None):
        """Returns a copy of self rotated by `degs` degrees (CCW) around the
        point `origin` (a complex number).  By default `origin` is either
        `self.point(0.5)`, or in the case that self is an Arc object,
        `origin` defaults to `self.center`."""
        return rotate(self, degs, origin=origin)

    def translated(self, z0):
        """Returns a copy of self shifted by the complex quantity `z0` such
        that self.translated(z0).point(t) = self.point(t) + z0 for any t."""
        return translate(self, z0)

    def scaled(self, sx, sy=None, origin=0j):
        """Scale transform.  See `scale` function for further explanation."""
        return scale(self, sx=sx, sy=sy, origin=origin)


class Arc(object):
    def __init__(self, start, radius, rotation, large_arc, sweep, end,
                 autoscale_radius=True):
        """
        This should be thought of as a part of an ellipse connecting two
        points on that ellipse, start and end.
        Parameters
        ----------
        start : complex
            The start point of the curve. Note: `start` and `end` cannot be the
            same.  To make a full ellipse or circle, use two `Arc` objects.
        radius : complex
            rx + 1j*ry, where rx and ry are the radii of the ellipse (also
            known as its semi-major and semi-minor axes, or vice-versa or if
            rx < ry).
            Note: If rx = 0 or ry = 0 then this arc is treated as a
            straight line segment joining the endpoints.
            Note: If rx or ry has a negative sign, the sign is dropped; the
            absolute value is used instead.
            Note:  If no such ellipse exists, the radius will be scaled so
            that one does (unless autoscale_radius is set to False).
        rotation : float
            This is the CCW angle (in degrees) from the positive x-axis of the
            current coordinate system to the x-axis of the ellipse.
        large_arc : bool
            Given two points on an ellipse, there are two elliptical arcs
            connecting those points, the first going the short way around the
            ellipse, and the second going the long way around the ellipse.  If
            `large_arc == False`, the shorter elliptical arc will be used.  If
            `large_arc == True`, then longer elliptical will be used.
            In other words, `large_arc` should be 0 for arcs spanning less than
            or equal to 180 degrees and 1 for arcs spanning greater than 180
            degrees.
        sweep : bool
            For any acceptable parameters `start`, `end`, `rotation`, and
            `radius`, there are two ellipses with the given major and minor
            axes (radii) which connect `start` and `end`.  One which connects
            them in a CCW fashion and one which connected them in a CW
            fashion.  If `sweep == True`, the CCW ellipse will be used.  If
            `sweep == False`, the CW ellipse will be used.  See note on curve
            orientation below.
        end : complex
            The end point of the curve. Note: `start` and `end` cannot be the
            same.  To make a full ellipse or circle, use two `Arc` objects.
        autoscale_radius : bool
            If `autoscale_radius == True`, then will also scale `self.radius`
            in the case that no ellipse exists with the input parameters
            (see inline comments for further explanation).

        Derived Parameters/Attributes
        -----------------------------
        self.theta : float
            This is the phase (in degrees) of self.u1transform(self.start).
            It is $\theta_1$ in the official documentation and ranges from
            -180 to 180.
        self.delta : float
            This is the angular distance (in degrees) between the start and
            end of the arc after the arc has been sent to the unit circle
            through self.u1transform().
            It is $\Delta\theta$ in the official documentation and ranges from
            -360 to 360; being positive when the arc travels CCW and negative
            otherwise (i.e. is positive/negative when sweep == True/False).
        self.center : complex
            This is the center of the arc's ellipse.
        self.phi : float
            The arc's rotation in radians, i.e. `radians(self.rotation)`.
        self.rot_matrix : complex
            Equal to `exp(1j * self.phi)` which is also equal to
            `cos(self.phi) + 1j*sin(self.phi)`.


        Note on curve orientation (CW vs CCW)
        -------------------------------------
        The notions of clockwise (CW) and counter-clockwise (CCW) are reversed
        in some sense when viewing SVGs (as the y coordinate starts at the top
        of the image and increases towards the bottom).
        """
        assert start != end
        assert radius.real != 0 and radius.imag != 0

        self.start = start
        self.radius = abs(radius.real) + 1j*abs(radius.imag)
        self.rotation = rotation
        self.large_arc = bool(large_arc)
        self.sweep = bool(sweep)
        self.end = end
        self.autoscale_radius = autoscale_radius

        # Convenience parameters
        self.phi = radians(self.rotation)
        self.rot_matrix = exp(1j*self.phi)

        # Derive derived parameters
        self._parameterize()

    def __repr__(self):
        params = (self.start, self.radius, self.rotation,
                  self.large_arc, self.sweep, self.end)
        return ("Arc(start={}, radius={}, rotation={}, "
                "large_arc={}, sweep={}, end={})".format(*params))

    def __eq__(self, other):
        if not isinstance(other, Arc):
            return NotImplemented
        return self.start == other.start and self.end == other.end \
            and self.radius == other.radius \
            and self.rotation == other.rotation \
            and self.large_arc == other.large_arc and self.sweep == other.sweep

    def __ne__(self, other):
        if not isinstance(other, Arc):
            return NotImplemented
        return not self == other

    def _parameterize(self):
        # See http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
        # my notation roughly follows theirs
        rx = self.radius.real
        ry = self.radius.imag
        rx_sqd = rx*rx
        ry_sqd = ry*ry

        # Transform z-> z' = x' + 1j*y'
        # = self.rot_matrix**(-1)*(z - (end+start)/2)
        # coordinates.  This translates the ellipse so that the midpoint
        # between self.end and self.start lies on the origin and rotates
        # the ellipse so that the its axes align with the xy-coordinate axes.
        # Note:  This sends self.end to -self.start
        zp1 = (1/self.rot_matrix)*(self.start - self.end)/2
        x1p, y1p = zp1.real, zp1.imag
        x1p_sqd = x1p*x1p
        y1p_sqd = y1p*y1p

        # Correct out of range radii
        # Note: an ellipse going through start and end with radius and phi
        # exists if and only if radius_check is true
        radius_check = (x1p_sqd/rx_sqd) + (y1p_sqd/ry_sqd)
        if radius_check > 1:
            if self.autoscale_radius:
                rx *= sqrt(radius_check)
                ry *= sqrt(radius_check)
                self.radius = rx + 1j*ry
                rx_sqd = rx*rx
                ry_sqd = ry*ry
            else:
                raise ValueError("No such elliptic arc exists.")

        # Compute c'=(c_x', c_y'), the center of the ellipse in (x', y') coords
        # Noting that, in our new coord system, (x_2', y_2') = (-x_1', -x_2')
        # and our ellipse is cut out by of the plane by the algebraic equation
        # (x'-c_x')**2 / r_x**2 + (y'-c_y')**2 / r_y**2 = 1,
        # we can find c' by solving the system of two quadratics given by
        # plugging our transformed endpoints (x_1', y_1') and (x_2', y_2')
        tmp = rx_sqd*y1p_sqd + ry_sqd*x1p_sqd
        radicand = (rx_sqd*ry_sqd - tmp) / tmp
        try:
            radical = sqrt(radicand)
        except ValueError:
            radical = 0
        if self.large_arc == self.sweep:
            cp = -radical*(rx*y1p/ry - 1j*ry*x1p/rx)
        else:
            cp = radical*(rx*y1p/ry - 1j*ry*x1p/rx)

        # The center in (x,y) coordinates is easy to find knowing c'
        self.center = exp(1j*self.phi)*cp + (self.start + self.end)/2

        # Now we do a second transformation, from (x', y') to (u_x, u_y)
        # coordinates, which is a translation moving the center of the
        # ellipse to the origin and a dilation stretching the ellipse to be
        # the unit circle
        u1 = (x1p - cp.real)/rx + 1j*(y1p - cp.imag)/ry  # transformed start
        u2 = (-x1p - cp.real)/rx + 1j*(-y1p - cp.imag)/ry  # transformed end

        # clip in case of floating point error
        u1 = np.clip(u1.real, -1, 1) + 1j*np.clip(u1.imag, -1, 1)
        u2 = np.clip(u2.real, -1, 1) + 1j * np.clip(u2.imag, -1, 1)

        # Now compute theta and delta (we'll define them as we go)
        # delta is the angular distance of the arc (w.r.t the circle)
        # theta is the angle between the positive x'-axis and the start point
        # on the circle
        if u1.imag > 0:
            self.theta = degrees(acos(u1.real))
        elif u1.imag < 0:
            self.theta = -degrees(acos(u1.real))
        else:
            if u1.real > 0:  # start is on pos u_x axis
                self.theta = 0
            else:  # start is on neg u_x axis
                # Note: This behavior disagrees with behavior documented in
                # http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
                # where theta is set to 0 in this case.
                self.theta = 180

        det_uv = u1.real*u2.imag - u1.imag*u2.real

        acosand = u1.real*u2.real + u1.imag*u2.imag
        acosand = np.clip(acosand.real, -1, 1) + np.clip(acosand.imag, -1, 1)
        
        if det_uv > 0:
            self.delta = degrees(acos(acosand))
        elif det_uv < 0:
            self.delta = -degrees(acos(acosand))
        else:
            if u1.real*u2.real + u1.imag*u2.imag > 0:
                # u1 == u2
                self.delta = 0
            else:
                # u1 == -u2
                # Note: This behavior disagrees with behavior documented in
                # http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
                # where delta is set to 0 in this case.
                self.delta = 180

        if not self.sweep and self.delta >= 0:
            self.delta -= 360
        elif self.large_arc and self.delta <= 0:
            self.delta += 360

    def point(self, t):
        if t == 0:
            return self.start
        if t == 1:
            return self.end
        angle = radians(self.theta + t*self.delta)
        cosphi = self.rot_matrix.real
        sinphi = self.rot_matrix.imag
        rx = self.radius.real
        ry = self.radius.imag

        # z = self.rot_matrix*(rx*cos(angle) + 1j*ry*sin(angle)) + self.center
        x = rx*cosphi*cos(angle) - ry*sinphi*sin(angle) + self.center.real
        y = rx*sinphi*cos(angle) + ry*cosphi*sin(angle) + self.center.imag
        return complex(x, y)

    def point_to_t(self, point):
        """If the point lies on the Arc, returns its `t` parameter.
        If the point does not lie on the Arc, returns None.
        This function only works on Arcs with rotation == 0.0"""

        def in_range(min, max, val):
            return (min <= val) and (max >= val)

        # Single-precision floats have only 7 significant figures of
        # resolution, so test that we're within 6 sig figs.
        if np.isclose(point, self.start, rtol=0.0, atol=1e-6):
            return 0.0
        elif np.isclose(point, self.end, rtol=0.0, atol=1e-6):
            return 1.0

        if self.rotation != 0.0:
            raise ValueError("Arc.point_to_t() only works on non-rotated Arcs.")

        v = point - self.center
        distance_from_center = sqrt((v.real * v.real) + (v.imag * v.imag))
        min_radius = min(self.radius.real, self.radius.imag)
        max_radius = max(self.radius.real, self.radius.imag)
        if (distance_from_center < min_radius) and not np.isclose(distance_from_center, min_radius):
            return None
        if (distance_from_center > max_radius) and not np.isclose(distance_from_center, max_radius):
            return None

        # x = center_x + radius_x cos(radians(theta + t delta))
        # y = center_y + radius_y sin(radians(theta + t delta))
        #
        # For x:
        # cos(radians(theta + t delta)) = (x - center_x) / radius_x
        # radians(theta + t delta) = acos((x - center_x) / radius_x)
        # theta + t delta = degrees(acos((x - center_x) / radius_x))
        # t_x = (degrees(acos((x - center_x) / radius_x)) - theta) / delta
        #
        # Similarly for y:
        # t_y = (degrees(asin((y - center_y) / radius_y)) - theta) / delta

        x = point.real
        y = point.imag

        #
        # +Y points down!
        #
        # sweep mean clocwise
        # sweep && (delta > 0)
        # !sweep && (delta < 0)
        #
        # -180 <= theta_1 <= 180
        #
        # large_arc && (-360 <= delta <= 360)
        # !large_arc && (-180 < delta < 180)
        #

        end_angle = self.theta + self.delta
        min_angle = min(self.theta, end_angle)
        max_angle = max(self.theta, end_angle)

        acos_arg = (x - self.center.real) / self.radius.real
        if acos_arg > 1.0:
            acos_arg = 1.0
        elif acos_arg < -1.0:
            acos_arg = -1.0

        x_angle_0 = degrees(acos(acos_arg))
        while x_angle_0 < min_angle:
            x_angle_0 += 360.0
        while x_angle_0 > max_angle:
            x_angle_0 -= 360.0

        x_angle_1 = -1.0 * x_angle_0
        while x_angle_1 < min_angle:
            x_angle_1 += 360.0
        while x_angle_1 > max_angle:
            x_angle_1 -= 360.0

        t_x_0 = (x_angle_0 - self.theta) / self.delta
        t_x_1 = (x_angle_1 - self.theta) / self.delta

        asin_arg = (y - self.center.imag) / self.radius.imag
        if asin_arg > 1.0:
            asin_arg = 1.0
        elif asin_arg < -1.0:
            asin_arg = -1.0

        y_angle_0 = degrees(asin(asin_arg))
        while y_angle_0 < min_angle:
            y_angle_0 += 360.0
        while y_angle_0 > max_angle:
            y_angle_0 -= 360.0

        y_angle_1 = 180 - y_angle_0
        while y_angle_1 < min_angle:
            y_angle_1 += 360.0
        while y_angle_1 > max_angle:
            y_angle_1 -= 360.0

        t_y_0 = (y_angle_0 - self.theta) / self.delta
        t_y_1 = (y_angle_1 - self.theta) / self.delta

        t = None
        if np.isclose(t_x_0, t_y_0):
            t = (t_x_0 + t_y_0) / 2.0
        elif np.isclose(t_x_0, t_y_1):
            t= (t_x_0 + t_y_1) / 2.0
        elif np.isclose(t_x_1, t_y_0):
            t = (t_x_1 + t_y_0) / 2.0
        elif np.isclose(t_x_1, t_y_1):
            t = (t_x_1 + t_y_1) / 2.0
        else:
            # Comparing None and float yields a result in python2,
            # but throws TypeError in python3.  This fix (suggested by
            # @CatherineH) explicitly handles and avoids the case where
            # the None-vs-float comparison would have happened below.
            return None

        if (t >= 0.0) and (t <= 1.0):
            return t

        return None

    def centeriso(self, z):
        """This is an isometry that translates and rotates self so that it
        is centered on the origin and has its axes aligned with the xy axes."""
        return (1/self.rot_matrix)*(z - self.center)

    def icenteriso(self, zeta):
        """This is an isometry, the inverse of standardiso()."""
        return self.rot_matrix*zeta + self.center

    def u1transform(self, z):
        """This is an affine transformation (same as used in
        self._parameterize()) that sends self to the unit circle."""
        zeta = (1/self.rot_matrix)*(z - self.center)  # same as centeriso(z)
        x, y = real(zeta), imag(zeta)
        return x/self.radius.real + 1j*y/self.radius.imag

    def iu1transform(self, zeta):
        """This is an affine transformation, the inverse of
        self.u1transform()."""
        x = real(zeta)
        y = imag(zeta)
        z = x*self.radius.real + y*self.radius.imag
        return self.rot_matrix*z + self.center

    def length(self, t0=0, t1=1, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        """The length of an elliptical large_arc segment requires numerical
        integration, and in that case it's simpler to just do a geometric
        approximation, as for cubic bezier curves."""
        assert 0 <= t0 <= 1 and 0 <= t1 <= 1
        if _quad_available:
            return quad(lambda tau: abs(self.derivative(tau)), t0, t1,
                        epsabs=error, limit=1000)[0]
        else:
            return segment_length(self, t0, t1, self.point(t0), self.point(t1),
                                  error, min_depth, 0)

    def ilength(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns a float, t, such that self.length(0, t) is approximately s.
        See the inv_arclength() docstring for more details."""
        return inv_arclength(self, s, s_tol=s_tol, maxits=maxits, error=error,
                             min_depth=min_depth)

    def joins_smoothly_with(self, previous, wrt_parameterization=False,
                            error=0):
        """Checks if this segment joins smoothly with previous segment.  By
        default, this only checks that this segment starts moving (at t=0) in
        the same direction (and from the same positive) as previous stopped
        moving (at t=1).  To check if the tangent magnitudes also match, set
        wrt_parameterization=True."""
        if wrt_parameterization:
            return self.start == previous.end and abs(
                self.derivative(0) - previous.derivative(1)) <= error
        else:
            return self.start == previous.end and abs(
                self.unit_tangent(0) - previous.unit_tangent(1)) <= error

    def derivative(self, t, n=1):
        """returns the nth derivative of the segment at t."""
        angle = radians(self.theta + t*self.delta)
        phi = radians(self.rotation)
        rx = self.radius.real
        ry = self.radius.imag
        k = (self.delta*2*pi/360)**n  # ((d/dt)angle)**n

        if n % 4 == 0 and n > 0:
            return rx*cos(phi)*cos(angle) - ry*sin(phi)*sin(angle) + 1j*(
                rx*sin(phi)*cos(angle) + ry*cos(phi)*sin(angle))
        elif n % 4 == 1:
            return k*(-rx*cos(phi)*sin(angle) - ry*sin(phi)*cos(angle) + 1j*(
                -rx*sin(phi)*sin(angle) + ry*cos(phi)*cos(angle)))
        elif n % 4 == 2:
            return k*(-rx*cos(phi)*cos(angle) + ry*sin(phi)*sin(angle) + 1j*(
                -rx*sin(phi)*cos(angle) - ry*cos(phi)*sin(angle)))
        elif n % 4 == 3:
            return k*(rx*cos(phi)*sin(angle) + ry*sin(phi)*cos(angle) + 1j*(
                rx*sin(phi)*sin(angle) - ry*cos(phi)*cos(angle)))
        else:
            raise ValueError("n should be a positive integer.")

    def unit_tangent(self, t):
        """returns the unit tangent vector of the segment at t (centered at
        the origin and expressed as a complex number)."""
        dseg = self.derivative(t)
        return dseg/abs(dseg)

    def normal(self, t):
        """returns the (right hand rule) unit normal vector to self at t."""
        return -1j*self.unit_tangent(t)

    def curvature(self, t):
        """returns the curvature of the segment at t."""
        return segment_curvature(self, t)

    # def icurvature(self, kappa):
    #     """returns a list of t-values such that 0 <= t<= 1 and
    #     seg.curvature(t) = kappa."""
    #
    #     a, b = self.radius.real, self.radius.imag
    #     if kappa > min(a, b)/max(a, b)**2 or kappa <= 0:
    #         return []
    #     if a==b:
    #         if kappa != 1/a:
    #             return []
    #         else:
    #             raise ValueError(
    #                 "The .icurvature() method for Arc elements with "
    #                 "radius.real == radius.imag (i.e. circle segments) "
    #                 "will raise this exception when kappa is 1/radius.real as "
    #                 "this is true at every point on the circle segment.")
    #
    #     # kappa = a*b / (a^2sin^2(tau) + b^2cos^2(tau))^(3/2), tau=2*pi*phase
    #     sin2 = np.poly1d([1, 0])
    #     p = kappa**2*(a*sin2 + b*(1 - sin2))**3 - a*b
    #     sin2s = polyroots01(p)
    #     taus = []
    #
    #     for sin2 in sin2s:
    #         taus += [np.arcsin(sqrt(sin2)), np.arcsin(-sqrt(sin2))]
    #
    #     # account for the other branch of arcsin
    #     sgn = lambda x: x/abs(x) if x else 0
    #     other_taus = [sgn(tau)*np.pi - tau for tau in taus if abs(tau) != np.pi/2]
    #     taus = taus + other_taus
    #
    #     # get rid of points not included in segment
    #     ts = [phase2t(tau) for tau in taus]
    #
    #     return [t for t in ts if 0<=t<=1]


    def reversed(self):
        """returns a copy of the Arc object with its orientation reversed."""
        return Arc(self.end, self.radius, self.rotation, self.large_arc,
                   not self.sweep, self.start)

    def phase2t(self, psi):
        """Given phase -pi < psi <= pi,
        returns the t value such that
        exp(1j*psi) = self.u1transform(self.point(t)).
        """
        def _deg(rads, domain_lower_limit):
            # Convert rads to degrees in [0, 360) domain
            degs = degrees(rads % (2*pi))

            # Convert to [domain_lower_limit, domain_lower_limit + 360) domain
            k = domain_lower_limit // 360
            degs += k * 360
            if degs < domain_lower_limit:
                degs += 360
            return degs

        if self.delta > 0:
            degs = _deg(psi, domain_lower_limit=self.theta)
        else:
            degs = _deg(psi, domain_lower_limit=self.theta)
        return (degs - self.theta)/self.delta


    def intersect(self, other_seg, tol=1e-12):
        """NOT FULLY IMPLEMENTED.  Finds the intersections of two segments.
        returns a list of tuples (t1, t2) such that
        self.point(t1) == other_seg.point(t2).
        Note: This will fail if the two segments coincide for more than a
        finite collection of points.

        Note: Arc related intersections are only partially supported, i.e. are
        only half-heartedly implemented and not well tested.  Please feel free
        to let me know if you're interested in such a feature -- or even better
        please submit an implementation if you want to code one."""

        # This special case can be easily solved algebraically.
        if (self.rotation == 0) and isinstance(other_seg, Line):
            a = self.radius.real
            b = self.radius.imag

            # Ignore the ellipse's center point (to pretend that it's
            # centered at the origin), and translate the Line to match.
            l = Line(start=(other_seg.start-self.center), end=(other_seg.end-self.center))

            # This gives us the translated Line as a parametric equation.
            # s = p1 t + p0
            p = l.poly()

            if p[1].real == 0.0:
                # The `x` value doesn't depend on `t`, the line is vertical.
                c = p[0].real
                x_values = [c]

                # Substitute the line `x = c` into the equation for the
                # (origin-centered) ellipse.
                #
                # x^2/a^2 + y^2/b^2 = 1
                # c^2/a^2 + y^2/b^2 = 1
                # y^2/b^2 = 1 - c^2/a^2
                # y^2 = b^2(1 - c^2/a^2)
                # y = +-b sqrt(1 - c^2/a^2)

                discriminant = 1 - (c * c)/(a * a)
                if discriminant < 0:
                    return []
                elif discriminant == 0:
                    y_values = [0]
                else:
                    val = b * sqrt(discriminant)
                    y_values = [val, -val]

            else:
                # This is a non-vertical line.
                #
                # Convert the Line's parametric equation to the "y = mx + c" format.
                # x = p1.real t + p0.real
                # y = p1.imag t + p0.imag
                #
                # t = (x - p0.real) / p1.real
                # t = (y - p0.imag) / p1.imag
                #
                # (y - p0.imag) / p1.imag = (x - p0.real) / p1.real
                # (y - p0.imag) = ((x - p0.real) * p1.imag) / p1.real
                # y = ((x - p0.real) * p1.imag) / p1.real + p0.imag
                # y = (x p1.imag - p0.real * p1.imag) / p1.real + p0.imag
                # y = x p1.imag/p1.real - p0.real p1.imag / p1.real + p0.imag
                # m = p1.imag/p1.real
                # c = -m p0.real + p0.imag
                m = p[1].imag / p[1].real
                c = (-m * p[0].real) + p[0].imag

                # Substitute the line's y(x) equation into the equation for
                # the ellipse.  We can pretend the ellipse is centered at the
                # origin, since we shifted the Line by the ellipse's center.
                #
                # x^2/a^2 + y^2/b^2 = 1
                # x^2/a^2 + (mx+c)^2/b^2 = 1
                # (b^2 x^2 + a^2 (mx+c)^2)/(a^2 b^2) = 1
                # b^2 x^2 + a^2 (mx+c)^2 = a^2 b^2
                # b^2 x^2 + a^2(m^2 x^2 + 2mcx + c^2) = a^2 b^2
                # b^2 x^2 + a^2 m^2 x^2 + 2a^2 mcx + a^2 c^2 - a^2 b^2 = 0
                # (a^2 m^2 + b^2)x^2 + 2a^2 mcx + a^2(c^2 - b^2) = 0
                #
                # The quadratic forumla tells us:  x = (-B +- sqrt(B^2 - 4AC)) / 2A
                # Where:
                #     A = a^2 m^2 + b^2
                #     B = 2 a^2 mc
                #     C = a^2(c^2 - b^2)
                #
                # The determinant is: B^2 - 4AC
                #
                # The solution simplifies to:
                # x = (-a^2 mc +- a b sqrt(a^2 m^2 + b^2 - c^2)) / (a^2 m^2 + b^2)
                #
                # Solving the line for x(y) and substituting *that* into
                # the equation for the ellipse gives this solution for y:
                # y = (b^2 c +- abm sqrt(a^2 m^2 + b^2 - c^2)) / (a^2 m^2 + b^2)

                denominator = (a * a * m * m) + (b * b)

                discriminant = denominator - (c * c)
                if discriminant < 0:
                    return []

                x_sqrt = a * b * sqrt(discriminant)
                x1 = (-(a * a * m * c) + x_sqrt) / denominator 
                x2 = (-(a * a * m * c) - x_sqrt) / denominator 
                x_values = [x1]
                if x1 != x2:
                    x_values.append(x2)

                y_sqrt = x_sqrt * m
                y1 = ((b * b * c) + y_sqrt) / denominator
                y2 = ((b * b * c) - y_sqrt) / denominator
                y_values = [y1]
                if y1 != y2:
                    y_values.append(y2)

            intersections = []
            for x in x_values:
                for y in y_values:
                    p = complex(x, y) + self.center
                    my_t = self.point_to_t(p)
                    if my_t == None:
                        continue
                    other_t = other_seg.point_to_t(p)
                    if other_t == None:
                        continue
                    intersections.append([my_t, other_t])
            return intersections

        elif is_bezier_segment(other_seg):
            u1poly = self.u1transform(other_seg.poly())
            u1poly_mag2 = real(u1poly)**2 + imag(u1poly)**2
            t2s = polyroots01(u1poly_mag2 - 1)
            t1s = [self.phase2t(phase(u1poly(t2))) for t2 in t2s]
            return list(zip(t1s, t2s))
        elif isinstance(other_seg, Arc):
            assert other_seg != self
            # This could be made explicit to increase efficiency
            longer_length = max(self.length(), other_seg.length())
            inters = bezier_intersections(self, other_seg,
                                          longer_length=longer_length,
                                          tol=tol, tol_deC=tol)

            # ad hoc fix for redundant solutions
            if len(inters) > 2:
                def keyfcn(tpair):
                    t1, t2 = tpair
                    return abs(self.point(t1) - other_seg.point(t2))
                inters.sort(key=keyfcn)
                for idx in range(1, len(inters)-1):
                    if (abs(inters[idx][0] - inters[idx + 1][0])
                            <  abs(inters[idx][0] - inters[0][0])):
                        return [inters[0], inters[idx]]
                else:
                    return [inters[0], inters[-1]]
            return inters
        else:
            raise TypeError("other_seg should be a Arc, Line, "
                            "QuadraticBezier, or CubicBezier object.")

    def bbox(self):
        """returns a bounding box for the segment in the form
        (xmin, xmax, ymin, ymax)."""
        # a(t) = radians(self.theta + self.delta*t)
        #      = (2*pi/360)*(self.theta + self.delta*t)
        # x'=0: ~~~~~~~~~
        # -rx*cos(phi)*sin(a(t)) = ry*sin(phi)*cos(a(t))
        # -(rx/ry)*cot(phi)*tan(a(t)) = 1
        # a(t) = arctan(-(ry/rx)tan(phi)) + pi*k === atan_x
        # y'=0: ~~~~~~~~~~
        # rx*sin(phi)*sin(a(t)) = ry*cos(phi)*cos(a(t))
        # (rx/ry)*tan(phi)*tan(a(t)) = 1
        # a(t) = arctan((ry/rx)*cot(phi))
        # atanres = arctan((ry/rx)*cot(phi)) === atan_y
        # ~~~~~~~~
        # (2*pi/360)*(self.theta + self.delta*t) = atanres + pi*k
        # Therfore, for both x' and y', we have...
        # t = ((atan_{x/y} + pi*k)*(360/(2*pi)) - self.theta)/self.delta
        # for all k s.t. 0 < t < 1
        from math import atan, tan

        if cos(self.phi) == 0:
            atan_x = pi/2
            atan_y = 0
        elif sin(self.phi) == 0:
            atan_x = 0
            atan_y = pi/2
        else:
            rx, ry = self.radius.real, self.radius.imag
            atan_x = atan(-(ry/rx)*tan(self.phi))
            atan_y = atan((ry/rx)/tan(self.phi))

        def angle_inv(ang, k):  # inverse of angle from Arc.derivative()
            return ((ang + pi*k)*(360/(2*pi)) - self.theta)/self.delta

        xtrema = [self.start.real, self.end.real]
        ytrema = [self.start.imag, self.end.imag]

        for k in range(-4, 5):
            tx = angle_inv(atan_x, k)
            ty = angle_inv(atan_y, k)
            if 0 <= tx <= 1:
                xtrema.append(self.point(tx).real)
            if 0 <= ty <= 1:
                ytrema.append(self.point(ty).imag)
        xmin = max(xtrema)
        return min(xtrema), max(xtrema), min(ytrema), max(ytrema)

    def split(self, t):
        """returns two segments, whose union is this segment and which join
        at self.point(t)."""
        return self.cropped(0, t), self.cropped(t, 1)

    def cropped(self, t0, t1):
        """returns a cropped copy of this segment which starts at
        self.point(t0) and ends at self.point(t1)."""
        if abs(self.delta*(t1 - t0)) <= 180:
            new_large_arc = 0
        else:
            new_large_arc = 1
        return Arc(self.point(t0), radius=self.radius, rotation=self.rotation,
                   large_arc=new_large_arc, sweep=self.sweep,
                   end=self.point(t1), autoscale_radius=self.autoscale_radius)

    def radialrange(self, origin, return_all_global_extrema=False):
        """returns the tuples (d_min, t_min) and (d_max, t_max) which minimize
        and maximize, respectively, the distance,
        d = |self.point(t)-origin|."""

        # u1orig = self.u1transform(origin)
        # if abs(u1orig) == 1:  # origin lies on ellipse
        #     t = self.phase2t(phase(u1orig))
        #     d_min = 0
        #
        # # Transform to a coordinate system where the ellipse is centered
        # # at the origin and its axes are horizontal/vertical
        # zeta0 = self.centeriso(origin)
        # a, b = self.radius.real, self.radius.imag
        # x0, y0 = zeta0.real, zeta0.imag
        #
        # # Find t s.t. z'(t)
        # a2mb2 = (a**2 - b**2)
        # if u1orig.imag:  # x != x0
        #
        #     coeffs = [a2mb2**2,
        #               2*a2mb2*b**2*y0,
        #               (-a**4 + (2*a**2 - b**2 + y0**2)*b**2 + x0**2)*b**2,
        #               -2*a2mb2*b**4*y0,
        #               -b**6*y0**2]
        #     ys = polyroots(coeffs, realroots=True,
        #                    condition=lambda r: -b <= r <= b)
        #     xs = (a*sqrt(1 - y**2/b**2) for y in ys)
        #
        #     ts = [self.phase2t(phase(self.u1transform(self.icenteriso(
        #         complex(x, y))))) for x, y in zip(xs, ys)]
        #
        # else:  # This case is very similar, see notes and assume instead y0!=y
        #     b2ma2 = (b**2 - a**2)
        #     coeffs = [b2ma2**2,
        #               2*b2ma2*a**2*x0,
        #               (-b**4 + (2*b**2 - a**2 + x0**2)*a**2 + y0**2)*a**2,
        #               -2*b2ma2*a**4*x0,
        #               -a**6*x0**2]
        #     xs = polyroots(coeffs, realroots=True,
        #                    condition=lambda r: -a <= r <= a)
        #     ys = (b*sqrt(1 - x**2/a**2) for x in xs)
        #
        #     ts = [self.phase2t(phase(self.u1transform(self.icenteriso(
        #         complex(x, y))))) for x, y in zip(xs, ys)]

        raise _NotImplemented4ArcException

    def rotated(self, degs, origin=None):
        """Returns a copy of self rotated by `degs` degrees (CCW) around the
        point `origin` (a complex number).  By default `origin` is either
        `self.point(0.5)`, or in the case that self is an Arc object,
        `origin` defaults to `self.center`."""
        return rotate(self, degs, origin=origin)

    def translated(self, z0):
        """Returns a copy of self shifted by the complex quantity `z0` such
        that self.translated(z0).point(t) = self.point(t) + z0 for any t."""
        return translate(self, z0)

    def scaled(self, sx, sy=None, origin=0j):
        """Scale transform.  See `scale` function for further explanation."""
        return scale(self, sx=sx, sy=sy, origin=origin)


def is_bezier_segment(x):
    return (isinstance(x, Line) or
            isinstance(x, QuadraticBezier) or
            isinstance(x, CubicBezier))


def is_path_segment(x):
    return is_bezier_segment(x) or isinstance(x, Arc)


class Path(MutableSequence):
    """A Path is a sequence of path segments"""

    # Put it here, so there is a default if unpickled.
    _closed = False
    _start = None
    _end = None

    def __init__(self, *segments, **kw):
        self._segments = list(segments)
        self._length = None
        self._lengths = None
        if 'closed' in kw:
            self.closed = kw['closed']  # DEPRECATED
        if self._segments:
            self._start = self._segments[0].start
            self._end = self._segments[-1].end
        else:
            self._start = None
            self._end = None

        if 'tree_element' in kw:
            self._tree_element = kw['tree_element']

    def __getitem__(self, index):
        return self._segments[index]

    def __setitem__(self, index, value):
        self._segments[index] = value
        self._length = None
        self._start = self._segments[0].start
        self._end = self._segments[-1].end

    def __delitem__(self, index):
        del self._segments[index]
        self._length = None
        self._start = self._segments[0].start
        self._end = self._segments[-1].end

    def __iter__(self):
        return self._segments.__iter__()

    def __contains__(self, x):
        return self._segments.__contains__(x)

    def insert(self, index, value):
        self._segments.insert(index, value)
        self._length = None
        self._start = self._segments[0].start
        self._end = self._segments[-1].end

    def reversed(self):
        """returns a copy of the Path object with its orientation reversed."""
        newpath = [seg.reversed() for seg in self]
        newpath.reverse()
        return Path(*newpath)

    def __len__(self):
        return len(self._segments)

    def __repr__(self):
        return "Path({})".format(
            ",\n     ".join(repr(x) for x in self._segments))

    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        if len(self) != len(other):
            return False
        for s, o in zip(self._segments, other._segments):
            if not s == o:
                return False
        return True

    def __ne__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return not self == other

    def _calc_lengths(self, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        if self._length is not None:
            return

        lengths = [each.length(error=error, min_depth=min_depth) for each in
                   self._segments]
        self._length = sum(lengths)
        self._lengths = [each/self._length for each in lengths]

    def point(self, pos):

        # Shortcuts
        if pos == 0.0:
            return self._segments[0].point(pos)
        if pos == 1.0:
            return self._segments[-1].point(pos)

        self._calc_lengths()
        # Find which segment the point we search for is located on:
        segment_start = 0
        for index, segment in enumerate(self._segments):
            segment_end = segment_start + self._lengths[index]
            if segment_end >= pos:
                # This is the segment! How far in on the segment is the point?
                segment_pos = (pos - segment_start)/(
                    segment_end - segment_start)
                return segment.point(segment_pos)
            segment_start = segment_end

    def length(self, T0=0, T1=1, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        self._calc_lengths(error=error, min_depth=min_depth)
        if T0 == 0 and T1 == 1:
            return self._length
        else:
            if len(self) == 1:
                return self[0].length(t0=T0, t1=T1)
            idx0, t0 = self.T2t(T0)
            idx1, t1 = self.T2t(T1)
            if idx0 == idx1:
                return self[idx0].length(t0=t0, t1=t1)
            return (self[idx0].length(t0=t0) +
                    sum(self[idx].length() for idx in range(idx0 + 1, idx1)) +
                    self[idx1].length(t1=t1))

    def ilength(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns a float, t, such that self.length(0, t) is approximately s.
        See the inv_arclength() docstring for more details."""
        return inv_arclength(self, s, s_tol=s_tol, maxits=maxits, error=error,
                             min_depth=min_depth)

    def iscontinuous(self):
        """Checks if a path is continuous with respect to its
        parameterization."""
        return all(self[i].end == self[i+1].start for i in range(len(self) - 1))

    def continuous_subpaths(self):
        """Breaks self into its continuous components, returning a list of
        continuous subpaths.
        I.e.
        (all(subpath.iscontinuous() for subpath in self.continuous_subpaths())
         and self == concatpaths(self.continuous_subpaths()))
        )
        """
        subpaths = []
        subpath_start = 0
        for i in range(len(self) - 1):
            if self[i].end != self[(i+1) % len(self)].start:
                subpaths.append(Path(*self[subpath_start: i+1]))
                subpath_start = i+1
        subpaths.append(Path(*self[subpath_start: len(self)]))
        return subpaths

    def isclosed(self):
        """This function determines if a connected path is closed."""
        assert len(self) != 0
        assert self.iscontinuous()
        return self.start == self.end

    def isclosedac(self):
        assert len(self) != 0
        return self.start == self.end

    def _is_closable(self):
        end = self[-1].end
        for segment in self:
            if segment.start == end:
                return True
        return False

    @property
    def closed(self, warning_on=CLOSED_WARNING_ON):
        """The closed attribute is deprecated, please use the isclosed()
        method instead.  See _closed_warning for more information."""
        mes = ("This attribute is deprecated, consider using isclosed() "
               "method instead.\n\nThis attribute is kept for compatibility "
               "with scripts created using svg.path (v2.0). You can prevent "
               "this warning in the future by setting "
               "CLOSED_WARNING_ON=False.")
        if warning_on:
            warn(mes)
        return self._closed and self._is_closable()

    @closed.setter
    def closed(self, value):
        value = bool(value)
        if value and not self._is_closable():
            raise ValueError("End does not coincide with a segment start.")
        self._closed = value

    @property
    def start(self):
        if not self._start:
            self._start = self._segments[0].start
        return self._start

    @start.setter
    def start(self, pt):
        self._start = pt
        self._segments[0].start = pt

    @property
    def end(self):
        if not self._end:
            self._end = self._segments[-1].end
        return self._end

    @end.setter
    def end(self, pt):
        self._end = pt
        self._segments[-1].end = pt

    def d(self, useSandT=False, use_closed_attrib=False):
        """Returns a path d-string for the path object.
        For an explanation of useSandT and use_closed_attrib, see the
        compatibility notes in the README."""

        if use_closed_attrib:
            self_closed = self.closed(warning_on=False)
            if self_closed:
                segments = self[:-1]
            else:
                segments = self[:]
        else:
            self_closed = False
            segments = self[:]

        current_pos = None
        parts = []
        previous_segment = None
        end = self[-1].end

        for segment in segments:
            seg_start = segment.start
            # If the start of this segment does not coincide with the end of
            # the last segment or if this segment is actually the close point
            # of a closed path, then we should start a new subpath here.
            if current_pos != seg_start or \
                    (self_closed and seg_start == end and use_closed_attrib):
                parts.append('M {},{}'.format(seg_start.real, seg_start.imag))

            if isinstance(segment, Line):
                args = segment.end.real, segment.end.imag
                parts.append('L {},{}'.format(*args))
            elif isinstance(segment, CubicBezier):
                if useSandT and segment.is_smooth_from(previous_segment,
                                                       warning_on=False):
                    args = (segment.control2.real, segment.control2.imag,
                            segment.end.real, segment.end.imag)
                    parts.append('S {},{} {},{}'.format(*args))
                else:
                    args = (segment.control1.real, segment.control1.imag,
                            segment.control2.real, segment.control2.imag,
                            segment.end.real, segment.end.imag)
                    parts.append('C {},{} {},{} {},{}'.format(*args))
            elif isinstance(segment, QuadraticBezier):
                if useSandT and segment.is_smooth_from(previous_segment,
                                                       warning_on=False):
                    args = segment.end.real, segment.end.imag
                    parts.append('T {},{}'.format(*args))
                else:
                    args = (segment.control.real, segment.control.imag,
                            segment.end.real, segment.end.imag)
                    parts.append('Q {},{} {},{}'.format(*args))

            elif isinstance(segment, Arc):
                args = (segment.radius.real, segment.radius.imag,
                        segment.rotation,int(segment.large_arc),
                        int(segment.sweep),segment.end.real, segment.end.imag)
                parts.append('A {},{} {} {:d},{:d} {},{}'.format(*args))
            current_pos = segment.end
            previous_segment = segment

        if self_closed:
            parts.append('Z')

        return ' '.join(parts)

    def joins_smoothly_with(self, previous, wrt_parameterization=False):
        """Checks if this Path object joins smoothly with previous
        path/segment.  By default, this only checks that this Path starts
        moving (at t=0) in the same direction (and from the same positive) as
        previous stopped moving (at t=1).  To check if the tangent magnitudes
        also match, set wrt_parameterization=True."""
        if wrt_parameterization:
            return self[0].start == previous.end and self.derivative(
                0) == previous.derivative(1)
        else:
            return self[0].start == previous.end and self.unit_tangent(
                0) == previous.unit_tangent(1)

    def T2t(self, T):
        """returns the segment index, `seg_idx`, and segment parameter, `t`,
        corresponding to the path parameter `T`.  In other words, this is the
        inverse of the `Path.t2T()` method."""
        if T == 1:
            return len(self)-1, 1
        if T == 0:
            return 0, 0
        self._calc_lengths()
        # Find which segment self.point(T) falls on:
        T0 = 0  # the T-value the current segment starts on
        for seg_idx, seg_length in enumerate(self._lengths):
            T1 = T0 + seg_length  # the T-value the current segment ends on
            if T1 >= T:
                # This is the segment!
                t = (T - T0)/seg_length
                return seg_idx, t
            T0 = T1

        assert 0 <= T <= 1
        raise BugException

    def t2T(self, seg, t):
        """returns the path parameter T which corresponds to the segment
        parameter t.  In other words, for any Path object, path, and any
        segment in path, seg,  T(t) = path.t2T(seg, t) is the unique
        reparameterization such that path.point(T(t)) == seg.point(t) for all
        0 <= t <= 1.
        Input Note: seg can be a segment in the Path object or its
        corresponding index."""
        self._calc_lengths()
        # Accept an index or a segment for seg
        if isinstance(seg, int):
            seg_idx = seg
        else:
            try:
                seg_idx = self.index(seg)
            except ValueError:
                assert is_path_segment(seg) or isinstance(seg, int)
                raise

        segment_start = sum(self._lengths[:seg_idx])
        segment_end = segment_start + self._lengths[seg_idx]
        T = (segment_end - segment_start)*t + segment_start
        return T

    def derivative(self, T, n=1):
        """returns the tangent vector of the Path at T (centered at the origin
        and expressed as a complex number).
        Note: Bezier curves can have points where their derivative vanishes.
        If you are interested in the tangent direction, use unit_tangent()
        method instead."""
        seg_idx, t = self.T2t(T)
        seg = self._segments[seg_idx]
        return seg.derivative(t, n=n)/seg.length()**n

    def unit_tangent(self, T):
        """returns the unit tangent vector of the Path at T (centered at the
        origin and expressed as a complex number).  If the tangent vector's
        magnitude is zero, this method will find the limit of
        self.derivative(tau)/abs(self.derivative(tau)) as tau approaches T."""
        seg_idx, t = self.T2t(T)
        return self._segments[seg_idx].unit_tangent(t)

    def normal(self, t):
        """returns the (right hand rule) unit normal vector to self at t."""
        return -1j*self.unit_tangent(t)

    def curvature(self, T):
        """returns the curvature of this Path object at T and outputs
        float('inf') if not differentiable at T."""
        seg_idx, t = self.T2t(T)
        seg = self[seg_idx]
        if np.isclose(t, 0) and (seg_idx != 0 or self.end==self.start):
            previous_seg_in_path = self._segments[
                (seg_idx - 1) % len(self._segments)]
            if not seg.joins_smoothly_with(previous_seg_in_path):
                return float('inf')
        elif np.isclose(t, 1) and (seg_idx != len(self) - 1 or
                                   self.end == self.start):
            next_seg_in_path = self._segments[
                (seg_idx + 1) % len(self._segments)]
            if not next_seg_in_path.joins_smoothly_with(seg):
                return float('inf')
        dz = self.derivative(T)
        ddz = self.derivative(T, n=2)
        dx, dy = dz.real, dz.imag
        ddx, ddy = ddz.real, ddz.imag
        return abs(dx*ddy - dy*ddx)/(dx*dx + dy*dy)**1.5

    # def icurvature(self, kappa):
    #     """returns a list of T-values such that 0 <= T <= 1 and
    #     seg.curvature(t) = kappa.
    #     Note: not implemented for paths containing Arc segments."""
    #     assert is_bezier_path(self)
    #     Ts = []
    #     for i, seg in enumerate(self):
    #         Ts += [self.t2T(i, t) for t in seg.icurvature(kappa)]
    #     return Ts

    def area(self, chord_length=1e-4):
        """Find area enclosed by path.
        
        Approximates any Arc segments in the Path with lines
        approximately `chord_length` long, and returns the area enclosed
        by the approximated Path.  Default chord length is 0.01.  If Arc
        segments are included in path, to ensure accurate results, make
        sure this `chord_length` is set to a reasonable value (e.g. by
        checking curvature).
                
        Notes
        -----
        * Negative area results from clockwise (as opposed to
        counter-clockwise) parameterization of the input Path.
        
        To Contributors
        ---------------
        This is one of many parts of `svgpathtools` that could be 
        improved by a noble soul implementing a piecewise-linear 
        approximation scheme for paths (one with controls to guarantee a
        desired accuracy).
        """

        def area_without_arcs(path):
            area_enclosed = 0
            for seg in path:
                x = real(seg.poly())
                dy = imag(seg.poly()).deriv()
                integrand = x*dy
                integral = integrand.integ()
                area_enclosed += integral(1) - integral(0)
            return area_enclosed

        def seg2lines(seg):
            """Find piecewise-linear approximation of `seg`."""
            num_lines = int(ceil(seg.length() / chord_length))
            pts = [seg.point(t) for t in np.linspace(0, 1, num_lines+1)]
            return [Line(pts[i], pts[i+1]) for i in range(num_lines)]

        assert self.isclosed()

        bezier_path_approximation = []
        for seg in self:
            if isinstance(seg, Arc):
                bezier_path_approximation += seg2lines(seg)
            else:
                bezier_path_approximation.append(seg)
        return area_without_arcs(Path(*bezier_path_approximation))

    def intersect(self, other_curve, justonemode=False, tol=1e-12):
        """returns list of pairs of pairs ((T1, seg1, t1), (T2, seg2, t2))
        giving the intersection points.
        If justonemode==True, then returns just the first
        intersection found.
        tol is used to check for redundant intersections (see comment above
        the code block where tol is used).
        Note:  If the two path objects coincide for more than a finite set of
        points, this code will fail."""
        path1 = self
        if isinstance(other_curve, Path):
            path2 = other_curve
        else:
            path2 = Path(other_curve)
        assert path1 != path2
        intersection_list = []
        for seg1 in path1:
            for seg2 in path2:
                if justonemode and intersection_list:
                    return intersection_list[0]
                for t1, t2 in seg1.intersect(seg2, tol=tol):
                    T1 = path1.t2T(seg1, t1)
                    T2 = path2.t2T(seg2, t2)
                    intersection_list.append(((T1, seg1, t1), (T2, seg2, t2)))
        if justonemode and intersection_list:
            return intersection_list[0]

        # Note: If the intersection takes place at a joint (point one seg ends
        # and next begins in path) then intersection_list may contain a
        # redundant intersection.  This code block checks for and removes said
        # redundancies.
        if intersection_list:
            pts = [seg1.point(_t1)
                   for _T1, _seg1, _t1 in list(zip(*intersection_list))[0]]
            indices2remove = []
            for ind1 in range(len(pts)):
                for ind2 in range(ind1 + 1, len(pts)):
                    if abs(pts[ind1] - pts[ind2]) < tol:
                        # then there's a redundancy. Remove it.
                        indices2remove.append(ind2)
            intersection_list = [inter for ind, inter in
                                 enumerate(intersection_list) if
                                 ind not in indices2remove]
        return intersection_list

    def bbox(self):
        """returns a bounding box for the input Path object in the form
        (xmin, xmax, ymin, ymax)."""
        bbs = [seg.bbox() for seg in self._segments]
        xmins, xmaxs, ymins, ymaxs = list(zip(*bbs))
        xmin = min(xmins)
        xmax = max(xmaxs)
        ymin = min(ymins)
        ymax = max(ymaxs)
        return xmin, xmax, ymin, ymax

    def cropped(self, T0, T1):
        """returns a cropped copy of the path."""
        assert 0 <= T0 <= 1 and 0 <= T1<= 1
        assert T0 != T1
        assert not (T0 == 1 and T1 == 0)

        if T0 == 1 and 0 < T1 < 1 and self.isclosed():
            return self.cropped(0, T1)

        if T1 == 1:
            seg1 = self[-1]
            t_seg1 = 1
            i1 = len(self) - 1
        else:
            seg1_idx, t_seg1 = self.T2t(T1)
            seg1 = self[seg1_idx]
            if np.isclose(t_seg1, 0):
                i1 = (self.index(seg1) - 1) % len(self)
                seg1 = self[i1]
                t_seg1 = 1
            else:
                i1 = self.index(seg1)
        if T0 == 0:
            seg0 = self[0]
            t_seg0 = 0
            i0 = 0
        else:
            seg0_idx, t_seg0 = self.T2t(T0)
            seg0 = self[seg0_idx]
            if np.isclose(t_seg0, 1):
                i0 = (self.index(seg0) + 1) % len(self)
                seg0 = self[i0]
                t_seg0 = 0
            else:
                i0 = self.index(seg0)

        if T0 < T1 and i0 == i1:
            new_path = Path(seg0.cropped(t_seg0, t_seg1))
        else:
            new_path = Path(seg0.cropped(t_seg0, 1))

            # T1<T0 must cross discontinuity case
            if T1 < T0:
                if not self.isclosed():
                    raise ValueError("This path is not closed, thus T0 must "
                                     "be less than T1.")
                else:
                    for i in range(i0 + 1, len(self)):
                        new_path.append(self[i])
                    for i in range(0, i1):
                        new_path.append(self[i])

            # T0<T1 straight-forward case
            else:
                for i in range(i0 + 1, i1):
                    new_path.append(self[i])

            if t_seg1 != 0:
                new_path.append(seg1.cropped(0, t_seg1))
        return new_path

    def radialrange(self, origin, return_all_global_extrema=False):
        """returns the tuples (d_min, t_min, idx_min), (d_max, t_max, idx_max)
        which minimize and maximize, respectively, the distance
        d = |self[idx].point(t)-origin|."""
        if return_all_global_extrema:
            raise NotImplementedError
        else:
            global_min = (np.inf, None, None)
            global_max = (0, None, None)
            for seg_idx, seg in enumerate(self):
                seg_global_min, seg_global_max = seg.radialrange(origin)
                if seg_global_min[0] < global_min[0]:
                    global_min = seg_global_min + (seg_idx,)
                if seg_global_max[0] > global_max[0]:
                    global_max = seg_global_max + (seg_idx,)
            return global_min, global_max

    def rotated(self, degs, origin=None):
        """Returns a copy of self rotated by `degs` degrees (CCW) around the
        point `origin` (a complex number).  By default `origin` is either
        `self.point(0.5)`, or in the case that self is an Arc object,
        `origin` defaults to `self.center`."""
        return rotate(self, degs, origin=origin)

    def translated(self, z0):
        """Returns a copy of self shifted by the complex quantity `z0` such
        that self.translated(z0).point(t) = self.point(t) + z0 for any t."""
        return translate(self, z0)

    def scaled(self, sx, sy=None, origin=0j):
        """Scale transform.  See `scale` function for further explanation."""
        return scale(self, sx=sx, sy=sy, origin=origin)

"""This submodule contains tools for creating svg files from paths and path
segments."""

# External dependencies:
from __future__ import division, absolute_import, print_function
from math import ceil
from os import getcwd, path as os_path, makedirs
from xml.dom.minidom import parse as md_xml_parse
from svgwrite import Drawing, text as txt
from time import time
from warnings import warn
import re

# Internal dependencies
from .path import Path, Line, is_path_segment
from .misctools import open_in_browser

# Used to convert a string colors (identified by single chars) to a list.
color_dict = {'a': 'aqua',
              'b': 'blue',
              'c': 'cyan',
              'd': 'darkblue',
              'e': '',
              'f': '',
              'g': 'green',
              'h': '',
              'i': '',
              'j': '',
              'k': 'black',
              'l': 'lime',
              'm': 'magenta',
              'n': 'brown',
              'o': 'orange',
              'p': 'pink',
              'q': 'turquoise',
              'r': 'red',
              's': 'salmon',
              't': 'tan',
              'u': 'purple',
              'v': 'violet',
              'w': 'white',
              'x': '',
              'y': 'yellow',
              'z': 'azure'}


def str2colorlist(s, default_color=None):
    color_list = [color_dict[ch] for ch in s]
    if default_color:
        for idx, c in enumerate(color_list):
            if not c:
                color_list[idx] = default_color
    return color_list


def is3tuple(c):
    return isinstance(c, tuple) and len(c) == 3


def big_bounding_box(paths_n_stuff):
    """Finds a BB containing a collection of paths, Bezier path segments, and
    points (given as complex numbers)."""
    bbs = []
    for thing in paths_n_stuff:
        if is_path_segment(thing) or isinstance(thing, Path):
            bbs.append(thing.bbox())
        elif isinstance(thing, complex):
            bbs.append((thing.real, thing.real, thing.imag, thing.imag))
        else:
            try:
                complexthing = complex(thing)
                bbs.append((complexthing.real, complexthing.real,
                            complexthing.imag, complexthing.imag))
            except ValueError:
                raise TypeError(
                    "paths_n_stuff can only contains Path, CubicBezier, "
                    "QuadraticBezier, Line, and complex objects.")
    xmins, xmaxs, ymins, ymaxs = list(zip(*bbs))
    xmin = min(xmins)
    xmax = max(xmaxs)
    ymin = min(ymins)
    ymax = max(ymaxs)
    return xmin, xmax, ymin, ymax


def disvg(paths=None, colors=None,
          filename=os_path.join(getcwd(), 'disvg_output.svg'),
          stroke_widths=None, nodes=None, node_colors=None, node_radii=None,
          openinbrowser=True, timestamp=False,
          margin_size=0.1, mindim=600, dimensions=None,
          viewbox=None, text=None, text_path=None, font_size=None,
          attributes=None, svg_attributes=None, svgwrite_debug=False, paths2Drawing=False):
    """Takes in a list of paths and creates an SVG file containing said paths.
    REQUIRED INPUTS:
        :param paths - a list of paths

    OPTIONAL INPUT:
        :param colors - specifies the path stroke color.  By default all paths
        will be black (#000000).  This paramater can be input in a few ways
        1) a list of strings that will be input into the path elements stroke
            attribute (so anything that is understood by the svg viewer).
        2) a string of single character colors -- e.g. setting colors='rrr' is
            equivalent to setting colors=['red', 'red', 'red'] (see the
            'color_dict' dictionary above for a list of possibilities).
        3) a list of rgb 3-tuples -- e.g. colors = [(255, 0, 0), ...].

        :param filename - the desired location/filename of the SVG file
        created (by default the SVG will be stored in the current working
        directory and named 'disvg_output.svg').

        :param stroke_widths - a list of stroke_widths to use for paths
        (default is 0.5% of the SVG's width or length)

        :param nodes - a list of points to draw as filled-in circles

        :param node_colors - a list of colors to use for the nodes (by default
        nodes will be red)

        :param node_radii - a list of radii to use for the nodes (by default
        nodes will be radius will be 1 percent of the svg's width/length)

        :param text - string or list of strings to be displayed

        :param text_path - if text is a list, then this should be a list of
        path (or path segments of the same length.  Note: the path must be
        long enough to display the text or the text will be cropped by the svg
        viewer.

        :param font_size - a single float of list of floats.

        :param openinbrowser -  Set to True to automatically open the created
        SVG in the user's default web browser.

        :param timestamp - if True, then the a timestamp will be appended to
        the output SVG's filename.  This will fix issues with rapidly opening
        multiple SVGs in your browser.

        :param margin_size - The min margin (empty area framing the collection
        of paths) size used for creating the canvas and background of the SVG.

        :param mindim - The minimum dimension (height or width) of the output
        SVG (default is 600).

        :param dimensions - The (x,y) display dimensions of the output SVG.
        I.e. this specifies the `width` and `height` SVG attributes. Note that 
        these also can be used to specify units other than pixels. Using this 
        will override the `mindim` parameter.

        :param viewbox - This specifies the coordinated system used in the svg.
        The SVG `viewBox` attribute works together with the the `height` and 
        `width` attrinutes.  Using these three attributes allows for shifting 
        and scaling of the SVG canvas without changing the any values other 
        than those in `viewBox`, `height`, and `width`.  `viewbox` should be 
        input as a 4-tuple, (min_x, min_y, width, height), or a string 
        "min_x min_y width height".  Using this will override the `mindim` 
        parameter.

        :param attributes - a list of dictionaries of attributes for the input
        paths.  Note: This will override any other conflicting settings.

        :param svg_attributes - a dictionary of attributes for output svg.
        
        :param svgwrite_debug - This parameter turns on/off `svgwrite`'s 
        debugging mode.  By default svgwrite_debug=False.  This increases 
        speed and also prevents `svgwrite` from raising of an error when not 
        all `svg_attributes` key-value pairs are understood.
        
        :param paths2Drawing - If true, an `svgwrite.Drawing` object is 
        returned and no file is written.  This `Drawing` can later be saved 
        using the `svgwrite.Drawing.save()` method.

    NOTES:
        * The `svg_attributes` parameter will override any other conflicting 
        settings.

        * Any `extra` parameters that `svgwrite.Drawing()` accepts can be 
        controlled by passing them in through `svg_attributes`.

        * The unit of length here is assumed to be pixels in all variables.

        * If this function is used multiple times in quick succession to
        display multiple SVGs (all using the default filename), the
        svgviewer/browser will likely fail to load some of the SVGs in time.
        To fix this, use the timestamp attribute, or give the files unique
        names, or use a pause command (e.g. time.sleep(1)) between uses.
    """


    _default_relative_node_radius = 5e-3
    _default_relative_stroke_width = 1e-3
    _default_path_color = '#000000'  # black
    _default_node_color = '#ff0000'  # red
    _default_font_size = 12


    # append directory to filename (if not included)
    if os_path.dirname(filename) == '':
        filename = os_path.join(getcwd(), filename)

    # append time stamp to filename
    if timestamp:
        fbname, fext = os_path.splitext(filename)
        dirname = os_path.dirname(filename)
        tstamp = str(time()).replace('.', '')
        stfilename = os_path.split(fbname)[1] + '_' + tstamp + fext
        filename = os_path.join(dirname, stfilename)

    # check paths and colors are set
    if isinstance(paths, Path) or is_path_segment(paths):
        paths = [paths]
    if paths:
        if not colors:
            colors = [_default_path_color] * len(paths)
        else:
            assert len(colors) == len(paths)
            if isinstance(colors, str):
                colors = str2colorlist(colors,
                                       default_color=_default_path_color)
            elif isinstance(colors, list):
                for idx, c in enumerate(colors):
                    if is3tuple(c):
                        colors[idx] = "rgb" + str(c)

    # check nodes and nodes_colors are set (node_radii are set later)
    if nodes:
        if not node_colors:
            node_colors = [_default_node_color] * len(nodes)
        else:
            assert len(node_colors) == len(nodes)
            if isinstance(node_colors, str):
                node_colors = str2colorlist(node_colors,
                                            default_color=_default_node_color)
            elif isinstance(node_colors, list):
                for idx, c in enumerate(node_colors):
                    if is3tuple(c):
                        node_colors[idx] = "rgb" + str(c)

    # set up the viewBox and display dimensions of the output SVG
    # along the way, set stroke_widths and node_radii if not provided
    assert paths or nodes
    stuff2bound = []
    if viewbox:
        if not isinstance(viewbox, str):
            viewbox = '%s %s %s %s' % viewbox
        if dimensions is None:
            dimensions = viewbox.split(' ')[2:4]
    elif dimensions:
        dimensions = tuple(map(str, dimensions))
        def strip_units(s):
            return re.search(r'\d*\.?\d*', s.strip()).group()
        viewbox = '0 0 %s %s' % tuple(map(strip_units, dimensions))
    else:
        if paths:
            stuff2bound += paths
        if nodes:
            stuff2bound += nodes
        if text_path:
            stuff2bound += text_path
        xmin, xmax, ymin, ymax = big_bounding_box(stuff2bound)
        dx = xmax - xmin
        dy = ymax - ymin

        if dx == 0:
            dx = 1
        if dy == 0:
            dy = 1

        # determine stroke_widths to use (if not provided) and max_stroke_width
        if paths:
            if not stroke_widths:
                sw = max(dx, dy) * _default_relative_stroke_width
                stroke_widths = [sw]*len(paths)
                max_stroke_width = sw
            else:
                assert len(paths) == len(stroke_widths)
                max_stroke_width = max(stroke_widths)
        else:
            max_stroke_width = 0

        # determine node_radii to use (if not provided) and max_node_diameter
        if nodes:
            if not node_radii:
                r = max(dx, dy) * _default_relative_node_radius
                node_radii = [r]*len(nodes)
                max_node_diameter = 2*r
            else:
                assert len(nodes) == len(node_radii)
                max_node_diameter = 2*max(node_radii)
        else:
            max_node_diameter = 0

        extra_space_for_style = max(max_stroke_width, max_node_diameter)
        xmin -= margin_size*dx + extra_space_for_style/2
        ymin -= margin_size*dy + extra_space_for_style/2
        dx += 2*margin_size*dx + extra_space_for_style
        dy += 2*margin_size*dy + extra_space_for_style
        viewbox = "%s %s %s %s" % (xmin, ymin, dx, dy)

        if dx > dy:
            szx = str(mindim) + 'px'
            szy = str(int(ceil(mindim * dy / dx))) + 'px'
        else:
            szx = str(int(ceil(mindim * dx / dy))) + 'px'
            szy = str(mindim) + 'px'
        dimensions = szx, szy

    # Create an SVG file
    if svg_attributes is not None:
        dimensions = (svg_attributes.get("width", dimensions[0]),
                      svg_attributes.get("height", dimensions[1]))
        debug = svg_attributes.get("debug", svgwrite_debug)
        dwg = Drawing(filename=filename, size=dimensions, debug=debug,
                      **svg_attributes)
    else:
        dwg = Drawing(filename=filename, size=dimensions, debug=svgwrite_debug,
                      viewBox=viewbox)

    # add paths
    if paths:
        for i, p in enumerate(paths):
            if isinstance(p, Path):
                ps = p.d()
            elif is_path_segment(p):
                ps = Path(p).d()
            else:  # assume this path, p, was input as a Path d-string
                ps = p

            if attributes:
                good_attribs = {'d': ps}
                for key in attributes[i]:
                    val = attributes[i][key]
                    if key != 'd':
                        try:
                            dwg.path(ps, **{key: val})
                            good_attribs.update({key: val})
                        except Exception as e:
                            warn(str(e))

                dwg.add(dwg.path(**good_attribs))
            else:
                dwg.add(dwg.path(ps, stroke=colors[i],
                                 stroke_width=str(stroke_widths[i]),
                                 fill='none'))

    # add nodes (filled in circles)
    if nodes:
        for i_pt, pt in enumerate([(z.real, z.imag) for z in nodes]):
            dwg.add(dwg.circle(pt, node_radii[i_pt], fill=node_colors[i_pt]))

    # add texts
    if text:
        assert isinstance(text, str) or (isinstance(text, list) and
                                         isinstance(text_path, list) and
                                         len(text_path) == len(text))
        if isinstance(text, str):
            text = [text]
            if not font_size:
                font_size = [_default_font_size]
            if not text_path:
                pos = complex(xmin + margin_size*dx, ymin + margin_size*dy)
                text_path = [Line(pos, pos + 1).d()]
        else:
            if font_size:
                if isinstance(font_size, list):
                    assert len(font_size) == len(text)
                else:
                    font_size = [font_size] * len(text)
            else:
                font_size = [_default_font_size] * len(text)
        for idx, s in enumerate(text):
            p = text_path[idx]
            if isinstance(p, Path):
                ps = p.d()
            elif is_path_segment(p):
                ps = Path(p).d()
            else:  # assume this path, p, was input as a Path d-string
                ps = p

            # paragraph = dwg.add(dwg.g(font_size=font_size[idx]))
            # paragraph.add(dwg.textPath(ps, s))
            pathid = 'tp' + str(idx)
            dwg.defs.add(dwg.path(d=ps, id=pathid))
            txter = dwg.add(dwg.text('', font_size=font_size[idx]))
            txter.add(txt.TextPath('#'+pathid, s))

    if paths2Drawing:
        return dwg
      
    # save svg
    if not os_path.exists(os_path.dirname(filename)):
        makedirs(os_path.dirname(filename))
    dwg.save()

    # re-open the svg, make the xml pretty, and save it again
    xmlstring = md_xml_parse(filename).toprettyxml()
    with open(filename, 'w') as f:
        f.write(xmlstring)

    # try to open in web browser
    if openinbrowser:
        try:
            open_in_browser(filename)
        except:
            print("Failed to open output SVG in browser.  SVG saved to:")
            print(filename)


def wsvg(paths=None, colors=None,
          filename=os_path.join(getcwd(), 'disvg_output.svg'),
          stroke_widths=None, nodes=None, node_colors=None, node_radii=None,
          openinbrowser=False, timestamp=False,
          margin_size=0.1, mindim=600, dimensions=None,
          viewbox=None, text=None, text_path=None, font_size=None,
          attributes=None, svg_attributes=None, svgwrite_debug=False, paths2Drawing=False):
    """Convenience function; identical to disvg() except that
    openinbrowser=False by default.  See disvg() docstring for more info."""
    disvg(paths, colors=colors, filename=filename,
          stroke_widths=stroke_widths, nodes=nodes,
          node_colors=node_colors, node_radii=node_radii,
          openinbrowser=openinbrowser, timestamp=timestamp,
          margin_size=margin_size, mindim=mindim, dimensions=dimensions,
          viewbox=viewbox, text=text, text_path=text_path, font_size=font_size,
          attributes=attributes, svg_attributes=svg_attributes,
          svgwrite_debug=svgwrite_debug, paths2Drawing=paths2Drawing)
    
    
def paths2Drawing(paths=None, colors=None,
          filename=os_path.join(getcwd(), 'disvg_output.svg'),
          stroke_widths=None, nodes=None, node_colors=None, node_radii=None,
          openinbrowser=False, timestamp=False,
          margin_size=0.1, mindim=600, dimensions=None,
          viewbox=None, text=None, text_path=None, font_size=None,
          attributes=None, svg_attributes=None, svgwrite_debug=False, paths2Drawing=True):
    """Convenience function; identical to disvg() except that
    paths2Drawing=True by default.  See disvg() docstring for more info."""
    disvg(paths, colors=colors, filename=filename,
          stroke_widths=stroke_widths, nodes=nodes,
          node_colors=node_colors, node_radii=node_radii,
          openinbrowser=openinbrowser, timestamp=timestamp,
          margin_size=margin_size, mindim=mindim, dimensions=dimensions,
          viewbox=viewbox, text=text, text_path=text_path, font_size=font_size,
          attributes=attributes, svg_attributes=svg_attributes,
          svgwrite_debug=svgwrite_debug, paths2Drawing=paths2Drawing)

"""This submodule contains tools for working with numpy.poly1d objects."""

# External Dependencies
from __future__ import division, absolute_import
from itertools import combinations
import numpy as np

# Internal Dependencies
from .misctools import isclose


def polyroots(p, realroots=False, condition=lambda r: True):
    """
    Returns the roots of a polynomial with coefficients given in p.
      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    INPUT:
    p - Rank-1 array-like object of polynomial coefficients.
    realroots - a boolean.  If true, only real roots will be returned  and the
        condition function can be written assuming all roots are real.
    condition - a boolean-valued function.  Only roots satisfying this will be
        returned.  If realroots==True, these conditions should assume the roots
        are real.
    OUTPUT:
    A list containing the roots of the polynomial.
    NOTE:  This uses np.isclose and np.roots"""
    roots = np.roots(p)
    if realroots:
        roots = [r.real for r in roots if isclose(r.imag, 0)]
    roots = [r for r in roots if condition(r)]

    duplicates = []
    for idx, (r1, r2) in enumerate(combinations(roots, 2)):
        if isclose(r1, r2):
            duplicates.append(idx)
    return [r for idx, r in enumerate(roots) if idx not in duplicates]


def polyroots01(p):
    """Returns the real roots between 0 and 1 of the polynomial with
    coefficients given in p,
      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    p can also be a np.poly1d object.  See polyroots for more information."""
    return polyroots(p, realroots=True, condition=lambda tval: 0 <= tval <= 1)


def rational_limit(f, g, t0):
    """Computes the limit of the rational function (f/g)(t)
    as t approaches t0."""
    assert isinstance(f, np.poly1d) and isinstance(g, np.poly1d)
    assert g != np.poly1d([0])
    if g(t0) != 0:
        return f(t0)/g(t0)
    elif f(t0) == 0:
        return rational_limit(f.deriv(), g.deriv(), t0)
    else:
        raise ValueError("Limit does not exist.")


def real(z):
    try:
        return np.poly1d(z.coeffs.real)
    except AttributeError:
        return z.real


def imag(z):
    try:
        return np.poly1d(z.coeffs.imag)
    except AttributeError:
        return z.imag


def poly_real_part(poly):
    """Deprecated."""
    return np.poly1d(poly.coeffs.real)


def poly_imag_part(poly):
    """Deprecated."""
    return np.poly1d(poly.coeffs.imag)

"""This submodule contains functions related to smoothing paths of Bezier
curves."""

# External Dependencies
from __future__ import division, absolute_import, print_function

# Internal Dependencies
from .path import Path, CubicBezier, Line
from .misctools import isclose
from .paths2svg import disvg


def is_differentiable(path, tol=1e-8):
    for idx in range(len(path)):
        u = path[(idx-1) % len(path)].unit_tangent(1)
        v = path[idx].unit_tangent(0)
        u_dot_v = u.real*v.real + u.imag*v.imag
        if abs(u_dot_v - 1) > tol:
            return False
    return True


def kinks(path, tol=1e-8):
    """returns indices of segments that start on a non-differentiable joint."""
    kink_list = []
    for idx in range(len(path)):
        if idx == 0 and not path.isclosed():
            continue
        try:
            u = path[(idx - 1) % len(path)].unit_tangent(1)
            v = path[idx].unit_tangent(0)
            u_dot_v = u.real*v.real + u.imag*v.imag
            flag = False
        except ValueError:
            flag = True

        if flag or abs(u_dot_v - 1) > tol:
            kink_list.append(idx)
    return kink_list


def _report_unfixable_kinks(_path, _kink_list):
    mes = ("\n%s kinks have been detected at that cannot be smoothed.\n"
           "To ignore these kinks and fix all others, run this function "
           "again with the second argument 'ignore_unfixable_kinks=True' "
           "The locations of the unfixable kinks are at the beginnings of "
           "segments: %s" % (len(_kink_list), _kink_list))
    disvg(_path, nodes=[_path[idx].start for idx in _kink_list])
    raise Exception(mes)


def smoothed_joint(seg0, seg1, maxjointsize=3, tightness=1.99):
    """ See Andy's notes on
    Smoothing Bezier Paths for an explanation of the method.
    Input: two segments seg0, seg1 such that seg0.end==seg1.start, and
    jointsize, a positive number

    Output: seg0_trimmed, elbow, seg1_trimmed, where elbow is a cubic bezier
        object that smoothly connects seg0_trimmed and seg1_trimmed.

    """
    assert seg0.end == seg1.start
    assert 0 < maxjointsize
    assert 0 < tightness < 2
#    sgn = lambda x:x/abs(x)
    q = seg0.end

    try: v = seg0.unit_tangent(1)
    except: v = seg0.unit_tangent(1 - 1e-4)
    try: w = seg1.unit_tangent(0)
    except: w = seg1.unit_tangent(1e-4)

    max_a = maxjointsize / 2
    a = min(max_a, min(seg1.length(), seg0.length()) / 20)
    if isinstance(seg0, Line) and isinstance(seg1, Line):
        '''
        Note: Letting
            c(t) = elbow.point(t), v= the unit tangent of seg0 at 1, w = the
            unit tangent vector of seg1 at 0,
            Q = seg0.point(1) = seg1.point(0), and a,b>0 some constants.
            The elbow will be the unique CubicBezier, c, such that
            c(0)= Q-av, c(1)=Q+aw, c'(0) = bv, and c'(1) = bw
            where a and b are derived above/below from tightness and
            maxjointsize.
        '''
#        det = v.imag*w.real-v.real*w.imag
        # Note:
        # If det is negative, the curvature of elbow is negative for all
        # real t if and only if b/a > 6
        # If det is positive, the curvature of elbow is negative for all
        # real t if and only if b/a < 2

#        if det < 0:
#            b = (6+tightness)*a
#        elif det > 0:
#            b = (2-tightness)*a
#        else:
#            raise Exception("seg0 and seg1 are parallel lines.")
        b = (2 - tightness)*a
        elbow = CubicBezier(q - a*v, q - (a - b/3)*v, q + (a - b/3)*w, q + a*w)
        seg0_trimmed = Line(seg0.start, elbow.start)
        seg1_trimmed = Line(elbow.end, seg1.end)
        return seg0_trimmed, [elbow], seg1_trimmed
    elif isinstance(seg0, Line):
        '''
        Note: Letting
            c(t) = elbow.point(t), v= the unit tangent of seg0 at 1,
            w = the unit tangent vector of seg1 at 0,
            Q = seg0.point(1) = seg1.point(0), and a,b>0 some constants.
            The elbow will be the unique CubicBezier, c, such that
            c(0)= Q-av, c(1)=Q, c'(0) = bv, and c'(1) = bw
            where a and b are derived above/below from tightness and
            maxjointsize.
        '''
#        det = v.imag*w.real-v.real*w.imag
        # Note: If g has the same sign as det, then the curvature of elbow is
        # negative for all real t if and only if b/a < 4
        b = (4 - tightness)*a
#        g = sgn(det)*b
        elbow = CubicBezier(q - a*v, q + (b/3 - a)*v, q - b/3*w, q)
        seg0_trimmed = Line(seg0.start, elbow.start)
        return seg0_trimmed, [elbow], seg1
    elif isinstance(seg1, Line):
        args = (seg1.reversed(), seg0.reversed(), maxjointsize, tightness)
        rseg1_trimmed, relbow, rseg0 = smoothed_joint(*args)
        elbow = relbow[0].reversed()
        return seg0, [elbow], rseg1_trimmed.reversed()
    else:
        # find a point on each seg that is about a/2 away from joint.  Make
        # line between them.
        t0 = seg0.ilength(seg0.length() - a/2)
        t1 = seg1.ilength(a/2)
        seg0_trimmed = seg0.cropped(0, t0)
        seg1_trimmed = seg1.cropped(t1, 1)
        seg0_line = Line(seg0_trimmed.end, q)
        seg1_line = Line(q, seg1_trimmed.start)

        args = (seg0_trimmed, seg0_line, maxjointsize, tightness)
        dummy, elbow0, seg0_line_trimmed = smoothed_joint(*args)

        args = (seg1_line, seg1_trimmed, maxjointsize, tightness)
        seg1_line_trimmed, elbow1, dummy = smoothed_joint(*args)

        args = (seg0_line_trimmed, seg1_line_trimmed, maxjointsize, tightness)
        seg0_line_trimmed, elbowq, seg1_line_trimmed = smoothed_joint(*args)

        elbow = elbow0 + [seg0_line_trimmed] + elbowq + [seg1_line_trimmed] + elbow1
        return seg0_trimmed, elbow, seg1_trimmed


def smoothed_path(path, maxjointsize=3, tightness=1.99, ignore_unfixable_kinks=False):
    """returns a path with no non-differentiable joints."""
    if len(path) == 1:
        return path

    assert path.iscontinuous()

    sharp_kinks = []
    new_path = [path[0]]
    for idx in range(len(path)):
        if idx == len(path)-1:
            if not path.isclosed():
                continue
            else:
                seg1 = new_path[0]
        else:
            seg1 = path[idx + 1]
        seg0 = new_path[-1]

        try:
            unit_tangent0 = seg0.unit_tangent(1)
            unit_tangent1 = seg1.unit_tangent(0)
            flag = False
        except ValueError:
            flag = True  # unit tangent not well-defined

        if not flag and isclose(unit_tangent0, unit_tangent1):  # joint is already smooth
            if idx != len(path)-1:
                new_path.append(seg1)
            continue
        else:
            kink_idx = (idx + 1) % len(path)  # kink at start of this seg
            if not flag and isclose(-unit_tangent0, unit_tangent1):
                # joint is sharp 180 deg (must be fixed manually)
                new_path.append(seg1)
                sharp_kinks.append(kink_idx)
            else:  # joint is not smooth, let's  smooth it.
                args = (seg0, seg1, maxjointsize, tightness)
                new_seg0, elbow_segs, new_seg1 = smoothed_joint(*args)
                new_path[-1] = new_seg0
                new_path += elbow_segs
                if idx == len(path) - 1:
                    new_path[0] = new_seg1
                else:
                    new_path.append(new_seg1)

    # If unfixable kinks were found, let the user know
    if sharp_kinks and not ignore_unfixable_kinks:
        _report_unfixable_kinks(path, sharp_kinks)

    return Path(*new_path)

"""(Experimental) replacement for import/export functionality SAX

"""

# External dependencies
from __future__ import division, absolute_import, print_function
import os
from xml.etree.ElementTree import iterparse, Element, ElementTree, SubElement

# Internal dependencies
from .parser import parse_path
from .parser import parse_transform
from .svg_to_paths import (path2pathd, ellipse2pathd, line2pathd,
                           polyline2pathd, polygon2pathd, rect2pathd)
from .misctools import open_in_browser
from .path import *

# To maintain forward/backward compatibility
try:
    str = basestring
except NameError:
    pass

NAME_SVG = "svg"
ATTR_VERSION = "version"
VALUE_SVG_VERSION = "1.1"
ATTR_XMLNS = "xmlns"
VALUE_XMLNS = "http://www.w3.org/2000/svg"
ATTR_XMLNS_LINK = "xmlns:xlink"
VALUE_XLINK = "http://www.w3.org/1999/xlink"
ATTR_XMLNS_EV = "xmlns:ev"
VALUE_XMLNS_EV = "http://www.w3.org/2001/xml-events"
ATTR_WIDTH = "width"
ATTR_HEIGHT = "height"
ATTR_VIEWBOX = "viewBox"
NAME_PATH = "path"
ATTR_DATA = "d"
ATTR_FILL = "fill"
ATTR_STROKE = "stroke"
ATTR_STROKE_WIDTH = "stroke-width"
ATTR_TRANSFORM = "transform"
VALUE_NONE = "none"


class SaxDocument:
    def __init__(self, filename):
        """A container for a SAX SVG light tree objects document.

        This class provides functions for extracting SVG data into Path objects.

        Args:
            filename (str): The filename of the SVG file
        """
        self.root_values = {}
        self.tree = []
        # remember location of original svg file
        if filename is not None and os.path.dirname(filename) == '':
            self.original_filename = os.path.join(os.getcwd(), filename)
        else:
            self.original_filename = filename

        if filename is not None:
            self.sax_parse(filename)

    def sax_parse(self, filename):
        self.root_values = {}
        self.tree = []
        stack = []
        values = {}
        matrix = None
        for event, elem in iterparse(filename, events=('start', 'end')):
            if event == 'start':
                stack.append((values, matrix))
                if matrix is not None:
                    matrix = matrix.copy()  # copy of matrix
                current_values = values
                values = {}
                values.update(current_values)  # copy of dictionary
                attrs = elem.attrib
                values.update(attrs)
                name = elem.tag[28:]
                if "style" in attrs:
                    for equate in attrs["style"].split(";"):
                        equal_item = equate.split(":")
                        values[equal_item[0]] = equal_item[1]
                if "transform" in attrs:
                    transform_matrix = parse_transform(attrs["transform"])
                    if matrix is None:
                        matrix = np.identity(3)
                    matrix = transform_matrix.dot(matrix)
                if "svg" == name:
                    current_values = values
                    values = {}
                    values.update(current_values)
                    self.root_values = current_values
                    continue
                elif "g" == name:
                    continue
                elif 'path' == name:
                    values['d'] = path2pathd(values)
                elif 'circle' == name:
                    values["d"] = ellipse2pathd(values)
                elif 'ellipse' == name:
                    values["d"] = ellipse2pathd(values)
                elif 'line' == name:
                    values["d"] = line2pathd(values)
                elif 'polyline' == name:
                    values["d"] = polyline2pathd(values['points'])
                elif 'polygon' == name:
                    values["d"] = polygon2pathd(values['points'])
                elif 'rect' == name:
                    values["d"] = rect2pathd(values)
                else:
                    continue
                values["matrix"] = matrix
                values["name"] = name
                self.tree.append(values)
            else:
                v = stack.pop()
                values = v[0]
                matrix = v[1]

    def flatten_all_paths(self):
        flat = []
        for values in self.tree:
            pathd = values['d']
            matrix = values['matrix']
            parsed_path = parse_path(pathd)
            if matrix is not None:
                transform(parsed_path, matrix)
            flat.append(parsed_path)
        return flat

    def get_pathd_and_matrix(self):
        flat = []
        for values in self.tree:
            pathd = values['d']
            matrix = values['matrix']
            flat.append((pathd, matrix))
        return flat

    def generate_dom(self):
        root = Element(NAME_SVG)
        root.set(ATTR_VERSION, VALUE_SVG_VERSION)
        root.set(ATTR_XMLNS, VALUE_XMLNS)
        root.set(ATTR_XMLNS_LINK, VALUE_XLINK)
        root.set(ATTR_XMLNS_EV, VALUE_XMLNS_EV)
        width = self.root_values.get(ATTR_WIDTH, None)
        height = self.root_values.get(ATTR_HEIGHT, None)
        if width is not None:
            root.set(ATTR_WIDTH, width)
        if height is not None:
            root.set(ATTR_HEIGHT, height)
        viewbox = self.root_values.get(ATTR_VIEWBOX, None)
        if viewbox is not None:
            root.set(ATTR_VIEWBOX, viewbox)
        identity = np.identity(3)
        for values in self.tree:
            pathd = values.get('d', '')
            matrix = values.get('matrix', None)
            # path_value = parse_path(pathd)

            path = SubElement(root, NAME_PATH)
            if matrix is not None and not np.all(np.equal(matrix, identity)):
                matrix_string = "matrix("
                matrix_string += " "
                matrix_string += str(matrix[0][0])
                matrix_string += " "
                matrix_string += str(matrix[1][0])
                matrix_string += " "
                matrix_string += str(matrix[0][1])
                matrix_string += " "
                matrix_string += str(matrix[1][1])
                matrix_string += " "
                matrix_string += str(matrix[0][2])
                matrix_string += " "
                matrix_string += str(matrix[1][2])
                matrix_string += ")"
                path.set(ATTR_TRANSFORM, matrix_string)
            if ATTR_DATA in values:
                path.set(ATTR_DATA, values[ATTR_DATA])
            if ATTR_FILL in values:
                path.set(ATTR_FILL, values[ATTR_FILL])
            if ATTR_STROKE in values:
                path.set(ATTR_STROKE, values[ATTR_STROKE])
        return ElementTree(root)

    def save(self, filename):
        with open(filename, 'wb') as output_svg:
            dom_tree = self.generate_dom()
            dom_tree.write(output_svg)

    def display(self, filename=None):
        """Displays/opens the doc using the OS's default application."""
        if filename is None:
            filename = 'display_temp.svg'
        self.save(filename)
        open_in_browser(filename)

"""This submodule contains tools for creating path objects from SVG files.
The main tool being the svg2paths() function."""

# External dependencies
from __future__ import division, absolute_import, print_function
from xml.dom.minidom import parse
from os import path as os_path, getcwd
import re

# Internal dependencies
from .parser import parse_path


COORD_PAIR_TMPLT = re.compile(
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)' +
    r'(?:\s*,\s*|\s+|(?=-))' +
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)'
)

def path2pathd(path):
    return path.get('d', '')

def ellipse2pathd(ellipse):
    """converts the parameters from an ellipse or a circle to a string for a 
    Path object d-attribute"""

    cx = ellipse.get('cx', 0)
    cy = ellipse.get('cy', 0)
    rx = ellipse.get('rx', None)
    ry = ellipse.get('ry', None)
    r = ellipse.get('r', None)

    if r is not None:
        rx = ry = float(r)
    else:
        rx = float(rx)
        ry = float(ry)

    cx = float(cx)
    cy = float(cy)

    d = ''
    d += 'M' + str(cx - rx) + ',' + str(cy)
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(2 * rx) + ',0'
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(-2 * rx) + ',0'

    return d


def polyline2pathd(polyline_d, is_polygon=False):
    """converts the string from a polyline points-attribute to a string for a
    Path object d-attribute"""
    points = COORD_PAIR_TMPLT.findall(polyline_d)
    closed = (float(points[0][0]) == float(points[-1][0]) and
              float(points[0][1]) == float(points[-1][1]))

    # The `parse_path` call ignores redundant 'z' (closure) commands
    # e.g. `parse_path('M0 0L100 100Z') == parse_path('M0 0L100 100L0 0Z')`
    # This check ensures that an n-point polygon is converted to an n-Line path.
    if is_polygon and closed:
        points.append(points[0])

    d = 'M' + 'L'.join('{0} {1}'.format(x,y) for x,y in points)
    if is_polygon or closed:
        d += 'z'
    return d


def polygon2pathd(polyline_d):
    """converts the string from a polygon points-attribute to a string 
    for a Path object d-attribute.
    Note:  For a polygon made from n points, the resulting path will be
    composed of n lines (even if some of these lines have length zero).
    """
    return polyline2pathd(polyline_d, True)


def rect2pathd(rect):
    """Converts an SVG-rect element to a Path d-string.
    
    The rectangle will start at the (x,y) coordinate specified by the 
    rectangle object and proceed counter-clockwise."""
    x0, y0 = float(rect.get('x', 0)), float(rect.get('y', 0))
    w, h = float(rect.get('width', 0)), float(rect.get('height', 0))
    x1, y1 = x0 + w, y0
    x2, y2 = x0 + w, y0 + h
    x3, y3 = x0, y0 + h

    d = ("M{} {} L {} {} L {} {} L {} {} z"
         "".format(x0, y0, x1, y1, x2, y2, x3, y3))
    return d

def line2pathd(l):
    return 'M' + l['x1'] + ' ' + l['y1'] + 'L' + l['x2'] + ' ' + l['y2']

def svg2paths(svg_file_location,
              return_svg_attributes=False,
              convert_circles_to_paths=True,
              convert_ellipses_to_paths=True,
              convert_lines_to_paths=True,
              convert_polylines_to_paths=True,
              convert_polygons_to_paths=True,
              convert_rectangles_to_paths=True):
    """Converts an SVG into a list of Path objects and attribute dictionaries. 

    Converts an SVG file into a list of Path objects and a list of
    dictionaries containing their attributes.  This currently supports
    SVG Path, Line, Polyline, Polygon, Circle, and Ellipse elements.

    Args:
        svg_file_location (string): the location of the svg file
        return_svg_attributes (bool): Set to True and a dictionary of
            svg-attributes will be extracted and returned.  See also the 
            `svg2paths2()` function.
        convert_circles_to_paths: Set to False to exclude SVG-Circle
            elements (converted to Paths).  By default circles are included as 
            paths of two `Arc` objects.
        convert_ellipses_to_paths (bool): Set to False to exclude SVG-Ellipse
            elements (converted to Paths).  By default ellipses are included as 
            paths of two `Arc` objects.
        convert_lines_to_paths (bool): Set to False to exclude SVG-Line elements
            (converted to Paths)
        convert_polylines_to_paths (bool): Set to False to exclude SVG-Polyline
            elements (converted to Paths)
        convert_polygons_to_paths (bool): Set to False to exclude SVG-Polygon
            elements (converted to Paths)
        convert_rectangles_to_paths (bool): Set to False to exclude SVG-Rect
            elements (converted to Paths).

    Returns: 
        list: The list of Path objects.
        list: The list of corresponding path attribute dictionaries.
        dict (optional): A dictionary of svg-attributes (see `svg2paths2()`).
    """
    if os_path.dirname(svg_file_location) == '':
        svg_file_location = os_path.join(getcwd(), svg_file_location)

    doc = parse(svg_file_location)

    def dom2dict(element):
        """Converts DOM elements to dictionaries of attributes."""
        keys = list(element.attributes.keys())
        values = [val.value for val in list(element.attributes.values())]
        return dict(list(zip(keys, values)))

    # Use minidom to extract path strings from input SVG
    paths = [dom2dict(el) for el in doc.getElementsByTagName('path')]
    d_strings = [el['d'] for el in paths]
    attribute_dictionary_list = paths

    # Use minidom to extract polyline strings from input SVG, convert to
    # path strings, add to list
    if convert_polylines_to_paths:
        plins = [dom2dict(el) for el in doc.getElementsByTagName('polyline')]
        d_strings += [polyline2pathd(pl['points']) for pl in plins]
        attribute_dictionary_list += plins

    # Use minidom to extract polygon strings from input SVG, convert to
    # path strings, add to list
    if convert_polygons_to_paths:
        pgons = [dom2dict(el) for el in doc.getElementsByTagName('polygon')]
        d_strings += [polygon2pathd(pg['points']) for pg in pgons]
        attribute_dictionary_list += pgons

    if convert_lines_to_paths:
        lines = [dom2dict(el) for el in doc.getElementsByTagName('line')]
        d_strings += [('M' + l['x1'] + ' ' + l['y1'] +
                       'L' + l['x2'] + ' ' + l['y2']) for l in lines]
        attribute_dictionary_list += lines

    if convert_ellipses_to_paths:
        ellipses = [dom2dict(el) for el in doc.getElementsByTagName('ellipse')]
        d_strings += [ellipse2pathd(e) for e in ellipses]
        attribute_dictionary_list += ellipses

    if convert_circles_to_paths:
        circles = [dom2dict(el) for el in doc.getElementsByTagName('circle')]
        d_strings += [ellipse2pathd(c) for c in circles]
        attribute_dictionary_list += circles

    if convert_rectangles_to_paths:
        rectangles = [dom2dict(el) for el in doc.getElementsByTagName('rect')]
        d_strings += [rect2pathd(r) for r in rectangles]
        attribute_dictionary_list += rectangles

    if return_svg_attributes:
        svg_attributes = dom2dict(doc.getElementsByTagName('svg')[0])
        doc.unlink()
        path_list = [parse_path(d) for d in d_strings]
        return path_list, attribute_dictionary_list, svg_attributes
    else:
        doc.unlink()
        path_list = [parse_path(d) for d in d_strings]
        return path_list, attribute_dictionary_list


def svg2paths2(svg_file_location,
               return_svg_attributes=True,
               convert_circles_to_paths=True,
               convert_ellipses_to_paths=True,
               convert_lines_to_paths=True,
               convert_polylines_to_paths=True,
               convert_polygons_to_paths=True,
               convert_rectangles_to_paths=True):
    """Convenience function; identical to svg2paths() except that
    return_svg_attributes=True by default.  See svg2paths() docstring for more
    info."""
    return svg2paths(svg_file_location=svg_file_location,
                     return_svg_attributes=return_svg_attributes,
                     convert_circles_to_paths=convert_circles_to_paths,
                     convert_ellipses_to_paths=convert_ellipses_to_paths,
                     convert_lines_to_paths=convert_lines_to_paths,
                     convert_polylines_to_paths=convert_polylines_to_paths,
                     convert_polygons_to_paths=convert_polygons_to_paths,
                     convert_rectangles_to_paths=convert_rectangles_to_paths)

