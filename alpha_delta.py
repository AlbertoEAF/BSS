#! /usr/bin/python3

from math import *
import sys
from os import path
import re

from ConfigParser import *

from color_codes import *

def p(m): print("<",m,">",sep="")

def error(msg):
    print(msg)
    exit(1)

if len(sys.argv) != 2: 
    error("Usage: alpha_delta.py <file_in_structured_path>") 

def degrees2rad(degrees): 
    return pi/180.0 * degrees

def rad2degrees(rad): 
    return 180.0/pi * rad

def abs(v1,v2): # arithmetic with tuples to avoid importing numpy just for this
    if len(v1) == 2:
        return sqrt( (v2[0]-v1[0])**2 + (v2[1]-v1[1])**2 )
    elif len(v1) == 3:
        return sqrt( (v2[0]-v1[0])**2 + (v2[1]-v1[1])**2 + (v2[2]-v1[2])**2 )

filepath = sys.argv[1]

filedir = path.split(path.abspath(filepath))[0]


splitpath = filedir.split("/")

#try: 
#    setup = ConfigParser("/".join(splitpath[:-2])+"/setup")
#except:
#    print("Couldn't find setup file at ", "/".join(splitpath[:-2])+"/setup")
#    exit(1)
setup = ConfigParser("/".join(splitpath[:-2])+"/setup")
y_offset = setup.f('d_forward_cm') * 0.01
z_offset = setup.f('z_offset_cm' ) * 0.01
d_mics   = setup.f('d_mics_cm'   ) * 0.01


try:
    if (splitpath[-1] == "axis"):
        angle = 90.0
        y_offset = 0 # This were particular cases with true parameters. Speaker aligned to the mic heads!
    elif splitpath[-1] == "-axis":
        angle = -90.0
        y_offset = 0 # This were particular cases with true parameters. Speaker aligned to the mic heads!
    else:
        angle = float( re.search('^(-?\d+\.?\d*)o$', splitpath[-1]).groups()[0] )
except:
    error("Invalid angle in structured directory path:" + filepath)

try:
    r = float( re.search('(-?\d+\.?\d*)m$', splitpath[-2]).groups()[0] )
except:
    error("Invalid distance in structured directory path: " + filepath)







# Coordinate system where the angle 0 is aligned with y (system with preferred central alignment)
x = r * sin( degrees2rad(angle) )
y = r * cos( degrees2rad(angle) )

# Source position (in the referential where the origin is the center of the microphone system)
s_x = x
s_y = y - y_offset # In real systems due to the small microphone support the microphones had to be aligned forward and away from the y-axis origin # We're correcting the angles for this.

true_r     = abs( (s_x,s_y) , (0,0) )
true_angle = atan( s_x / s_y )

# Mic positions
x2 = 0.5 * d_mics
x1 = -x2
y1 = y2 = 0 # y_offset already used on the source offset so we shifted the referential.



# length of vectors d1=(x1->source) d2=(x2->source)
d1 = abs( (x1,y1,0) , (s_x,s_y,z_offset) )
d2 = abs( (x2,y2,0) , (s_x,s_y,z_offset) )


a = d1/d2
alpha = a - 1/a
delta = (d2-d1)/setup.f('c')

print (alpha,delta)
