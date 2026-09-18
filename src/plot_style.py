#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 22:03:00 2025

@author: drminor
"""
import matplotlib as mpl

STYLE = {
    "figure.titlesize" : 40, #fig.suptitle()의 폰트 크기
    "figure.titleweight": "bold",
    "axes.titlesize"   : 40, # 각 subplot의 폰트 크기
    "axes.titleweight" : "bold",
    "font.size"        : 20,
    "font.weight"      : "bold",
    "axes.labelweight" : "bold",
    "axes.grid"        : True,
    "lines.linewidth"  : 3,
    "lines.markersize" : 9, # default 10
    "axes.labelsize"   : 30,
    "xtick.labelsize"  : 22,
    "ytick.labelsize"  : 22,
    "legend.fontsize"  : 25
}

def set_plot_style():
    mpl.rcParams.update(STYLE)    