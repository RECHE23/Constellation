# -*- coding: utf-8 -*-

from record import BlobsDS, MoonsDS, CirclesDS, IrisDS, Blobs5DDS, \
    MNIST784DS, BreastCancerDS, DS

datasets = [
    BlobsDS(scale=True),
    MoonsDS(scale=True),
    CirclesDS(scale=True),
    #BreastCancerDS(scale=True),
    #Blobs5DDS(scale=True),
    #IrisDS(scale=True, selected_class=1),
    #MNIST784DS(scale=True, selected_class='3')
           ]
