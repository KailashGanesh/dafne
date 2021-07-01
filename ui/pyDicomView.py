#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import pydicom.filereader
from PyQt5.QtWidgets import QInputDialog

try:
    import pydicom as dicom
except:
    import dicom
import numpy as np
import os
import sys
try:
    from utils.dicomUtils.misc import create_affine
except:
    from dicomUtils.misc import create_affine
import traceback

try:
    from utils.dicomUtils.multiframe import load_multi_dicom, is_enhanced_dicom, is_multi_dicom, convert_to_slices
except:
    from dicomUtils.multiframe import load_multi_dicom, is_enhanced_dicom, is_multi_dicom, convert_to_slices

import dosma
from dosma.core.io.dicom_io import to_RAS_affine

DEFAULT_INTERPOLATION = 'spline36'
#DEFAULT_INTERPOLATION = 'none' # DEBUG
INVERT_SCROLL = True
DO_DEBUG = True


class ImListProxy:

    def __init__(self, medical_volume):
        self.medical_volume = medical_volume

    def __getitem__(self, item):
        return self.medical_volume.volume[:, :, item].astype(np.float32)

    def __len__(self):
        return self.medical_volume.shape[2]


class ImageShow:
    
    contrastWindow = None
    channelBalance = np.array([1.0, 1.0, 1.0])
    
    def __init__(self, im = None, axes = None, window=None, cmap=None):
        ImageShow.contrastWindow = window
        
        self.imPlot = None
        
        if axes is None:
            #initialize the figure
            self.fig = plt.figure()
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = axes
            self.fig = self.axes.get_figure()
            
        self.axes.axis('off')
        self.connect_ids = []
        self.connectSignals()

        # stack of images
        self.imList = []
        self.dicomHeaderList = None
        self.curImage = None
        self.cmap = cmap
        self.isImageRGB = False
        self.basepath = ''
        self.basename = ''

        self.resolution = [1, 1, 1]
        self.resolution_valid = False
        self.affine = None
        self.medical_volume = None

        self.interpolation = DEFAULT_INTERPOLATION

        # These methods can be defined in a subclass and called when some event occurs
        #self.leftPressCB = None
        #self.leftMoveCB = None     
        #self.leftReleaseCB = None
        #self.refreshCB = None

        self.oldMouseXY = (0, 0)
        self.startXY = None

        if im is not None:
            if type(im) is np.ndarray:
                print("Display array")
                self.loadNumpyArray(im)
            elif type(im) is str:
                if os.path.isdir(im):
                    self.loadDirectory(im)
                else:
                    try:
                        im = self.loadDicomFile(im)
                    except:
                        pass
                    self.imList.append(im)
            self.curImage = 0
            self.displayImage(0)

    def connectSignals(self):
        if self.connect_ids: return
        self.connect_ids = []
        self.connect_ids.append(self.fig.canvas.mpl_connect('button_press_event', self.btnPressCB))
        self.connect_ids.append(self.fig.canvas.mpl_connect('button_release_event', self.btnReleaseCB))
        self.connect_ids.append(self.fig.canvas.mpl_connect('motion_notify_event', self.mouseMoveCB))
        self.connect_ids.append(self.fig.canvas.mpl_connect('scroll_event', self.mouseScrollCB))
        self.connect_ids.append(self.fig.canvas.mpl_connect('key_press_event', self.keyPressCB))
        self.connect_ids.append(self.fig.canvas.mpl_connect('key_release_event', self.keyReleaseCB))

    def disconnectSignals(self):
        for cid in self.connect_ids:
            self.fig.canvas.mpl_disconnect(cid)
        self.connect_ids = []


    def displayImageRGB(self):
        #print "Displaying image"
        dispImage = np.copy(self.image)
        dispImage[:,:,0] *= ImageShow.channelBalance[0]
        dispImage[:,:,1] *= ImageShow.channelBalance[1]
        dispImage[:,:,2] *= ImageShow.channelBalance[2]
        dispImage = (dispImage - ImageShow.contrastWindow[0])/(ImageShow.contrastWindow[1] - ImageShow.contrastWindow[0])
        dispImage[dispImage < 0] = 0
        dispImage[dispImage > 1] = 1
        if self.imPlot is None:
            self.imPlot = self.axes.imshow(dispImage, interpolation = self.interpolation, aspect=self.resolution[1]/self.resolution[0])
        else:
            self.imPlot.set_data(dispImage)
        self.redraw()
        
    
    def displayImage(self, im, cmap = None, redraw = True):
        if cmap is None:
          if self.cmap is None:
            cmap = 'gray'
          else:
            cmap = self.cmap

        try:
            oldSize = self.image.shape
        except:
            oldSize = (-1,-1)

        # im can be an integer index in the imList
        if isinstance(im, int):
            if im >= 0 and im < len(self.imList):
                self.curImage = im
                self.image = self.imList[im]
        else: # otherwise let's assume it is pixel data
            self.image = im

        title = ''
        try:
            title = self.instructions + '\n'
        except:
            pass
        
        title += 'Image: %d' % self.curImage
            
        try:
            self.axes.set_title(title)
        except:
            pass
            
            
        # calculate the contrast if it was not already defined
        if ImageShow.contrastWindow is None:
            ImageShow.contrastWindow = self.calcContrast(self.image)
        
        
        if self.image.ndim == 3:
            self.isImageRGB = True
            self.displayImageRGB()
            return
        else:
            self.isImageRGB = False
        
        self.setCmap(cmap, False)

        if self.imPlot:
            if oldSize != self.image.shape: # if the image shape is different, force a new imPlot to be created
                try:
                    self.imPlot.remove()
                except:
                    pass
                self.imPlot = None

        # Create the image plot if there is none; otherwise update the data in the existing frame (faster)
        if self.imPlot is None:
            self.imPlot = self.axes.imshow(self.image, interpolation = self.interpolation,
                                           vmin=ImageShow.contrastWindow[0],
                                           vmax=ImageShow.contrastWindow[1],
                                           cmap=self.cmap, zorder = -1, aspect=self.resolution[1]/self.resolution[0])
        else:
            self.imPlot.set_data(self.image)

        if redraw:
            self.redraw()
            
        
    def redraw(self):
        try:
            self.refreshCB()
        except Exception as err:
            if DO_DEBUG: traceback.print_exc()
        self.fig.canvas.draw()
        
    def mouseScrollCB(self, event):
        self.oldMouseXY = (event.x, event.y) # this will suppress the mouse move event
        step = -event.step if INVERT_SCROLL else event.step
        if self.curImage is None or (step > 0 and self.curImage == 0) or (step < 0 and self.curImage > len(self.imList)-1):
            return
            
        if event.inaxes != self.axes:
            return
            
        self.curImage = self.curImage - step;
        if self.curImage < 0:
            self.curImage = 0
        if self.curImage > len(self.imList)-1:
            self.curImage = len(self.imList) - 1
        self.displayImage(self.imList[int(self.curImage)], self.cmap, redraw=False)
        self.redraw() # already called in displayImage
        try:
            self.fig.canvas.setFocus()
        except:
            pass

    def keyReleaseCB(self, event):
        pass

    def keyPressCB(self, event):
        event.step = 0
        if event.key == 'right' or event.key == 'down':
            event.step = 1 if INVERT_SCROLL else -1
        elif event.key == 'left' or event.key == 'up':
            event.step = -1 if INVERT_SCROLL else 1
        self.mouseScrollCB(event)

    def isCursorNormal(self):
        try:
            isCursorNormal = ( self.fig.canvas.cursor().shape() == 0 ) # if backend is qt, it gets the shape of the
                # cursor. 0 is the arrow, which means we are not zooming or panning.
        except:
            isCursorNormal = True
        return isCursorNormal

    def btnPressCB(self, event):
        if not self.isCursorNormal():
            #print("Zooming or panning. Not processing clicks")
            return
        if event.button == 1:
            try:
                self.leftPressCB(event)
            except Exception as err:
                if DO_DEBUG: traceback.print_exc()
        if event.button == 3:
            if event.dblclick:
                self.resetContrast()
            else:
                self.startContrast = ImageShow.contrastWindow
                self.startBalance = np.copy(ImageShow.channelBalance)
                self.startXY = (event.x, event.y)
                self.rightPressCB(event)
    
    def rightPressCB(self, event):
        pass
            
    def btnReleaseCB(self, event):
        if event.button == 1:
            try:
                self.leftReleaseCB(event)
            except Exception as err:
                if DO_DEBUG: traceback.print_exc()
        if event.button == 3:
            self.imPlot.set_interpolation(self.interpolation)
            self.startXY = None # 
            self.redraw()
            self.rightReleaseCB(event)
        
    def rightReleaseCB(self, event):
        pass
        
    def resetContrast(self):
        ImageShow.contrastWindow = self.calcContrast(self.image)
        if not self.isImageRGB:
            self.imPlot.set_clim(ImageShow.contrastWindow)
            self.redraw()
        else:
            # if image is RGB, we need to redraw it completely. Maybe it will be too slow?
            self.displayImageRGB()
        
    # callback for mouse move
    def mouseMoveCB(self, event):
        xy = (event.x, event.y)
        if xy == self.oldMouseXY: return # reject mouse move events when the mouse doesn't move
        self.oldMouseXY = xy
        if event.button == 1:
            try:
                self.leftMoveCB(event)
            except Exception as err:
                if DO_DEBUG: traceback.print_exc()
        if event.button != 3 or self.startXY is None:
            return
        
        self.imPlot.set_interpolation('none')
        # convert contrast limits into window center and size
        contrastCenter = (self.startContrast[0] + self.startContrast[1])/2
        contrastExtent = (self.startContrast[1] - self.startContrast[0])/2
        
        # calculate displacemente of the mouse
        xDisplacement = event.x - self.startXY[0]
        yDisplacement = event.y - self.startXY[1]

        if event.key == 'control':
            ImageShow.channelBalance[0] = self.startBalance[0] - (float(xDisplacement)/100 + float(yDisplacement)/100)
            ImageShow.channelBalance[1] = self.startBalance[1] + float(xDisplacement)/100
            ImageShow.channelBalance[2] = self.startBalance[2] + float(yDisplacement)/100
            ImageShow.channelBalance[ImageShow.channelBalance < 0] = 0
            ImageShow.channelBalance[ImageShow.channelBalance > 1] = 1
            #print ImageShow.channelBalance
        else:
            # recalculate the window
            # the displacements have negative sign because it feels more natural
            contrastCenter = contrastCenter - yDisplacement
            contrastExtent = contrastExtent - xDisplacement
            if contrastExtent < 1:
                contrastExtent = 1
            
            # set the contrast window
            ImageShow.contrastWindow = (contrastCenter - contrastExtent, contrastCenter + contrastExtent)

        if not self.isImageRGB:
            self.imPlot.set_clim(ImageShow.contrastWindow)
            self.redraw()
        else:
            # if image is RGB, we need to redraw it completely. Maybe it will be too slow?
            self.displayImageRGB()
    
    def calcContrast(self, im):
        maxVal = np.percentile(im, 90)
        if maxVal <= 1: maxVal = np.max(im.flat)
        return (0, maxVal) # stretch contrast to remove outliers
    
    def setCmap(self, cmap, redraw=True):
        self.cmap = cmap;
        if self.imPlot is not None:
            self.imPlot.set_cmap(cmap)

        if redraw:
            self.redraw()


    def getDicomResolution(self, ds):
        resolution_valid = False
        try:
            slThickness = ds.SpacingBetweenSlices
        except:
            try:
                slThickness = ds.SliceThickness
            except:
                slThickness = 1

        try:
            pixelSpacing = ds.PixelSpacing
            resolution_valid = True
        except:
            pixelSpacing = [1, 1]

        resolution = [float(pixelSpacing[0]), float(pixelSpacing[1]), float(slThickness)]
        return resolution, resolution_valid

    def loadDicomFile(self, fname):
        print(fname)
        ds = dicom.read_file(fname)
        # rescale dynamic range to 0-4095
        try:
            pixelData = ds.pixel_array.astype(np.float32)
        except:
            ds.decompress()
            pixelData = ds.pixel_array.astype(np.float32)

        ds.PixelData = ""
        self.dicomHeaderList.append(ds)

        self.resolution, self.resolution_valid = self.getDicomResolution(ds)
        return pixelData
        
    # append one image to the internal list
    def appendImage(self, im):
        if type(im) is str:
            try:
                im = self.loadDicomFile(im)
            except:
                print("Error loading file:", im)
                return
        self.imList.append(im)
        
    def loadNumpyArray(self, data):
        if np.max(data.flat) <= 1: data *= 1000
        
        #print data.shape
        for sl in range(data.shape[2]):
            self.appendImage(data[:,:,sl])
            

    def load_dosma_volume(self, medical_volume):
        if np.max(medical_volume.volume) < 10:
            medical_volume *= 100
        while np.max(medical_volume.volume) > 10000:
            print(np.max(medical_volume.volume))
            medical_volume.volume /= 10
        self.medical_volume = medical_volume
        self.resolution = np.array(self.medical_volume.pixel_spacing)
        self.resolution_valid = True
        self.affine = self.medical_volume.affine
        self.imList = ImListProxy(self.medical_volume)
        if medical_volume.headers() is not None:
            self.dicomHeaderList = list(medical_volume.headers().squeeze())
        else:
            self.dicomHeaderList = None

    # load a whole directory of dicom files
    def loadDirectory(self, path):
        self.imList = []
        self.dicomHeaderList = None
        self.medical_volume = None
        self.affine = None
        self.resolution_valid = False
        self.resolution = [1,1,1]
        dicom_ext = ['.dcm', '.ima']
        nii_ext = ['.nii', '.gz']
        npy_ext = ['.npy']
        path = os.path.abspath(path)
        _, ext = os.path.splitext(path)

        basename = os.path.basename(path)

        self.basename = basename
        self.fig.canvas.manager.set_window_title(basename)

        if ext.lower() in npy_ext:
            data = np.load(path).astype(np.float32)
            self.loadNumpyArray(data)
            self.basepath = os.path.dirname(path)
        elif ext.lower() in nii_ext:
            niiReader = dosma.NiftiReader()
            medical_volume = niiReader.load(path)
            desired_orientation, accept = QInputDialog.getItem(self.fig.canvas,
                                                       "Nifti loader",
                                                       "Select orientation",
                                                       ['Axial', 'Sagittal', 'Coronal'],
                                                       editable=False)
            if not accept: return
            if desired_orientation == 'Axial':
                nifti_orient = ('AP', 'RL', 'IS')
            elif desired_orientation == 'Sagittal':
                nifti_orient = ('SI', 'PA', 'RL')
            else:
                nifti_orient = ('SI', 'RL', 'AP')

            self.load_dosma_volume(medical_volume.reformat(nifti_orient))
            self.basepath = os.path.dirname(path)
        else: # assume it's dicom
            load_dicom_dir = False
            if os.path.isfile(path):
                basepath = os.path.dirname(path)
                dataset = pydicom.read_file(path)
                if is_enhanced_dicom(dataset):
                    if is_multi_dicom(dataset):
                        multi_dicom_dataset = load_multi_dicom(dataset)
                        # this is a multi dicom dataset
                        # let the user choose which dataset to load
                        dataset_key, accept = QInputDialog.getItem(self.fig.canvas,
                                                                   "Multi dicom",
                                                                   "Choose dataset to load",
                                                                   list(multi_dicom_dataset.keys()),
                                                                   editable=False)
                        if not accept: return
                        header_list = multi_dicom_dataset[dataset_key][1]
                        data = multi_dicom_dataset[dataset_key][0].astype(np.float32)
                    else:
                        # enhanced dicom but not with multiple contrasts
                        data, header_list = convert_to_slices(dataset)

                    affine = to_RAS_affine(header_list)
                    medical_volume = dosma.core.MedicalVolume(data, affine, header_list)

                    self.fig.canvas.manager.set_window_title(os.path.basename(path))
                    load_dicom_dir = False
                else:
                    self.fig.canvas.manager.set_window_title(os.path.basename(basepath))
                    load_dicom_dir = True

            elif os.path.isdir(path):
                basepath = path
                self.fig.canvas.manager.set_window_title(basepath)
                load_dicom_dir = True

            if load_dicom_dir:
                dr = dosma.DicomReader(num_workers=1)
                #print("Loading", basepath)
                medical_volume = dr.load(basepath)[0]
                #print("Ok")
                self.basename = ''
                self.basepath = basepath

            self.load_dosma_volume(medical_volume)

        if len(self.imList) > 0:
            try:
                self.imPlot.remove()
            except:
                pass
            self.imPlot = None
            self.curImage = 0
            self.displayImage(int(0))
            self.axes.set_xlim(-0.5, self.image.shape[1] - 0.5)
            self.axes.set_ylim(self.image.shape[0] - 0.5, -0.5)


# when called as a script, load all the images in the directory
if __name__ == "__main__":
    # test file
    #imFig = imageShow("image0001.dcm")
    #imFig.appendImage("image0002.dcm")
    imFig = ImageShow()
    imFig.loadDirectory(sys.argv[1])
    #imFig.loadDirectory('image0001.dcm')
    plt.show()
