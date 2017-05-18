# -*- coding: iso-8859-15 -*-

import matplotlib as mpl
mpl.use('Agg')

import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform
import matplotlib
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms
import sys


class NorthPolarAxes(PolarAxes):
    '''
    A variant of PolarAxes where theta starts pointing north and goes
    clockwise.
    '''
    name = 'northpolar'

    class NorthPolarTransform(PolarAxes.PolarTransform):
        def transform(self, tr):
            xy   = np.zeros(tr.shape, np.float_)
            t    = tr[:, 0:1]
            r    = tr[:, 1:2]
            x    = xy[:, 0:1]
            y    = xy[:, 1:2]
            x[:] = r * np.sin(t)
            y[:] = r * np.cos(t)
            return xy

        transform_non_affine = transform

        def inverted(self):
            return NorthPolarAxes.InvertedNorthPolarTransform()

    class InvertedNorthPolarTransform(PolarAxes.InvertedPolarTransform):
        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:]
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(y, x)
            return np.concatenate((theta, r), 1)

        def inverted(self):
            return NorthPolarAxes.NorthPolarTransform()

    def _set_lim_and_transforms(self):
        PolarAxes._set_lim_and_transforms(self)
        self.transProjection = self.NorthPolarTransform()
        self.transData = (
            self.transScale + 
            self.transProjection + 
            (self.transProjectionAffine + self.transAxes))
        self._xaxis_transform = (
            self.transProjection +
            self.PolarAffine(IdentityTransform(), Bbox.unit()) +
            self.transAxes)
        self._xaxis_text1_transform = (
            self._theta_label1_position +
            self._xaxis_transform)
        self._yaxis_transform = (
            Affine2D().scale(np.pi * 2.0, 1.0) +
            self.transData)
        self._yaxis_text1_transform = (
            self._r_label1_position +
            Affine2D().scale(1.0 / 360.0, 1.0) +
            self._yaxis_transform)

register_projection(NorthPolarAxes)




""" **********************************************************************************
    Einlesen von KFÜ-Daten aus der RODOS-Datenbank;
    Abspeichern als graphische Darstellung der Windbedingungen der letzten 5 Stunden 
    ****************************************************************************** """

# Input Filename mit Daten zur Windgeschw. und Windrichtung
#filename='/tmp/kfue_wetter.txt'

filename=sys.argv[1]
print '\neinzulesende Datei:', filename
if len(sys.argv)!=3:
   raise ValueError('Es muessen genau 2 Parameter uebergeben werden!')



# Einlesen der meteorologischen KFÜ-Daten
# Windgeschw., Windrichtung, Stabilitaetsklasse, Niederschlag
data=np.loadtxt(filename, delimiter='|', dtype=str)

date0=data[:,0]
ns=np.array(data[:,1], dtype=float)
diffkat=np.asarray(data[:,2], dtype=str)
speed_lowlevel=np.asarray(data[:,4], dtype=float)
winddir_lowlevel=np.asarray(data[:,3], dtype=float)


# Umwandlung Datumsangabe in datetime-Objekt
date=np.zeros(len(date0), dtype=datetime.datetime)
for i in range(len(date0)):
    date[i]=datetime.datetime.strptime(date0[i], '%d.%m.%Y %H:%M ')


# aktuelles Datum
dateact=datetime.datetime.strftime(date[0], '%H:%M Uhr')

# Umrechnung von degree in rad (wird zum plotten benoetigt)
for i in range(len(winddir_lowlevel)):
    winddir_lowlevel[i]=math.radians((winddir_lowlevel[i]))

# Umdrehen, dass neueste Werte unten stehen
winddir_lowlevel=winddir_lowlevel[::-1]
speed_lowlevel=speed_lowlevel[::-1]
date=date[::-1]
ns=ns[::-1]
diffkat=diffkat[::-1]

# Intervall zwischen erster und letzter Messung in Minuten
dt = (date[-1]-date[0])
interval = int((dt.seconds)/60.)

"""------------------------------ PLOT WINDROSE --------------------------------"""
    
fig=plt.figure(1, figsize=(5*0.8, 6*0.8))

"""------------------- WINDROSE AUF AX1 (Hauptbild) ----------------------------"""

ax1=fig.add_axes([0.17,0.21,0.70,0.70], projection='northpolar' , axisbg='#FFFBE5')

# roter Freisetzungspunkt
ax1.plot(0.,0.,'ro', markersize=12, mec='red', mfc='#ff9892', mew=2, zorder=0)
thetaticks1 = np.array([0,45,90, 135,180,225,270, 315]) #range(0,360, 45)
ax1.set_thetagrids(thetaticks1, frac=1.15)



al=np.linspace(0.1,0.8,len(winddir_lowlevel)) # Transparenz: schwaecher je aelter die Windmessung

# Plotten von blauen Windpfeilen (Verlauf der letzten Werte)
for i in range(len(winddir_lowlevel)):
    headlen= 0.13*speed_lowlevel[i]
    speed_lowlevel[i]=speed_lowlevel[i]-headlen
    arr2 = plt.arrow(winddir_lowlevel[i],0,0,-speed_lowlevel[i], width = 0.05, alpha=al[i], head_length=headlen, 
                 edgecolor = 'black', facecolor = 'blue', lw = 1.0 )

# Plotten des lila Windpfeils (aktuellster Wert)
plt.arrow(winddir_lowlevel[-1],0,0,-speed_lowlevel[-1], width = 0.05, alpha=0.8, head_length=headlen, 
                 edgecolor = 'black', facecolor = 'magenta', lw = 1.0, label ='aktueller Wind')  # winkel, startpunkt, ? , Laenge minus 1?
                 

# Skalierung des Radius anhand des Maximalwertes 
ax1.set_rmax(np.amax(speed_lowlevel)+headlen+0.1*np.amax(speed_lowlevel))                 

ax1.grid(True)

# Ticklabels Windgeschw.
for q in ax1.get_yticklabels():
    q.set_fontsize(10) 

# Ticklabels Windrichtung
for q in ax1.get_xticklabels():
    q.set_fontsize(13)  

text=ax1.text(0.67,0.87, 'm/s', ha='center', va='center',fontsize=10)   
text.set_transform(fig.transFigure)

# Ueberschrift
text=ax1.text(0.5,0.95, u'Ausbreitungsrichtung\nbasierend auf vergangenen KFÜ-Daten', ha='center', va='center', fontsize=12, fontweight='bold')
text.set_transform(fig.transFigure)



"--------------------------- LEGENDE --------------------------------------"

# ax4 ist fuer die Box um die Legendenbestandteile 
ax4=fig.add_axes([0.0,0.0,1,1], frameon=False)
ax4.axes.get_yaxis().set_visible(False)
ax4.axes.get_xaxis().set_visible(False)

bb = mtransforms.Bbox([[0.01, 0.01], [0.99, 0.2]])

p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             boxstyle="square, pad=0",
                             fc='white',
                             ec='black' )

p_fancy.set_transform(fig.transFigure)
ax4.add_patch(p_fancy)


# ax5 ist fuer die Legende des Freisetzungspunkts plus Text
ax5=fig.add_axes([0.01, 0.01, 0.8, 0.8], frameon=False)
ax5.axes.get_yaxis().set_visible(False)
ax5.axes.get_xaxis().set_visible(False)
ax5.plot(0.16,0.18, 'ro', markersize=10, mec='red', mfc='#ff9892', mew=2)
ax5.set_xlim(0,1)
ax5.set_ylim(0,1)
ax5.text(0.16,0.07, 'Freisetzungs-\npunkt', ha='center', va='center', fontsize=10)

# ax2 ist fuer die Legende des lila Pfeils plus Text
ax2=fig.add_axes([0.06,-0.1,0.5,0.5], projection='northpolar')
ax2.arrow(math.radians(270),0,0,-8, width = 0.09, alpha=0.8, head_length=2, 
                 edgecolor = 'black', facecolor = 'magenta', lw = 1.)
text=ax2.text(0.42,0.07, 'aktueller Wind\n ('+str(dateact)+')', ha='center', va='center',fontsize=10)   
text.set_transform(fig.transFigure)   

ax2.set_rmax(11)       
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_frame_on(False)

# ax3 ist fuer die Legende des blauen Pfeils plus Text
ax3=fig.add_axes([0.41,-0.1,0.5,0.5], projection='northpolar')
ax3.arrow(math.radians(270),0,0,-8, width = 0.09 ,alpha=0.8, head_length=2, 
                 edgecolor = 'black', facecolor = 'blue', lw = 1.)
#text = ax3.text(0.81,0.07, '  Windverlauf der\nletzten 5 Stunden in\n1-h-Intervallen', ha='center', va='center',  fontsize=9)
text = ax3.text(0.79,0.07, u'Ausbreitungsrichtung in\nden letzten '+str(interval)+' Minuten\n(heller werdend)', ha='center', va='center', fontsize=9)
text.set_transform(fig.transFigure)         

ax3.set_rmax(11)       
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
ax3.set_frame_on(False)

# Abspeichern des Plots
plt.savefig(sys.argv[2])
print 'Name des gespeicherten Plots:', sys.argv[2]




# plt.show()

