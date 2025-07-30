import Hamiltonians.ham as ham
from Code.Spectrum_Generation.boltzmann import getdenom, boltzmann
import numpy as np
import pandas as pd
from Code.Spectrum_Plotting.fancyplotter import plotter as fp
from Code.Spectrum_Plotting.standardplotter import plotter as spl
from Code.general import timing



class Model():
    def __init__(self, d1, d2, F, Fprime, Qx, Qz, consts, uconsts, V3, V6, V3p, V3m, chop, mu, jmin, jmax, T, lims, width, shift,
                 stats=[1, 1, 1, 1]):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)
        self.__hams()

    def __hams(self):
        self.DMSops = ham.ham(self.d1, self.d2, self.jmin, self.jmax, self.F, self.Fprime, self.Qx, self.Qz, self.consts,self.uconsts, self.mu, self.V3, self.V6, self.V3p, self.V3m, self.chop)

    def newcalcspectrum(self, save=False, name='spectrum', l=None):
        with timing("Spectrum Calculation") as t:
            uidxs, lidxs = self.DMSops.dipole.nonzero()
            couplings = self.DMSops.dipole.data
            denom = getdenom(self.DMSops.lsubham.energies, self.T)
            ulabels = [self.DMSops.usubham.diagbasis[i] for i in uidxs]
            llabels = [self.DMSops.lsubham.diagbasis[i] for i in lidxs]
            freqs = np.array([self.DMSops.usubham.energies[i] - self.DMSops.lsubham.energies[j] for i, j in zip(uidxs, lidxs)])
            boltzs = np.array([boltzmann(self.DMSops.lsubham.energies[j], self.T, denom) for j in lidxs])
            intensities = np.abs(couplings) ** 2 * boltzs
            spectrum_data = []
            for freq, inten, lower, upper in zip(freqs, intensities, llabels, ulabels):
                spectrum_data.append({
                    "frequency": freq,
                    "intensity": inten,
                    "lower_state": tuple(lower),
                    "upper_state": tuple(upper)
                })
            # Create DataFrame from the list of dictionaries
            # Convert accumulated list to a DataFrame at the end
            self.spectrum = pd.DataFrame(spectrum_data)
            print(len(self.spectrum), ' transitions evaluated.')
            self.spectrum["frequency"] = self.spectrum["frequency"].astype("float")
            self.spectrum["intensity"] = self.spectrum["intensity"].astype("float")
            self.spectrum = self.spectrum.sort_values(by="frequency", ascending=True)
            self.spectrum = self.spectrum[(self.spectrum['intensity'] > self.spectrum['intensity'].max() * 10**(-5))]
            self.spectrum = self.spectrum[
                (self.spectrum['frequency'] >= self.lims[0]) & (self.spectrum['frequency'] <= self.lims[1])]
            self.spectrum = self.spectrum.reset_index(drop=True)

            if l is not None:
                mask = self.spectrum["lower_state"].apply(lambda x: x[1] == l) & \
                       self.spectrum["upper_state"].apply(lambda x: x[1] == l)
                self.spectrum = self.spectrum[mask].reset_index(drop=True)

            # Save the spectrum to CSV
            if save:
                self.spectrum.to_csv(name + '.csv', index=False)
            return self.spectrum


    def fancyplot(self, other=None):
        fp(self.spectrum, self.lims, self.width, self.shift,'Dimethyl Sulfide Spectrum', other)

    def plot(self, other = None, title = 'Dimethyl Sulfide Spectrum (with Torsions)', show = True):
        spec, x = spl(self.spectrum, self.lims, self.width, self.shift,title , other, show = show)
        return spec, x

    def symanalysis(self, filename=None):
        self.DMSops.mixinganalysis(filename = filename)


