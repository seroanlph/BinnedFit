#!/usr/bin/python
import iminuit as imin
import numpy as np
import inspect
import sys
from os import system
from scipy.stats import kstest


class UnbinnedLLH():
    def __init__(self, model, x, start):
        if np.ma.isMaskedArray(x):
            self.x = x.compressed()
        elif type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        self.model = model
        if type(start) == dict:
            self.par0 = start
        else:
            self.par0 = dict(
                zip(inspect.getfullargspec(self.model).args[1:], start))

    def likelihood(self, *params):
        return -2 * np.log(self.model(self.x, *params)).sum()

    def run(self, verbose=True):
        """
        Run the fit, returns the minimized minimizer which stores the minimum and
        the values.
        """
        self.m = imin.Minuit(self.likelihood, **self.par0,
                             name=self.par0.keys())
        self.m.print_level = int(verbose)*2
        self.m.migrad()
        self.m.hesse()
        return self.m

    def goodness_of_fit(self):
        ks_ts, p = kstest(self.x, self.model, args=self.m.values)
        return(p)


class LLHFit(UnbinnedLLH):
    """
    General Template for binned likelihood fitting.
    Use it only if you have to implement your own edge case -
    Otherwise GeneralLLHFit is probably what your looking for.
    """

    def __init__(self, model, x, n_meas, start):
        """
        Read in Model and Data to fit.

        -----------
        parameters:

        model: function
        Model that predicts n_meas. Should take the data as an numpy array
        as first argument and arbitrary *params after that.
        x: numpy array
        The data used to create a prediction which is then compared to n_meas
        n_meas: numpy array with dtype int
        number of entries in the histogram, which is used to fit.
        start: array or dict:
        Initial guess on the parameters to fit. If an array is provided,
        the names of the parameters are tried to be inferred from the model.
        Otherwise a dictionary can be used to provide names for the parameter.

        """
        if np.ma.isMaskedArray(x) and np.ma.isMaskedArray(n_meas):
            or_mask = np.ma.mask_or(n_meas.mask, x.mask)
            self.n_meas = np.ma.masked_where(or_mask, n_meas)
            UnbinnedLLH.__init__(
                self, model, np.ma.masked_where(or_mask, x), start)
        elif np.ma.isMaskedArray(n_meas):
            self.n_meas = n_meas.compressed()
            UnbinnedLLH.__init__(
                self, model, np.ma.masked_where(n_meas.mask, x), start)
        elif np.ma.isMaskedArray(x):
            self.n_meas = np.ma.masked_where(x.mask, n_meas).compressed()
            UnbinnedLLH.__init__(self, model, x, start)
        else:
            self.n_meas = n_meas
            UnbinnedLLH.__init__(self, model, x, start)

    def likelihood(self, *params):
        """
        Dummy function for the likelihood calculation.
        Implement your own, if there is no fitting implementation
        """
        pass

    def goodness_of_fit(self):
        try:
            p_meas = self.p_meas
            self.asimov = self.model(self.x, *self.m.values)
            self.p_meas = self.asimov
            safe = self.llhvals
            print(self.llhvals)
            asimov_likelihood = self.likelihood(*self.m.values)
            print(self.llhvals)
            print(safe-self.llhvals)
            self.p_meas = p_meas
        except NameError:
            print(self.m.values)
            n_meas = self.n_meas
            self.asimov = self.model(self.x, *self.m.values)
            self.n_meas = self.asimov
            asimov_likelihood = self.likelihood(*self.m.values)
            self.n_meas = n_meas

        return (asimov_likelihood-self.m.fval)


class ChiSquare(LLHFit):
    def __init__(self, model, x, y, yerr, start):
        LLHFit.__init__(self, model, x, y,  start)
        try:
            self.yerr = yerr[[True for y in y]]
        except TypeError:
            self.yerr = np.array([yerr for item in y])
        except IndexError:
            raise IndexError('len of y and yerr have to be the same!')

    def likelihood(self, *params):
        return np.sum((self.n_meas - self.model(self.x, *params)**2 / self.yerr**2))


class NeymanChi2Fit(LLHFit):
    def likelihood(self, *params):
        return np.sum((self.n_meas - self.model(self.x, *params)**2/self.n_meas))


class PearsonChi2Fit(LLHFit):
    def likelihood(self, *params):
        n_model = self.model(self.x, *params)
        return np.sum((self.n_meas - n_model)**2/n_model)


class PoissonFit(LLHFit):
    def likelihood(self, *params):
        n_pred = self.model(self.x, *params)
        return -2*self.__likelihood(n_pred).sum()

    def __likelihood(self, n_pred):
        return self.n_meas*np.log(n_pred) - n_pred


class BinomialFit(LLHFit):
    def __init__(self, model, x, n_meas, N, start):
        LLHFit.__init__(self, model, x, n_meas, start)
        self.N = N
        self.p_meas = self.n_meas/N

    def likelihood(self, *params):
        p = self.model(self.x, *params)
        likelihood = self.__likelihood(p)
        return -2 * self.N * likelihood.sum()

    def goodness_of_fit(self):
        L = self.m.fval
        L_exp = -2*self.N * self.__likelihood(self.p_meas).sum()
        return(L-L_exp)

    def __likelihood(self, p):
        q = 1-p
        likelihood = np.zeros_like(p)
        likelihood += np.where(
            (self.p_meas != 0) & (self.p_meas != 1),
            np.log(q) + self.p_meas * np.log(p) - self.p_meas * np.log(q),
            np.zeros_like(p))
        likelihood += np.where(self.p_meas == 0, np.log(q),
                               np.zeros_like(q))
        likelihood += np.where(self.p_meas == 1, np.log(p), np.zeros_like(p))
        return(likelihood)


class GeneralLLHFit(LLHFit):
    def __init__(self, model, x, n_meas, start, distribution, *args):
        self.distribution = lambda measurement, model: distribution(
            measurement, model, *args)
        LLHFit.__init__(self, model, x, n_meas, start)

    def likelihood(self, *params):
        n_model = self.model(self.x, *params)
        return -2*__likelihood(n_model).sum()

    def __likelihood(self, n_model):
        return np.log(self.distribution(self.n_meas, n_model))

    def goodness_of_fit(self):
        return 2*(self.__likelihood(self.n_meas).sum()-self.m.fval)


def install():
    paths = sys.path
    for path in paths:
        if "site-packages" in path:
            destination = path
            break

    system(f'cp ./fit.py {destination}')


if __name__ == "__main__":
    install()
    print("setup successfull")
