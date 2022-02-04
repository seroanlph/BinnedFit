#!/usr/bin/python
import iminuit as imin
import numpy as np
import inspect
import sys
from os import system


class LLHFit():
    def __init__(self, model, x, n_meas, start):
        self.x = x
        self.model = model
        self.n_meas = n_meas

        if type(start) == dict:
            self.par0 = start
        else:
            self.par0 = dict(
                zip(inspect.getfullargspec(self.model).args[1:], start))

    def likelihood(self, *params):
        pass

    def run(self, verbose=True):
        m = imin.Minuit(self.likelihood, **self.par0, name=self.par0.keys())
        m.migrad()
        m.hesse()
        return m


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
        return -2*np.sum(self.n_meas*np.log(n_pred) - n_pred)


class BinomialFit(LLHFit):
    def __init__(self, model, x, n_meas, N, start):
        LLHFit.__init__(self, model, x, n_meas, start)
        self.N = N
        self.p_meas = self.n_meas/N

    def likelihood(self, *params):
        p = self.model(self.x, *params)
        q = 1 - p
        likelihood = np.zeros_like(p)
        likelihood += np.where(
            (self.p_meas != 0) & (self.p_meas != 1),
            np.log(q) + self.p_meas * np.log(p) - self.p_meas * np.log(q),
            np.zeros_like(p))
        likelihood += np.where(self.p_meas == 0, np.log(q),
                               np.zeros_like(q))
        likelihood += np.where(self.p_meas == 1, np.log(p), np.zeros_like(p))
        return -2 * self.N * likelihood.sum()


def install():
    paths = sys.path
    for path in paths:
        if "site-packages" in path:
            destination = path
            break

    system('cp ./fit.py {destination}')


if __name__ == "__main__":
    install()
    print("setup successfull")
