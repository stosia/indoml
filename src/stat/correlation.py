#!/usr/bin/env python
# -*- coding: utf-8-unix -*-
from __future__ import absolute_import, print_function, division, unicode_literals

import csv
import math

from sample import Sample
from session import Session
from tools import StatTool, read_input


__author__ = "Benny Prijono <benny@stosia.com>"
__copyright__ = "Copyright (C)2017 PT. Stosia Teknologi Investasi"
__license__ = "GNU Affero AGPL (AGPL) version 3.0 or later"


class Correlation:
    def __init__(self):
        self.groups = []
        self.alpha = 0.05

    def load_from_dict(self, d):
        """Load this instance from a dictionary.
        """
        self.alpha = d.get('alpha', 0.05)
        samples = d['samples']
        for g in samples:
            samp = Sample()
            samp.load_from_dict(g)
            self.groups.append(samp)

    def load_from_csv(self, filename, csv_indices=None):
        """Load the groups from the CSV file. Each column is treated as one group.
        """
        with open(filename) as f:
            r = csv.reader(f, delimiter=str(","))
            head = r.next()
        if not csv_indices:
            csv_indices = range(0, len(head))
        for i in csv_indices:
            col = head[i]
            samp = Sample(title=col, is_population=False)
            samp.load_from_csv(filename, i)
            self.groups.append(samp)

    def input_wizard(self, csv_filename=None, csv_indices=None):
        """Read input from the console interactively.
        """
        if csv_filename:
            self.load_from_csv(csv_filename, csv_indices=csv_indices)
        else:
            while True:
                samp = Sample(title='samp%d' % len(self.groups), is_population=False)
                samp.input_wizard(require_n=True)
                if not samp.n:
                    break
                self.groups.append(samp)
                more = read_input('Input more sample', default='y', choices='yn')
                if more != 'y':
                    break
        self.alpha = float(read_input("Alpha", default='%.03f' % self.alpha))

    def save_to_dict(self):
        """Save this instance to a dictionary to be serialized.
        """
        return self.__dict__

    def print_correlation(self, grp0, grp1):
        # x: predictor, explanatory, independent variable
        # y: outcome, response, dependent variable
        #
        # correlation coefficient (r): to quantify relationship. Also called Pearson's r.
        #      cov(x,y)
        # r = ---------
        #      Sx * Sy
        #
        # r^2 = % of variation in Y explained by variation in x
        # r^2 = coefficient of determination
        #
        # r = correlation for the sample
        # ρ (rho) = true correlation for population

        print("%s --> %s correlation" % (grp0.title, grp1.title))
        print('-' * 70)
        r, pval1 = StatTool.pearson_r(grp0.members, grp1.members)
        df = grp0.n - 2
        t = (r * math.sqrt(df)) / math.sqrt(1 - r ** 2)
        p = pval2 = StatTool.probability_for_t(t, StatTool.TWO_TAILED_TEST, df)

        ci = StatTool.pearson_r_confidence_interval(r, self.alpha, grp0.n)

        if pval2 <= self.alpha:
            conclusion = "reject the null -> p(%.3f) < %.3f" % (p, self.alpha)
        else:
            conclusion = "accept the null -> p(%.3f) > %.3f" % (p, self.alpha)

        # We can also probably accept or reject the null hypothesis by looking at the CI
        # If the CI's range does NOT cross 0, it means there IS correlation. True??

        print("DF                       : % d" % df)
        print("Pearson r                : % .3f" % r)
        print("r^2 (coef of determ.)    : % .3f (%.2f%%)" % (r ** 2, r ** 2 * 100.0))
        print("Confidence interval      : % .3f - %.3f" % ci)
        print("t-statistic              : % .3f" % t)
        print("P-value                  : % .5f" % p)
        print("Conclusion               :  %s" % conclusion)
        print("")

    def print_report(self):
        """Print report to stdout.
        """
        print("Correlation REPORT:")
        print('=' * 70)
        for i, samp in enumerate(self.groups):
            print("Sample-%d: %s" % (i, samp.title))
            print("-" * 70)
            print(str(samp))

        print("Parameters:")
        print('-' * 70)
        print("Alpha                    : % .3f" % self.alpha)
        print("")

        pairs = []
        for i in range(len(self.groups)):
            for j in range(i + 1, len(self.groups)):
                self.print_correlation(self.groups[i], self.groups[j])
