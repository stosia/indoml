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


class Correlation(Session):
    def __init__(self):
        self.groups = []
        self.alpha = 0.05

    def load_from_dict(self, d):
        """Load this instance from a dictionary.
        """
        self.alpha = d.get('alpha', 0.05)
        groups = d['groups']
        for g in groups:
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

    @classmethod
    def spell_conclusion_by_p(cls, p, alpha):
        """Spell conclusion on whether we reject or fail to reject the null hypothesis
        based on the pvalue.
        """
        if p <= alpha:
            conclusion = "reject the null -> p(%.3f) < %.3f" % (p, alpha)
        else:
            conclusion = "accept the null -> p(%.3f) > %.3f" % (p, alpha)
        return conclusion

    @classmethod
    def spell_conclusion_by_ci(cls, ci):
        """Spell conclusion on whether we reject or fail to reject the null hypothesis
        based on the confidence interval.
        """
        if ci[0] / ci[1] > 0:
            conclusion = "reject the null: CI (%.3f - %.3f) doesn't cross zero" % (ci[0], ci[1])
        else:
            conclusion = "accept the null: CI (%.3f - %.3f) crosses zero" % (ci[0], ci[1])
        return conclusion

    def print_correlation(self, grp0, grp1):
        print("%s --> %s correlation" % (grp0.title, grp1.title))
        print('-' * 70)

        r, _ = StatTool.pearson_r(grp0.members, grp1.members)
        """r, also called Pearson's r, is correlation coefficient, to quantify relationship.
        r measures the correlatoin for the sample
             cov(x,y)
        r = ---------
             Sx * Sy
        """

        r_squared = r ** 2
        """"r squared (r^2):
           r^2 = % of variation in Y explained by variation in x
           r^2 = coefficient of determination
        """

        df = grp0.n - 2
        """"Degree of freedom. We substract one from each sample"""

        # Convert r to t
        t = (r * math.sqrt(df)) / math.sqrt(1 - r ** 2)

        # Calculate the probability for t
        p = pval2 = StatTool.probability_for_t(t, StatTool.TWO_TAILED_TEST, df)

        ci = StatTool.pearson_r_confidence_interval(r, self.alpha, grp0.n)
        """
        if ρ (rho) is true correlation for population, CI is the confidence interval
        for ρ, meaning the range of likely values for the population correlation 
        coefficient ρ.
        """

        conclusion1 = self.spell_conclusion_by_ci(ci)
        conclusion2 = self.spell_conclusion_by_p(p, self.alpha)

        # We can also probably accept or reject the null hypothesis by looking at the CI
        # If the CI's range does NOT cross 0, it means there IS correlation. True??

        print("DF                       : % d" % df)
        print("Pearson r                : % .3f" % r)
        print("r^2 (coef of determ.)    : % .3f (%.2f%%)" % (r_squared, r_squared * 100.0))
        print("Confidence interval      : % .3f - %.3f" % ci)
        print("t-statistic              : % .3f" % t)
        print("P-value                  : % .5f" % p)
        print("Conclusion               : - %s" % conclusion1)
        print("                           - %s" % conclusion2)
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

    @classmethod
    def simple_decision(cls, r, n, alphas):
        """Decide whether to accept or reject the null hypothesis.
        """
        df = n - 2
        t = (r * math.sqrt(df)) / math.sqrt(1 - r ** 2)
        p = StatTool.probability_for_t(t, StatTool.TWO_TAILED_TEST, df)

        for alpha in alphas:
            ci = StatTool.pearson_r_confidence_interval(r, alpha, n)
            conclusion1 = cls.spell_conclusion_by_ci(ci)
            conclusion2 = cls.spell_conclusion_by_p(p, alpha)
            print("For alpha=%.4f:" % alpha)
            print(" - %s" % conclusion1)
            print(" - %s" % conclusion2)
            print("")
