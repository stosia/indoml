#!/usr/bin/env python
# -*- coding: utf-8-unix -*-
from __future__ import absolute_import, print_function, division, unicode_literals

import csv
import sys

from sample import Sample
from tools import read_input, StatTool


__author__ = "Benny Prijono <benny@stosia.com>"
__copyright__ = "Copyright (C)2017 PT. Stosia Teknologi Investasi"
__license__ = "GNU Affero AGPL (AGPL) version 3.0 or later"


class Anova:
    """ANOVA, or Analysis of Variance, is used to analyze two or more groups (samples)
    to see if any of it is significantly difference than the other. This test produces
    a value called F-ratio or F-statistic, which can be used to accept or reject our
    hypothesis by comparing it to a critical value.
    
    The F-statistic is ratio between between-group variability and within-group
    variability. Large F-statistic means between group variability is large, relative t 
    within group variability. Then we know that at least one pair of means is significantly
    different than the other. So we accept the alternate hypothesis.
    
    Small statistic means within group variability is large, relative to between group 
    variability, and none of the means are significantly different from each other. 
    So we accept the null hypothesis.
    """

    def __init__(self):
        self.groups = []
        self.alpha = 0.05

    def _fix_samples(self):
        """Perform pre-processing to the samples after they are input
        """
        pass

    def load_from_dict(self, d):
        """Load this instance from a dictionary.
        """
        self.alpha = d.get('alpha', 0.05)
        groups = d['groups']
        for g in groups:
            samp = Sample()
            samp.load_from_dict(g)
            self.groups.append(samp)

        self._fix_samples()

    def save_to_dict(self):
        """Save this instance to a dictionary to be serialized.
        """
        return self.__dict__

    def load_from_csv(self, filename, csv_start_col_idx=0):
        """Load the groups from the CSV file. Each column is treated as one group.
        """
        with open(filename) as f:
            r = csv.reader(f, delimiter=str(","))
            head = r.next()
        for i, col in enumerate(head):
            if i < csv_start_col_idx:
                continue
            samp = Sample(title=col, is_population=False)
            samp.load_from_csv(filename, i)
            self.groups.append(samp)

    def input_wizard(self, csv_filename=None, csv_start_col_idx=0):
        """Read input from the console interactively.
        """
        if csv_filename:
            self.load_from_csv(csv_filename, csv_start_col_idx=csv_start_col_idx)
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

    def spell_h0(self):
        """Spell the null hypothesis.
        """
        sym = " = ".join(["mu%d" % i for i in range(len(self.groups))])
        return "All samples are not significantly different (%s)" % (sym)

    def spell_hA(self):
        """Spell the alternate hypothesis. 
        """
        return "At least one sample is significantly different"

    def spell_type_of_test(self):
        """Spell the type of test"""
        return "Anova (always one directional positive)"

    def grand_mean(self):
        """The grand mean is the mean of all samples.
        """
        grand_sum = sum([samp.mean * samp.n for samp in self.groups])
        n_total = sum([samp.n for samp in self.groups])
        return grand_sum / float(n_total)

    def _ss_between(self, grand_mean=None):
        """Sum of squares between-groups."""
        if grand_mean is None:
            grand_sum = sum([samp.mean * samp.n for samp in self.groups])
            N = sum([samp.n for samp in self.groups])
            grand_mean = grand_sum / float(N)
        return sum([(samp.mean - grand_mean) ** 2 * samp.n for samp in self.groups])

    def _ss_within(self):
        """Sum of squares within-groups."""
        return sum([samp.sum_of_squared_diffs() for samp in self.groups])

    @classmethod
    def ss_total(cls, ss_b, ss_w):
        """Calculate total variation"""
        return ss_b + ss_w

    def df_between(self):
        """Degree of freedom for between-group. Also called DF numerator."""
        k = len(self.groups)
        return float(k - 1)

    def df_within(self, N=None):
        """Degree of freedom for within-group. Also called DF denumerator."""
        if N is None:
            N = sum([samp.n for samp in self.groups])
        k = len(self.groups)
        return float(N - k)

    @classmethod
    def df_total(cls, df_n, df_d):
        """Total degree of freedom is N-1.
        """
        return df_d + (df_n + 1) - 1

    def f_statistics(self):
        """Calculate the F-statistics or F-ratio
        """
        grand_sum = sum([samp.mean * samp.n for samp in self.groups])
        N = sum([samp.n for samp in self.groups])
        grand_mean = grand_sum / float(N)

        # k is number of samples
        k = len(self.groups)

        # bgvar is between-group variance
        bgvar = self._ss_between(grand_mean) / self.df_between()

        # wgvar is within-group variance
        wgvar = self._ss_within() / self.df_within(N)

        # The F-ratio
        return bgvar / wgvar

    def score(self):
        return self.f_statistics()

    def score_title(self):
        """The name for the score.
        """
        return "F-score"

    def critical(self):
        """Returns the critical value for the specified alpha. The critical value,
        expressed in the same unit as the standard score (see score()), is the
        standard value where the desired confidence is reached.
        """
        return StatTool.f_critical_value(self.alpha, self.df_between(), self.df_within())

    def critical_title(self):
        """The name for the critical value.
        """
        return "F-critical"

    def fall_in_critical_region(self):
        """Determine if the standard score falls IN the critical region.
        """
        return self.score() >= self.critical()

    def is_statistically_significant(self):
        """Determine if the results are statistically significant, which means
        it unlikely happens due to random chance or sampling error.
        """
        return self.fall_in_critical_region()

    def p_value(self):
        """Returns the probability (in proportion) for the result's score.
        """
        return StatTool.probability_for_f(self.score(), self.df_between(), self.df_within())

    def spell_reason(self):
        """Spell the reason why we reject/fail to reject the null hypothesis.
        """
        if self.fall_in_critical_region():
            return "p < %.3f (p=%.3f)" % (self.alpha, self.p_value())
        else:
            return "p > %.3f (p=%.3f)" % (self.alpha, self.p_value())

    def print_report(self):
        print("ANOVA REPORT:")
        print('=' * 70)
        for i, samp in enumerate(self.groups):
            print("Sample-%d: %s" % (i, samp.title))
            print("-" * 70)
            print(str(samp))

        print("Various descriptions:")
        print("-" * 70)
        print("Null hypothesis          : %s" % self.spell_h0())
        print("Alternate hypothesis     : %s" % self.spell_hA())
        print("Type of test             : %s" % self.spell_type_of_test())
        print("")
        print("Parameters:")
        print("-" * 70)
        print("alpha: %.3f" % self.alpha)
        print("")
        print("Results:")
        print("-" * 70)
        print("Grand mean:              % .3f" % self.grand_mean())
        print("Sum of sq. betwn groups: % .3f" % self._ss_between())
        print("Sum of sq. withn groups: % .3f" % self._ss_within())
        print("DF between groups:       % .3f" % self.df_between())
        print("DF within groups:        % .3f" % self.df_within())
        print("%-15s          % .3f" % (self.critical_title(), self.critical()))
        print("%-15s          % .3f" % (self.score_title(), self.score()))
        print("P-value:                 % .3f" % self.p_value())
        print("Fall in critical region:   %s" % self.fall_in_critical_region())
        print("Statistically significant: %s" % self.is_statistically_significant())
        print("")
        print("Conclusions:")
        print("-" * 70)
        print(" - %s" % (self.spell_hA() if self.fall_in_critical_region() else self.spell_h0()))
        sys.stdout.write(" - %s" % ("The null hypothesis is rejected"
                                    if self.fall_in_critical_region()
                                    else "Failed to reject the null hypothesis"))
        print(" because %s" % self.spell_reason())

    @classmethod
    def simple_calculation(cls, k, n, f_ratio, alphas):
        if f_ratio < 0:
            print("Cannot make decision because F is negative")
            return

        df_n = k - 1
        df_d = (k * n) - k
        print("df_n:        %d" % df_n)
        print("df_d:        %d" % df_d)
        print("F-ratio:    % .3f" % f_ratio)

        for alpha in alphas:
            critical = StatTool.f_critical_value(alpha, df_n, df_d)
            print("alpha %.3f: critical: %.3f conclusion: %s" %
                  (alpha, critical, "reject H0" if f_ratio >= critical else "accept H0"))

    @classmethod
    def simple_calculation2(cls, df_n, df_d, f_ratio, alphas):
        return cls.simple_calculation(df_n + 1, (cls.df_total(df_n, df_d) + 1) / (df_n + 1),
                                      f_ratio, alphas)
