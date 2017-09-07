#!/usr/bin/env python
# -*- coding: utf-8-unix -*-
from __future__ import absolute_import, print_function, division, unicode_literals

import copy
import csv
import math
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

        _cache = {}

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
        self._cache = {}

    def save_to_dict(self):
        """Save this instance to a dictionary to be serialized.
        """
        d = copy.copy(self.__dict__)
        del d['_cache']
        return d

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
        self._cache = {}

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

    def get_cached(self, varname, _calc):
        value = self._cache.get(varname, None)
        if value is not None:
            return value

        value = _calc(self)
        self._cache[varname] = value
        return value

    def N(self):
        """Get total number of values.
        """
        def _calc(self):
            return sum([samp.n for samp in self.groups])
        return self.get_cached('N', _calc)

    def grand_sum(self):
        """Sum of all values."""
        def _calc(self):
            return sum([samp.mean * samp.n for samp in self.groups])
        return self.get_cached('grand_sum', _calc)

    def grand_mean(self):
        """The grand mean is the mean of all samples.
        """
        def _calc(self):
            grand_sum = self.grand_sum()
            N = self.N()
            grand_mean = grand_sum / float(N)
            return grand_mean
        return self.get_cached('grand_mean', _calc)

    def ss_between(self):
        """Sum of squares between-groups."""
        def _calc(self):
            grand_sum = self.grand_sum()
            N = self.N()
            grand_mean = grand_sum / float(N)
            val = sum([(samp.mean - grand_mean) ** 2 * samp.n for samp in self.groups])
            return val
        return self.get_cached('ss_between', _calc)

    def ss_within(self):
        """Sum of squares within-groups."""
        def _calc(self):
            return sum([samp.sum_of_squared_diffs() for samp in self.groups])
        return self.get_cached('ss_within', _calc)

    def ss_total(self):
        """Calculate total variation"""
        return self.ss_between() + self.ss_within()

    def df_between(self):
        """Degree of freedom for between-group. Also called DF numerator."""
        k = len(self.groups)
        return float(k - 1)

    def df_n(self):
        return self.df_between()

    def df_within(self):
        """Degree of freedom for within-group. Also called DF denumerator."""
        k = len(self.groups)
        return float(self.N() - k)

    def df_d(self):
        return self.df_within()

    @classmethod
    def df_total(cls, df_n, df_d):
        """Total degree of freedom is N-1.
        """
        return df_d + (df_n + 1) - 1

    def mean_squares_between(self):
        """Mean Squares between"""
        return self.ss_between() / self.df_between()

    def mean_squares_within(self):
        """Mean Squares within"""
        return self.ss_within() / self.df_within()

    def f_statistics(self):
        """Calculate the F-statistics or F-ratio
        """
        def _calc(self):
            return self.mean_squares_between() / self.mean_squares_within()
        return self.get_cached('f_score', _calc)

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
        def _calc(self):
            return StatTool.f_critical_value(self.alpha, self.df_between(), self.df_within())
        return self.get_cached('f_critical', _calc)

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
        def _calc(self):
            return StatTool.probability_for_f(self.score(), self.df_n(), self.df_d())
        return self.get_cached('p_value', _calc)

    def spell_reason(self):
        """Spell the reason why we reject/fail to reject the null hypothesis.
        """
        if self.fall_in_critical_region():
            return "p < %.3f (p=%.3f)" % (self.alpha, self.p_value())
        else:
            return "p > %.3f (p=%.3f)" % (self.alpha, self.p_value())

    def eta_squared(self):
        """Get the proportion of total variation that is due to between-group differences
        (explained variation).
        """
        return self.ss_between() / float(self.ss_total())

    def tukeys_hsd(self):
        """Calculate and return Tukey's HSD value.
        """
        def _calc(self):
            qstar = StatTool.q_value(self.alpha, self.df_n(), self.df_d())
            msw = self.mean_squares_within()
            k = len(self.groups)
            return qstar * math.sqrt(sum([msw / grp.n for grp in self.groups]) / k)
        return self.get_cached('tukeys_hsd', _calc)

    def print_difference(self):
        hsd = self.tukeys_hsd()
        maxtitlelen = max([len(sample.title) for sample in self.groups])
        for i in range(len(self.groups)):
            grp0 = self.groups[i]
            for j in range(i + 1, len(self.groups)):
                grp1 = self.groups[j]
                abs_diff = abs(grp0.mean - grp1.mean)
                cohens_d = (grp0.mean - grp1.mean) / math.sqrt(self.mean_squares_within())
                relation = '!=' if abs_diff > hsd else ' ='
                print("%*s %s %-*s  abs.diff: %.3f, Cohen's d: %.3f" %
                      (maxtitlelen, grp0.title, relation, maxtitlelen, grp1.title, abs_diff, cohens_d))

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
        print("Alpha                    :  %.3f" % self.alpha)
        print("")
        print("Results:")
        print("-" * 70)
        print("Grand mean               : % .3f" % self.grand_mean())
        print("Sum of sq. betwn groups  : % .3f" % self.ss_between())
        print("Sum of sq. withn groups  : % .3f" % self.ss_within())
        print("DF between groups (df_n) : % .3f" % self.df_between())
        print("DF within groups (df_d)  : % .3f" % self.df_within())
        print("Mean squares between     : % .3f" % self.mean_squares_between())
        print("Mean squares within      : % .3f" % self.mean_squares_within())
        print("%-15s          : % .3f" % (self.critical_title(), self.critical()))
        print("%-15s          : % .3f" % (self.score_title(), self.score()))
        print("P-value                  : % .3f" % self.p_value())
        print("Fall in critical region  :  %s" % self.fall_in_critical_region())
        print("Statistically significant:  %s" % self.is_statistically_significant())
        print("Tukey's HSD              : % .3f" % self.tukeys_hsd())
        print("Eta squared              : % .3f" % self.eta_squared())
        print("")
        print("Difference matrix:")
        print("-" * 70)
        self.print_difference()
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
