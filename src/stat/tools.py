#!/usr/bin/env python
# -*- coding: utf-8-unix -*-
from __future__ import absolute_import, print_function, division, unicode_literals

import math
import sys

from scipy import stats
import qsturng


__author__ = "Benny Prijono <benny@stosia.com>"
__copyright__ = "Copyright (C)2017 PT. Stosia Teknologi Investasi"
__license__ = "GNU Affero AGPL (AGPL) version 3.0 or later"


def read_input(title, default=None, choices=None, optional=False):
    """Utility to prompt input from user via the console."""
    msg = title
    if choices:
        msg += ' (%s)' % ('/'.join(choices))
    msg += ':'
    if default:
        msg += ' [%s]' % (default)
    msg += ' '

    while True:
        if sys.version_info >= (3, 0):
            s = input(msg).strip()
        else:
            s = raw_input(msg).strip()
        if default and not s:
            s = default
        if choices and s not in choices:
            continue
        if s or optional:
            break

    return s


class StatTool:
    """Various statistical tools"""

    # The direction of hypothesis test
    TWO_TAILED_TEST = 't'
    ONE_TAILED_NEGATIVE_TEST = 'n'
    ONE_TAILED_POSITIVE_TEST = 'p'

    valid_dirs = [TWO_TAILED_TEST, ONE_TAILED_NEGATIVE_TEST,
                  ONE_TAILED_POSITIVE_TEST]

    @classmethod
    def calc_mean(cls, array):
        """Calculate the mean of the samples"""
        return sum(array) / float(len(array))

    @classmethod
    def calc_sum_squared_diffs(cls, array):
        """Calculate the sum of squared difference"""
        mean = cls.calc_mean(array)
        return sum([(m - mean) ** 2 for m in array])

    @classmethod
    def calc_sd(cls, array, is_population):
        """Calculate the standard deviation of the samples, either treating
        the sample as population or as sample"""
        n = len(array)
        mean = cls.calc_mean(array)
        squared_diffs = [(m - mean) ** 2 for m in array]
        sum_squared_diff = sum(squared_diffs)
        if is_population:
            return math.sqrt(sum_squared_diff / float(n))
        else:
            return math.sqrt(sum_squared_diff / float(n - 1))

    @classmethod
    def spell_directionality(cls, dir):
        """Spell directionality of the test"""
        assert dir in cls.valid_dirs

        if dir == StatTool.TWO_TAILED_TEST:
            return "two tailed"
        elif dir == StatTool.ONE_TAILED_POSITIVE_TEST:
            return "one tailed in positive direction"
        else:
            return "one tailed in negative direction"

    @classmethod
    def probability_for_z(cls, z_score, dir):
        """Calculate the actual probability value for the z_score.
        
        For positive z_score, this returns the probability that any samples
        will have Z AT LEAST this value.
        
        For negative z_score, this returns the probability that any samples
        will have Z LESS THAN this value.
        """
        assert dir in cls.valid_dirs

        # cdv is the probability that any sample will have LESS than z_score
        cdv = stats.norm.cdf(z_score)

        if dir == cls.TWO_TAILED_TEST:
            if z_score < 0:
                dir = cls.ONE_TAILED_NEGATIVE_TEST
            else:
                dir = cls.ONE_TAILED_POSITIVE_TEST

        if dir == cls.ONE_TAILED_POSITIVE_TEST:
            return 1 - cdv
        else:
            return cdv

    @classmethod
    def probability_for_t(cls, t_statistic, dir, df):
        """Calculate the probability value for the specified t_statistics and DF.
        
        For one tailed positive, this returns the probability that any samples
        will have t-stat AT LEAST this value.
        
        For one tailed negative, this returns the probability that any samples
        will have t-stat LESS THAN this value.
        
        For two tailed test, this returns the probability that any samples will
        have LESS than minus t_statistic OR MORE THAN t_statistic.
        """
        # pval is the probability that any samples will have
        # EQUAL OR MORE THAN abs(t_statistic).
        #
        # FWIW sf = survival function
        pval = stats.t.sf(abs(t_statistic), df)
        return pval * 2 if dir == StatTool.TWO_TAILED_TEST else pval

    @classmethod
    def probability_for_f(cls, f_score, df_n, df_d):
        """Calculate the actual probability value for the f_score
        """
        # cdv is the probability that any sample will have LESS than f_score
        cdv = stats.f.cdf(f_score, df_n, df_d)
        return 1 - cdv

    @classmethod
    def z_critical_value(cls, alpha, dir):
        """Return the Z-critical value for the specified alpha and directionality.
        For two tailed, the alpha will be halved.
        For one tailed negative, the z-critical value will be negative. 
        """
        if dir == cls.TWO_TAILED_TEST:
            alpha /= 2.0

        z = stats.norm.ppf(1 - alpha)
        if dir == StatTool.ONE_TAILED_NEGATIVE_TEST:
            z = 0 - z
        return z

    @classmethod
    def t_critical_value(cls, alpha, dir, df):
        """Return the T-critical value for the specified,  and directionality,
        and degrees of freedom.
        For two tailed, the alpha will be halved.
        For one tailed negative, the t-critical value will be negative. 
        """
        if dir == cls.TWO_TAILED_TEST:
            alpha /= 2.0

        t = stats.t.ppf(1 - alpha, df)
        if dir == cls.ONE_TAILED_NEGATIVE_TEST:
            t = 0 - t
        return t

    @classmethod
    def f_critical_value(cls, alpha, df_n, df_d):
        """Return the F-critical value for the specified alpha and degree of
        freedoms. 
        """
        return stats.f.ppf(1 - alpha, df_n, df_d)

    @classmethod
    def q_value(cls, alpha, df_n, df_d):
        """Get the Studentized Range Statistics (q*) value for the specified
        alpha, df_n (between-group DF), and df_d (within-group DF)
        """
        return qsturng.qsturng(1 - alpha, df_n + 1, df_d)
