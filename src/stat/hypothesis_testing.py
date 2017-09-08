#!/usr/bin/env python
# -*- coding: utf-8-unix -*-
from __future__ import absolute_import, print_function, division, unicode_literals

import math
import sys

from sample import Sample
from session import Session
from tools import read_input, StatTool


__author__ = "Benny Prijono <benny@stosia.com>"
__copyright__ = "Copyright (C)2017 PT. Stosia Teknologi Investasi"
__license__ = "GNU Affero AGPL (AGPL) version 3.0 or later"


class HypothesisTesting(Session):
    """Hypothesis testing is used to test whether some results happen because
    of a specific cause, rather than just random chance or sampling error.
    It is used for example to test whether the population has significantly
    changed after a treatment, whether population B is significantly different
    than population A, whether a sample is significantly different than
    the population, and so on.
    """

    def __init__(self, samp0_is_population=None, samp1_is_population=None):
        self.samp0 = Sample(title='samp0', is_population=samp0_is_population)
        """The first sample, is usually the population, or the pre-test sample"""

        self.samp1 = Sample(title='samp1', is_population=samp1_is_population)
        """The second sample, the post-test sample."""

        # Parameters
        self.alpha = 0.05
        """Requested confidence level"""

        self.dir = None  # see StatTool.xxx_TEST constants
        """Directionality of the test."""

        # Expected difference
        self.expected_difference = 0.0
        """The expected difference between the two sample means. 
        
        For two tailed test, usually we write the hypothesis as μ1 != μ2.
        This can be rewritten as μ1 - μ2 != 0. And actually the general 
        expression is μ1 - μ2 != expected_difference.
        """

        # Description
        self.treatment_title = "treatment"
        """The name of the treatment"""

        self.results_title = "results"
        """The name of the dependent variable"""

    def name(self):
        """The name of this test.
        """
        raise RuntimeError("Missing implementation")

    def load_from_dict(self, d):
        """Load this instance from a dictionary.
        """
        if d.get('samp0'):
            self.samp0.load_from_dict(d.get('samp0'))

        if d.get('samp1'):
            self.samp1.load_from_dict(d.get('samp1'))

        # Parameters
        self.alpha = d.get('alpha', 0.05)
        self.dir = d.get('dir', None)
        self.expected_difference = d.get('expected_difference', 0.0)

        # Description
        self.treatment_title = d.get('treatment_title', "treatment")
        self.results_title = d.get('results_title', "results")

        self._fix_samples()

    def save_to_dict(self):
        """Save this instance to a dictionary to be serialized.
        """
        return self.__dict__

    def _fix_samples(self):
        """Perform pre-processing to the samples after they are input
        """
        raise RuntimeError("Missing implementation")

    def input_wizard(self, csv_filename=None, csv_start_col_idx=0):
        """Wizard to input the parameters from console.
        """
        if csv_filename:
            # self.samp0.is_population = read_input('Treat first sample as population',
            #                                      default='n', choices="yn") == 'y'
            self.samp0.load_from_csv(csv_filename, csv_start_col_idx + 0)

            # self.samp1.is_population = read_input('Treat second sample as population',
            #                                      default='n', choices="yn") == 'y'
            self.samp1.load_from_csv(csv_filename, csv_start_col_idx + 1)
        else:
            print("Input the first sample (samp0)")
            self.samp0.input_wizard()

            print("")
            print("Input the second sample (samp1)")
            ref_pop = self.samp0 if self.samp0.is_population else None
            self.samp1.input_wizard(require_n=True, ref_pop=ref_pop)

        print("Info: mean difference: %.3f" % self.mean_difference())

        self._fix_samples()

        print("")
        independent_t_test = self.__class__.__name__ == "IndependentTTesting"
        if not independent_t_test:
            self.treatment_title = read_input("The name of the treatment",
                                              default=self.treatment_title, optional=True)
            self.results_title = read_input("The name of the results",
                                            default=self.results_title, optional=True)

        self.dir = read_input('Directionality: Two tailed (t), one tailed negative (n), or one tailed positive (p)',
                              default=StatTool.TWO_TAILED_TEST, choices=StatTool.valid_dirs)

        self.alpha = float(read_input("Alpha", default='%.03f' % self.alpha))
        self.expected_difference = float(read_input("Expected difference", default='0.0',
                                                    optional=True))
        print("Critical value: %.3f" % self.critical())
        return self

    def spell_h0(self):
        """Spell the null hypothesis. Null hypothesis assumes that there is no 
        significant difference between current population parameters and what 
        will be the new population parameters after some sort of an intervention. 
        
        Independent T-test will override this.
        """
        if self.dir == StatTool.TWO_TAILED_TEST:
            return "\"%s\" DOES NOT change \"%s\"" % (self.treatment_title, self.results_title)
        elif self.dir == StatTool.ONE_TAILED_POSITIVE_TEST:
            return "\"%s\" DOES NOT increase \"%s\"" % (self.treatment_title, self.results_title)
        else:
            return "\"%s\" DOES NOT reduce \"%s\"" % (self.treatment_title, self.results_title)

    def spell_hA(self):
        """Spell the alternate hypothesis. Alternative hypothesis guesses that there 
        will be significant differences, either less than, greater than, or just 
        different without saying particular direction, between current population and
        what will be the new population after some sort of intervention. Alternative 
        hypothesis is the objective that we want to achieve if we are doing intervention.
        
        Independent T-test will override this.
        """
        if self.dir == StatTool.TWO_TAILED_TEST:
            return "\"%s\" DOES change \"%s\"" % (self.treatment_title, self.results_title)
        elif self.dir == StatTool.ONE_TAILED_POSITIVE_TEST:
            return "\"%s\" DOES increase \"%s\"" % (self.treatment_title, self.results_title)
        else:
            return "\"%s\" DOES reduce \"%s\"" % (self.treatment_title, self.results_title)

    def spell_type_of_test(self):
        """Spell the type of t-test"""
        return StatTool.spell_directionality(self.dir)

    def mean_difference(self):
        """The mean difference calculates the difference between the mean
        of the second sample and the mean of the first sample.
        """
        return self.samp1.mean - self.samp0.mean

    def standard_deviation_difference(self):
        """Calculate the standard deviation difference between two
        samples' standard deviations.
        """
        return math.sqrt(self.samp0.orig_sd ** 2 + self.samp1.orig_sd ** 2)

    def fall_in_critical_region(self):
        """Determine if the standard score falls IN the critical region.
        """
        score = self.score()
        crit = self.critical()
        if self.dir == StatTool.TWO_TAILED_TEST:
            # TODO: with equal or not?
            return score <= -crit or score > crit
        # TODO: with equal or not?
        elif self.dir == StatTool.ONE_TAILED_POSITIVE_TEST:
            return score > crit
        else:
            return score <= crit

    def is_statistically_significant(self):
        """Determine if the results are statistically significant, which means
        it unlikely happens due to random chance or sampling error.
        """
        return self.fall_in_critical_region()

    def is_meaningful(self):
        """Determine if the results are meaningful. This is of course highly
        subjective!"""
        return "it depends"

    def cohens_d(self):
        """Cohen's D (named after statistician Jacob Cohen) measures the distance
        between two means in standardized unit (hence similar to Z).
        """
        return self.mean_difference() / self.samp1.sd

    def margin_of_error(self):
        """Margin of error is the distance from μ (in either direction), expressed
        in the same unit as the original sample, where it is within the confidence
        interval (CI). Margin of error is half the width of the CI.
        
        For example, if μ is $50, and alpha is 5%, and Standard Error is $15.
        Then for 95% confidence, the confidence interval is between $50 - (1.96 * $15)
        and $50 + (1.96 * $15). In this case, (1.96 * $15) is the margin of error. 
        """
        raise RuntimeError("Missing implementation")

    def confidence_interval(self):
        """Returns the confidence interval (CI) as sequence (low, hi). 
        CI is the inverval where we are (1-alpha) percent sure that the sample values
        will fall in this interval.
        """
        merr = self.margin_of_error()
        return (self.samp1.mean - merr, self.samp1.mean + merr)

    def SEM(self):
        """The standard error or standard error of mean.
        """
        raise RuntimeError("Missing implementatoin")

    def score(self):
        """Returns the standard score. The standard score measures the difference
        of two sample means in standard unit, which in this case can be Z score
        or T-statistics, depending on the the test being performed.
        """
        raise RuntimeError("Missing implementatoin")

    def score_title(self):
        """The name for the score.
        """
        raise RuntimeError("Missing implementatoin")

    def critical(self):
        """Returns the critical value for the specified alpha. The critical value,
        expressed in the same unit as the standard score (see score()), is the
        standard value where the desired confidence is reached.
        """
        raise RuntimeError("Missing implementatoin")

    def critical_title(self):
        """The name for the critical value.
        """
        raise RuntimeError("Missing implementatoin")

    def p_value(self):
        """Returns the probability (in proportion) for the result's score.
        """
        raise RuntimeError("Missing implementatoin")

    def spell_reason(self):
        """Spell the reason why we reject/fail to reject the null hypothesis.
        """
        if self.fall_in_critical_region():
            return "p < %.3f (p=%.3f)" % (self.alpha, self.p_value())
        else:
            return "p > %.3f (p=%.3f)" % (self.alpha, self.p_value())

    def print_extra_params(self):
        pass

    def print_extra_results(self):
        pass

    def print_extra_conclusions(self):
        pass

    def print_report(self):
        print("%s REPORT:" % self.name())
        print('=' * 70)
        print("Sample-0: %s" % self.samp0.title)
        print("-" * 70)
        print(str(self.samp0))

        print("Sample-1: %s" % self.samp1.title)
        print("-" * 70)
        print(str(self.samp1))

        print("Various descriptions:")
        print("-" * 70)
        print("The dependent variable is:  %s" % self.results_title)
        print("The treatment            :  %s" % self.treatment_title)
        print("Null hypothesis          :  %s" % self.spell_h0())
        print("Alternate hypothesis     :  %s" % self.spell_hA())
        print("Type of test             :  %s" % self.spell_type_of_test())
        print("")
        print("Parameters:")
        print("-" * 70)
        print("Alpha                    :  %.3f" % self.alpha)
        self.print_extra_params()
        print("")
        print("Results:")
        print("-" * 70)
        print("mean difference          : % .3f" % self.mean_difference())
        if self.samp0.orig_sd is not None and self.samp1.orig_sd is not None:
            print("SD difference            : % .3f" % self.standard_deviation_difference())
        print("SEM                      : % .3f" % self.SEM())
        print("%-15s          : % .3f" % (self.critical_title(), self.critical()))
        print("%-15s          : % .3f" % (self.score_title(), self.score()))
        print("p-value                  : % .3f" % self.p_value())
        self.print_extra_results()
        print("Margin of error          : % .3f" % self.margin_of_error())
        print("Confidence interval      : % .3f - %.3f" % self.confidence_interval())
        print("Fall in critical region  :  %s" % self.fall_in_critical_region())
        print("Statistically significant:  %s" % self.is_statistically_significant())
        print("")
        print("Conclusions:")
        print("-" * 70)
        print(" - %s" % (self.spell_hA() if self.fall_in_critical_region() else self.spell_h0()))
        sys.stdout.write(" - %s" % ("The null hypothesis is rejected"
                                    if self.fall_in_critical_region()
                                    else "Failed to reject the null hypothesis"))
        print(" because %s" % self.spell_reason())
        self.print_extra_conclusions()


class ZTesting(HypothesisTesting):
    """This hypothesis testing is used when we have the population parameters,
    especially the standard deviation. Then given a sample, we would like to
    know if the results happens due to certain cause and not just random chance
    or sampling errors."""
    def __init__(self):
        HypothesisTesting.__init__(self, samp0_is_population=True, samp1_is_population=False)

    def name(self):
        """The name of this test.
        """
        return "Z-Test"

    def _fix_samples(self):
        # Nothing to do
        pass

    def SEM(self):
        """Returns the standard error, or standard error of mean. A standard error
        is the standard deviation of the sampling distribution of a statistic.
        """
        return self.samp0.sd / math.sqrt(self.samp1.n)

    def score(self):
        """The Z-score"""
        return (self.mean_difference() - self.expected_difference) / self.SEM()

    def score_title(self):
        """The name for the score is Z-score"""
        return "Z-score"

    def critical(self):
        """Z-critical value for the specified alpha and direction"""
        return StatTool.z_critical_value(self.alpha, self.dir)

    def critical_title(self):
        """The name for the critical value is z-critical"""
        return "z-critical"

    def margin_of_error(self):
        """Margin of error is the distance from μ (in either direction), expressed
        in the same unit as the original sample, where it is within the confidence
        interval (CI). Margin of error is half the width of the CI.
        
        For example, if μ is $50, and alpha is 5%, and Standard Error is $15.
        Then for 95% confidence, the confidence interval is between $50 - (1.96 * $15)
        and $50 + (1.96 * $15). In this case, (1.96 * $15) is the margin of error. 
        """
        z_critical2 = StatTool.z_critical_value(self.alpha, StatTool.TWO_TAILED_TEST)
        return z_critical2 * self.SEM()

    def p_value(self):
        """Returns the probability (in proportion) for the result's z-score.
        """
        return StatTool.probability_for_z(self.score(), self.dir)

    def print_extra_conclusions(self):
        print("")
        print("Additional note:")
        print("-" * 70)
        if self.is_statistically_significant():
            print(" - Type I error would be if H0 is true but we reject that")
        else:
            print(" - Type II error would be if H1 is true but we reject that")


class TTesting(HypothesisTesting):
    """This hypothesis testing is used when we do not have the population
    parameters."""
    def __init__(self):
        HypothesisTesting.__init__(self, samp0_is_population=False, samp1_is_population=False)

    def df(self):
        raise RuntimeError("Missing implementation")

    def critical(self):
        """t-critical value for the specified confidence/alpha.
        Value may be negative!"""
        return StatTool.t_critical_value(self.alpha, self.dir, self.df())

    def t_statistics(self):
        """Returns the t-statistic value"""
        return (self.mean_difference() - self.expected_difference) / self.SEM()

    def score(self):
        return self.t_statistics()

    def score_title(self):
        """The name for the score"""
        return "t-statistics"

    def critical_title(self):
        """The name for the critical value"""
        return "t-critical"

    def margin_of_error(self):
        """Margin of error is the distance from μ (in either direction), expressed
        in the same unit as the original sample, where it is within the confidence
        interval (CI). Margin of error is half the width of the CI.
        
        For example, if μ is $50, and alpha is 5%, and Standard Error is $15.
        Then for 95% confidence, the confidence interval is between $50 - (1.96 * $15)
        and $50 + (1.96 * $15). In this case, (1.96 * $15) is the margin of error. 
        """
        # Margin of error always uses two tailed
        t_critical2 = StatTool.t_critical_value(self.alpha,
                                                StatTool.TWO_TAILED_TEST,
                                                self.df())
        return t_critical2 * self.SEM()

    def p_value(self):
        """Returns the probability (in proportion) for the result's t-statistic value.
        """
        return StatTool.probability_for_t(self.t_statistics(), self.dir, self.df())

    def r_squared(self):
        """R^2, or sometimes called coefficient of determination, is proportion (%) of 
        variation (=change) in one variable that is related to ("explained by") 
        another variable.
        """
        t = self.t_statistics()
        return t ** 2 / (t ** 2 + self.df())

    def print_extra_params(self):
        print("df                       :  %d" % self.df())

    def print_extra_results(self):
        print("Cohen's d                : % .3f" % self.cohens_d())
        print("r^2                      : % .3f" % self.r_squared())


class DependentTTesting(TTesting):
    """This hypothesis testing is used when we do not have the population
    parameters."""
    def __init__(self):
        TTesting.__init__(self)

    def name(self):
        """The name of this test.
        """
        return "Dependent T-Test"

    def _fix_samples(self):
        """Decide if the two samples are dependent and we should do something
        about it"""
        samp_dependent = not self.samp0.is_population and not self.samp1.is_population
        if samp_dependent and self.samp0.members and self.samp1.members:
            members = [self.samp1.members[i] - self.samp0.members[i]
                       for i in range(len(self.samp1.members))]
            self.samp0.mean = 0.0
            self.samp0.sd = None
            self.samp0.notes = "mean and sd have been reset"
            self.samp1.mean = StatTool.calc_mean(members)
            self.samp1.sd = StatTool.calc_sd(members, self.samp1.is_population)
            self.samp1.notes = "mean and sd are difference from sample-0"
        elif samp_dependent and self.samp0.orig_sd is not None and self.samp1.orig_sd is not None:
            self.samp1.sd = self.standard_deviation_difference()
            self.samp1.notes = "sd is difference from sample-0"
            # self.samp0.sd = 0
            # self.samp0.notes = "sd has been reset"

    def df(self):
        """Degrees of freedom"""
        return self.samp1.n - 1

    def SEM(self):
        """Standard error of mean"""
        return self.samp1.sd / math.sqrt(self.samp1.n)


class IndependentTTesting(TTesting):
    """This hypothesis testing is used when we do not have the population
    parameters. With the independent T-Testing, it is assumed that:
    
    1) the two samples are random samples from two independent populations,
    2) the populations are approximately normal. This is less important
       if n is large (>30)
    3) the sample data can be used to estimate the population variance
    4) the population variances are roughly equal
    """
    def __init__(self):
        TTesting.__init__(self)

    def name(self):
        """The name of this test.
        """
        return "Independent T-Test"

    def _fix_samples(self):
        pass

    def df(self):
        """Degrees of freedom"""
        return self.samp0.n + self.samp1.n - 2

    def pooled_variance(self):
        """Get the pooled variance. The pool variance is used to calculate SEM
        when we cannot assume that the size of the two samples are approximately
        the same.
        """
        # return (self.samp0.sum_of_squared_diffs() +
        #         self.samp1.sum_of_squared_diffs()) / self.df()
        return (self.samp0.sd ** 2 * (self.samp0.n - 1) +
                self.samp1.sd ** 2 * (self.samp1.n - 1)) / self.df()

    def uncorrected_SEM(self):
        """(Uncorrected) Standard error of mean. This assumes that the two samples are
        approximately the same size.
        """
        return math.sqrt(self.samp0.sd ** 2 / float(self.samp0.n) +
                         self.samp1.sd ** 2 / float(self.samp1.n))

    def corrected_SEM(self):
        """Standard error calculated using the pooled variance.
        """
        sp = self.pooled_variance()
        return math.sqrt(sp / float(self.samp0.n) + sp / float(self.samp1.n))

    def SEM(self):
        return self.corrected_SEM()

    def spell_h0(self):
        """Spell the null hypothesis"""
        if self.dir == StatTool.TWO_TAILED_TEST:
            return "\"%s\" = \"%s\"" % (self.samp0.title, self.samp1.title)
        elif self.dir == StatTool.ONE_TAILED_POSITIVE_TEST:
            return "\"%s\" <= \"%s\"" % (self.samp0.title, self.samp1.title)
        else:
            return "\"%s\" >= \"%s\"" % (self.samp0.title, self.samp1.title)

    def spell_hA(self):
        """Spell the alternate hypothesis"""
        if self.dir == StatTool.TWO_TAILED_TEST:
            return "\"%s\" is significantly different than \"%s\"" % \
                   (self.samp0.title, self.samp1.title)
        elif self.dir == StatTool.ONE_TAILED_POSITIVE_TEST:
            return "\"%s\" is significantly greater than \"%s\"" % \
                   (self.samp0.title, self.samp1.title)
        else:
            return "\"%s\" is significantly less than \"%s\"" % \
                   (self.samp0.title, self.samp1.title)

    def confidence_interval(self):
        """Returns the confidence interval (CI) as sequence (low, hi). 
        CI is the inverval where we are (1-alpha) percent sure that the sample values
        will fall in this interval.
        
        For independent T-Testing, this is for true difference between the two samples,
        so we shouldn't use the mean of a single sample as the center"""
        md = abs(self.mean_difference())
        merr = self.margin_of_error()
        return (md - merr, md + merr)

    def print_extra_results(self):
        TTesting.print_extra_results(self)
        print("Pooled variance          : % .3f" % self.pooled_variance())
        # print("Corrected SEM            % .3f" % self.corrected_SEM())


