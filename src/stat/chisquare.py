#!/usr/bin/env python
# -*- coding: utf-8-unix -*-
from __future__ import absolute_import, print_function, division, unicode_literals

import csv
import math

from scipy.stats import chisquare, chisqprob, chi2

import pandas as pd
from sample import Sample
from session import Session
from tools import StatTool, read_input


__author__ = "Benny Prijono <benny@stosia.com>"
__copyright__ = "Copyright (C)2017 PT. Stosia Teknologi Investasi"
__license__ = "GNU Affero AGPL (AGPL) version 3.0 or later"


class ChiSquare(Session):
    """This Chi-square (pronounced khai square) Test can be used for both Goodness-of-Fit and 
    Independence tests. For Goodness-of-fit, it looks at how well the observed values match 
    the expected values for certain variable. For test for independence, it tests whether or not
    two variables are independent.
    
    Spelt differently, the chi-square for goodness-of-fit tests whether two groups are the same, 
    while chi-square for independent test tests whether two variables have relationship or not.
    
    For goodness-of-fit, the data will contain observed values and expected values, something
    like this, and the Null hypothesis is the observed group is no different than the expected
    group: 
    
                Successful  Unsuccessful
    ------------------------------------
    Observed            41            59       
    Expected            33            67
    
    For independent test, the data looks like this, and we're testing the relationship of 
    two variables. In this case, the reponse (yes, no) and the type of verb used (Hit, Smashed, 
    and Control). 
    
                     Response    
                    Yes     No 
    ---------------------------
    Hit             7.0    43.0
    Smashed        16.0    34.0
    Control         6.0    44.0
    """

    def __init__(self):
        self.groups = []
        self.expected_index = None
        self.alpha = None

    def _fix_samples(self):
        """Perform pre-processing to the samples after they are input
        """
        pass

    def load_from_dict(self, d):
        """Load this instance from a dictionary.
        """
        self.alpha = d['alpha']
        self.expected_index = d['expected_index']
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
                print("Set title to 'Expected' to make it the expected group")
                samp = Sample(title='Observed-%d' % len(self.groups), is_population=False)
                samp.input_wizard(individual_sample=True)
                if not samp.n:
                    break
                if samp.title.lower() == 'expected':
                    self.expected_index = len(self.groups)
                self.groups.append(samp)
                more = read_input('Input more sample', default='y', choices='yn')
                if more != 'y':
                    break

        self.alpha = float(read_input("Alpha", default='0.05'))

    def count_verb_conditions(self):
        """Get the number of columns, or verb condition."""
        return len(self.groups)

    def count_response_types(self):
        return self.groups[0].n

    def n(self):
        expected_grp = self.groups[self.expected_index] if self.expected_index is not None else None
        return sum([sum(grp.members) for grp in self.groups if grp != expected_grp])

    def get_observed(self):
        results = []
        for resp_idx in range(self.groups[0].n):
            for grp_idx, grp in enumerate(self.groups):
                if self.expected_index is not None and grp_idx == self.expected_index:
                    continue
                results.append(grp.members[resp_idx])
        return results

    def get_expected(self):
        if self.expected_index is not None:
            return self.groups[self.expected_index].members
        else:
            results = []

            # nresp is total number of subjects across groups that says each response
            nresp = [0] * self.groups[0].n
            for grp in self.groups:
                for i, cnt in enumerate(grp.members):
                    nresp[i] += cnt

            # Total number of subjects
            n = sum(nresp)

            # Ratio of each response to the total
            resp_ratio = [cnt / float(n) for cnt in nresp]

            for resp_idx in range(self.groups[0].n):
                for grp_idx, grp in enumerate(self.groups):
                    grp_n = sum(grp.members)
                    results.append(resp_ratio[resp_idx] * grp_n)

            return results

    @classmethod
    def spell_conclusion_by_chi(cls, chi, chi_crit):
        """Spell conclusion on whether we reject or fail to reject the null hypothesis
        based on the Chi-square and the critical value
        """
        if chi > chi_crit:
            conclusion = "reject the null -> chi^2 (%.3f) > %.3f" % (chi, chi_crit)
        else:
            conclusion = "accept the null -> chi^2 (%.3f) < %.3f" % (chi, chi_crit)
        return conclusion

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

    def print_report(self):
        """Print report to stdout.
        """
        print("Chi-square REPORT:")
        print('=' * 70)

        print("Samples:")
        print('-' * 70)
        arr = []
        for grp in self.groups:
            arr.append(grp.members)
        df = pd.DataFrame(arr)
        df.columns = ['resp-' + chr(ord('1') + i) for i in range(self.groups[0].n)]
        df.index = [grp.title for grp in self.groups]
        print(df)
        print("")

        # Flatten the observed values
        observed = self.get_observed()
        # print("Observed                 :  %s" % str(observed))

        # Flatten the expected values
        expected = self.get_expected()
        # print("Expected                 :  %s" % str(expected))

        df = (self.count_response_types() - 1) * (self.count_verb_conditions() - 1)

        chi_square, p2 = chisquare(observed, expected)
        p = chisqprob(chi_square, df)

        # The Chi-square critical value for the specified alpha and DF
        chi_crit = chi2.isf(self.alpha, df)

        # Cramer's V measures the Effect Size, i.e. how strong the relationship
        # between two variables.
        k = min(self.count_verb_conditions(), self.count_response_types())
        cramers_v = math.sqrt(chi_square / (self.n() * (k - 1)))

        print("Results:")
        print('-' * 70)
        print("N                        :  %d" % self.n())
        print("Number of groups         :  %d" % self.count_verb_conditions())
        print("Number of response types :  %d" % self.count_response_types())
        print("DF                       :  %d" % df)
        print("Chi square               : % .3f" % chi_square)
        print("Critical value           : % .3f" % chi_crit)
        print("P-value                  : % .3f" % p)
        print("Conclusion               : - %s" % self.spell_conclusion_by_chi(chi_square, chi_crit))
        print("                           - %s" % self.spell_conclusion_by_p(p, self.alpha))
        print("Cramer's V               : % .3f" % cramers_v)
