#!/usr/bin/env python
# -*- coding: utf-8-unix -*-
from __future__ import absolute_import, print_function, division, unicode_literals

import csv
import re

from session import Session
from tools import read_input, StatTool


__author__ = "Benny Prijono <benny@stosia.com>"
__copyright__ = "Copyright (C)2017 PT. Stosia Teknologi Investasi"
__license__ = "GNU Affero AGPL (AGPL) version 3.0 or later"


class Sample(Session):
    """Sample is an observation against a particular subject at one point in time."""

    def __init__(self, title='', is_population=None):
        self.title = title
        """Optional title for this sample"""

        self.notes = ''
        """Additional notes, such as treatment to SD and mean, if any."""

        self.is_population = is_population
        """Treat as population instead of sample?"""

        self.mean = None  # The mean
        """The sample mean. Must be specified."""

        self.orig_mean = None
        """The sample mean that was input by user. May be None"""

        self.sd = None
        """Sample standard deviation. May be None if not relevant"""

        self.orig_sd = None
        """The standard deviation that was input by user. May be None."""

        self.n = None
        """Number of samples. May be None if not relevant"""

        self.members = None
        """List of each individual sample. May be None."""

    def __str__(self):
        s = ''
        if True:
            s += "Title                    :  %s\n" % self.title
            s += "Notes                    :  %s\n" % self.notes
            s += "Treat as population      :  %s\n" % self.is_population
        if self.n is not None:
            s += "%s                        : % d\n" % ("N" if self.is_population else "n", self.n)
        if self.members:
            if len(self.members) <= 12:
                val = str(self.members)
            else:
                val = '[' + ", ".join([str(v) for v in self.members[:8]]) + \
                      ' ... ' + ", ".join([str(v) for v in self.members[-2:]]) + ']'
            s += "Members                  :  %s\n" % (val)
        if True:
            s += "Mean                     : % .3f\n" % (self.mean)
            if self.orig_mean is not None:
                s += "Orig. Mean               : % .3f\n" % (self.orig_mean)
            else:
                s += "Orig. Mean               :  None\n"
        if True:
            s += "Sum of squared diffs     : % .3f\n" % (self.sum_of_squared_diffs())
        if self.sd is not None:
            s += "SD                       : % .3f\n" % (self.sd)
            if self.orig_sd is not None:
                s += "Orig. SD                 : % .3f\n" % (self.orig_sd)
            else:
                s += "Orig. SD                 :  None\n"
        return s

    def load_from_dict(self, d):
        """Load this Sample object from a dictionary.
        """
        self.title = d.get('title', '')
        self.notes = d.get('notes', '')
        self.is_population = d.get('is_population', None)
        self.mean = d.get('mean', None)
        self.orig_mean = d.get('orig_mean', None)
        self.sd = d.get('sd', None)
        self.orig_sd = d.get('orig_sd', None)
        self.n = d.get('n', None)
        self.members = d.get('members', None)

    def save_to_dict(self):
        """Save this instance to a dictionary to be serialized.
        """
        return self.__dict__

    def load_from_csv(self, filename, column_idx, is_population=None):
        """Load the members from CSV file. The title of the sample will be taken
        from the column header.
        """
        f = open(filename)
        r = csv.reader(f, delimiter=str(","))
        head = r.next()
        if len(head) < column_idx + 1:
            raise RuntimeError("The CSV needs to have at least %d column(s)" % (column_idx + 1))

        self.title = head[column_idx]
        self.members = []
        if is_population is not None:
            self.is_population = is_population

        for row in r:
            if len(row) >= column_idx + 1:
                cell = row[column_idx].strip()
                if cell:
                    self.members.append(float(cell))

        self._update_parameters()

    def input_samples(self, s=None):
        """Parse the samples if given in s, or input from console.
        """
        if not s:
            s = read_input('Individual samples (space or comma separated)')
        self.members = re.findall(r"[\w\.\-\+]+", s)
        self.members = [m.strip() for m in self.members]
        self.members = [float(m) for m in self.members if m]
        self._update_parameters()

    def _update_parameters(self):
        """Update mean sd etc after we have updated our samples
        """
        self.n = len(self.members)
        self.mean = self.orig_mean = StatTool.calc_mean(self.members)
        self.sd = self.orig_sd = StatTool.calc_sd(self.members, self.is_population)
        print("Got %d samples, mean: %.3f, sd: %.3f" % (self.n, self.mean, self.sd))

    def input_wizard(self, require_n=False, ref_pop=None, individual_sample=None,
                     csv_filename=None, csv_indices=None):
        """Wizard to input the parameters from console.
        """
        if not csv_filename:
            s = read_input('The name of this sample', default=self.title, optional=True)
            if s:
                self.title = s

        if self.is_population is None:
            s = read_input('Treat as population', default='n', choices=['y', 'n'])
            self.is_population = True if s == 'y' else False

        if csv_filename:
            col_idx = csv_indices[0] if csv_indices else 0
            self.load_from_csv(csv_filename, col_idx)
            return self

        if not self.title:
            self.title = "population" if self.is_population else "sample"

        if individual_sample is None:
            s = read_input('Input parameters or individual sample', default='p',
                           choices=['p', 'i'])
            individual_sample = s == 'i'
        if not individual_sample:
            s = read_input('n (number of data)', optional=not require_n)
            if s:
                self.n = int(s)

            self.mean = self.orig_mean = float(read_input('Mean'))

            if ref_pop and (not ref_pop.is_population or ref_pop.sd is None):
                ref_pop = None

            title = 'Standard deviation%s: ' % \
                    (' (skip to calculate from population)' if ref_pop else '')
            s = read_input(title)
            if s:
                self.sd = self.orig_sd = float(s)
            elif ref_pop:
                self.sd = ref_pop.sd / math.sqrt(self.n)
                self.notes = "SD is derived from population"
                print("Note: Calculating SD as Standard Error from population.")
                print("      SE: %.3f." % self.sd)
        else:
            self.input_samples()

        return self

    def sum_of_squared_diffs(self):
        """Return the sum of squared difference between each member
        and the mean.
        """
        if self.sd is not None:
            return (self.sd ** 2) * (self.n if self.is_population else (self.n - 1))
        elif self.members:
            return sum([(x - self.mean) ** 2 for x in self.members])
        else:
            return -1

    def print_report(self):
        print(str(self))
