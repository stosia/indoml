import copy
import json
import math
import sys

from scipy import stats


if sys.version_info >= (3, 2):
    raw_input = input


class StatTool:
    """Various statistic tools"""
    TWO_TAILED_TEST = 't'
    ONE_TAILED_NEGATIVE_TEST = 'n'
    ONE_TAILED_POSITIVE_TEST = 'p'

    @classmethod
    def spell_directionality(cls, dir):
        """Spell directionality of the test"""
        assert dir in [cls.TWO_TAILED_TEST, cls.ONE_TAILED_NEGATIVE_TEST, cls.ONE_TAILED_POSITIVE_TEST]

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
        assert dir in [cls.TWO_TAILED_TEST, cls.ONE_TAILED_NEGATIVE_TEST, cls.ONE_TAILED_POSITIVE_TEST]

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
        
        For positive t_statistic, this returns the probability that any samples
        will have t-stat AT LEAST this value.
        
        For negative t_statistic, this returns the probability that any samples
        will have t-stat LESS THAN this value.
        """
        # pval is the probability that any samples will have
        # EQUAL OR MORE THAN abs(t_statistic).
        #
        # FWIW sf = survival function
        pval = stats.t.sf(abs(t_statistic), df)
        return pval

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


class Sample:
    """Sample is an observation against a particular subject at one point in time."""
    def __init__(self, is_population=None):
        self.title = ''
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
            s += "Title:                %s\n" % self.title
            s += "Notes:                %s\n" % self.notes
            s += "Treat as population:  %s\n" % self.is_population
        if self.n is not None:
            s += "%s:                   % d\n" % ("N" if self.is_population else "n", self.n)
        if self.members:
            s += "Members:              %s\n" % (str(self.members))
        if True:
            s += "Mean:                % .3f\n" % (self.mean)
            if self.orig_mean is not None:
                s += "Orig. Mean:          % .3f\n" % (self.orig_mean)
            else:
                s += "Orig. Mean:           None\n"
        if self.sd is not None:
            s += "SD:                  % .3f\n" % (self.sd)
            if self.orig_sd is not None:
                s += "Orig. SD:            % .3f\n" % (self.orig_sd)
            else:
                s += "Orig. SD:             None\n"
        return s

    def load_from_dict(self, d):
        self.title = d.get('title', '')
        self.notes = d.get('notes', '')
        self.is_population = d.get('is_population', None)
        self.mean = d.get('mean', None)
        self.orig_mean = d.get('orig_mean', None)
        self.sd = d.get('sd', None)
        self.orig_sd = d.get('orig_sd', None)
        self.n = d.get('n', None)
        self.members = d.get('members', None)

    @staticmethod
    def _calc_mean(members):
        return sum(members) / float(len(members))

    @staticmethod
    def _calc_sd(members, is_population):
        n = len(members)
        mean = Sample._calc_mean(members)
        squared_diffs = [(m - mean) ** 2 for m in members]
        sum_squared_diff = sum(squared_diffs)
        if is_population:
            return math.sqrt(sum_squared_diff / float(n))
        else:
            return math.sqrt(sum_squared_diff / float(n - 1))

    def input_samples(self, s=None):
        """Parse the samples if given in s, or input from console"""
        while not s:
            s = raw_input('Individual samples (space or comma separated): ').strip()

        if ',' in s:
            self.members = s.split(',')
        else:
            self.members = s.split(' ')
        self.members = [m.strip() for m in self.members]
        self.members = [float(m) for m in self.members if m]
        self._update_parameters()

    def _update_parameters(self):
        """Update mean sd etc after we have samples"""
        self.n = len(self.members)
        self.mean = self.orig_mean = Sample._calc_mean(self.members)
        self.sd = self.orig_sd = Sample._calc_sd(self.members, self.is_population)
        print("Got %d samples, mean: %.3f, sd: %.3f" % (self.n, self.mean, self.sd))

    def input_wizard(self, require_n=False, ref_pop=None):
        """Wizard to input the parameters from console."""
        s = raw_input('The name of this sample? [default] ').strip()
        if s and s != 'default':
            self.title = s

        if self.is_population is None:
            s = ''
            while s not in ['y', 'n']:
                s = raw_input('Treat as population (y/n)? [n] ').strip()
                s = s.lower()
                if not s:
                    s = 'n'
            self.is_population = True if s == 'y' else False

        if not self.title:
            self.title = "population" if self.is_population else "sample"

        s = ''
        while s not in ['p', 'i']:
            s = raw_input('Input parameters or individual sample (p/i)? [p] ').strip()
            if not s:
                s = 'p'

        if s == 'p':
            while True:
                s = raw_input("n (number of data): ").strip()
                if s:
                    self.n = int(s)
                if self.n is not None or not require_n:
                    break

            while self.mean is None:
                s = raw_input('Mean: ').strip()
                if s:
                    self.mean = self.orig_mean = float(s)

            if ref_pop and (not ref_pop.is_population or ref_pop.sd is None):
                ref_pop = None

            s = raw_input('Standard deviation%s: ' % (' (skip to calculate from population)' if ref_pop else '')).strip()
            if s:
                self.sd = self.orig_sd = float(s)
            elif ref_pop:
                self.sd = ref_pop.sd / math.sqrt(self.n)
                self.notes = "SD is derived from population"
                print("Note: Calculating standard deviation as Standard Error from population.")
                print("      SE: %.3f." % self.sd)
        else:
            self.input_samples()

        return self


class HypothesisTesting:
    """With a hypothesis testing, we are finding out if some results happen because
    of a specific cause, rather than just random chance or sampling error.
    """
    def __init__(self):
        self.samp0 = None
        self.samp1 = None

        # Parameters
        self.alpha = 0.05
        self.dir = None  # see StatTool.xxx_TEST constants

        # Description
        self.treatment_title = "treatment"
        self.results_title = "results"

    def load_from_dict(self, d):
        if d.get('samp0'):
            self.samp0 = Sample()
            self.samp0.load_from_dict(d.get('samp0'))

        if d.get('samp1'):
            self.samp1 = Sample()
            self.samp1.load_from_dict(d.get('samp1'))

        # Parameters
        self.alpha = d.get('alpha', 0.05)
        self.dir = d.get('dir', None)

        # Description
        self.treatment_title = d.get('treatment_title', "treatment")
        self.results_title = d.get('results_title', "results")

        self._fix_samples()

    def _fix_samples(self):
        """Decide if the two samples are dependent and we should do something
        about it"""
        if self.samp0.is_population == False and self.samp1.is_population == False and \
           self.samp0.members and self.samp1.members:
            print("Note: ")
            print("   Looks like we have two samples. Assuming these are dependent samples.")
            print("   Thus we will be using the difference instead.")

            members = [self.samp1.members[i] - self.samp0.members[i] for i in range(len(self.samp1.members))]
            self.samp0.mean = 0.0
            self.samp0.sd = None
            self.samp0.notes = "mean and sd have been reset"
            self.samp1.mean = Sample._calc_mean(members)
            self.samp1.sd = Sample._calc_sd(members, self.samp1.is_population)
            self.samp1.notes = "mean and sd are difference from sample-0"
        elif self.samp0.is_population == False and self.samp1.is_population == False and \
             self.samp0.orig_sd is not None and self.samp1.orig_sd is not None:
            # Lesson 7 problem set 10a
            print("Note: ")
            print("   Looks like we have two samples. Assuming these are dependent samples.")
            print("   Thus we will be using the difference instead.")
            self.samp1.sd = self.standard_deviation_difference()
            self.samp1.notes = "sd is difference from sample-0"
            # self.samp0.sd = 0
            # self.samp0.notes = "sd has been reset"


    def input_wizard(self):
        """Wizard to input the parameters from console."""
        print("Input the first sample (samp0)")
        self.samp0 = Sample().input_wizard()

        print("")
        print("Input the second sample (samp1)")
        self.samp1 = Sample(is_population=False).input_wizard(require_n=True,
                                                              ref_pop=self.samp0 if self.samp0.is_population else None)

        print("Mean difference: %.3f" % self.mean_difference())

        s = ''
        self._fix_samples()

        print("")
        s = raw_input("The name of the treatment ('%s'): " % self.treatment_title).strip()
        if s:
            self.treatment_title = s

        s = raw_input("The name of the results ('%s'): " % self.results_title).strip()
        if s:
            self.results_title = s

        dirs = [StatTool.TWO_TAILED_TEST, StatTool.ONE_TAILED_NEGATIVE_TEST, StatTool.ONE_TAILED_POSITIVE_TEST]
        self.dir = ''
        while self.dir not in dirs:
            self.dir = raw_input('Directionality: Two tailed (t), one tailed negative (n), or one tailed positive (p) (t/n/p)? [t] ').strip()

        s = raw_input("Alpha: [%.3f] " % self.alpha).strip()
        if s:
            self.alpha = float(s)

        print("Critical value: %.3f" % self.critical())
        return self

    def spell_h0(self):
        """Spell the null hypothesis"""
        if self.dir == StatTool.TWO_TAILED_TEST:
            return "\"%s\" DOES NOT change \"%s\"" % (self.treatment_title, self.results_title)
        elif self.dir == StatTool.ONE_TAILED_POSITIVE_TEST:
            return "\"%s\" DOES NOT increase \"%s\"" % (self.treatment_title, self.results_title)
        else:
            return "\"%s\" DOES NOT reduce \"%s\"" % (self.treatment_title, self.results_title)

    def spell_hA(self):
        """Spell the alternate hypothesis"""
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
        """The mean difference"""
        return self.samp1.mean - self.samp0.mean

    def standard_deviation_difference(self):
        """Standard deviation of the differences"""
        return math.sqrt(self.samp0.orig_sd ** 2 + self.samp1.orig_sd ** 2)

    def fall_in_critical_region(self):
        """Determine if the standard score falls IN the critical
        region"""
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
        """Determine if the results are not likely due to chance."""
        return self.fall_in_critical_region()

    def is_meaningful(self):
        """Is it meaningful?"""
        return "it depends"

    def cohens_d(self):
        """Returns Cohen's D"""
        return self.mean_difference() / self.samp1.sd

    def confidence_interval(self):
        """Return confidence interval (CI) as sequence of (low, hi). CI
        means that we are (1-alpha) percent sure that the sample values will fall
        in this interval"""
        merr = self.margin_of_error()
        return (self.samp1.mean - merr, self.samp1.mean + merr)

    def spell_reason(self):
        """Spell the reason why we reject/fail to reject the null hypothesis"""
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
        print("Sample-0: %s" % self.samp0.title)
        print("-" * 70)
        print(str(self.samp0))

        print("Sample-1: %s" % self.samp1.title)
        print("-" * 70)
        print(str(self.samp1))

        print("Various descriptions:")
        print("-" * 70)
        print("The dependent variable is: %s" % self.results_title)
        print("The treatment            : %s" % self.treatment_title)
        print("Null hypothesis          : %s" % self.spell_h0())
        print("Alternate hypothesis     : %s" % self.spell_hA())
        print("Type of test             : %s" % self.spell_type_of_test())
        print("")
        print("Parameters:")
        print("-" * 70)
        print("alpha: %.3f" % self.alpha)
        self.print_extra_params()
        print("")
        print("Results:")
        print("-" * 70)
        print("mean difference:         % .3f" % self.mean_difference())
        if self.samp0.orig_sd is not None and self.samp1.orig_sd is not None:
            print("SD difference:           % .3f" % self.standard_deviation_difference())
        print("SEM:                     % .3f" % self.SEM())
        print("%-15s          % .3f" % (self.critical_title(), self.critical()))
        print("%-15s          % .3f" % (self.score_title(), self.score()))
        print("p-value:                 % .3f" % self.p_value())
        self.print_extra_results()
        print("Margin of error:         % .3f" % self.margin_of_error())
        print("Confidence interval:     % .3f - %.3f" % self.confidence_interval())
        print("Fall in critical region:  %s" % self.fall_in_critical_region())
        print("Is statistically significant: %s" % self.is_statistically_significant())
        print("")
        print("Conclusions:")
        print("-" * 70)
        print(" - %s" % (self.spell_hA() if self.fall_in_critical_region() else self.spell_h0()))
        sys.stdout.write(" - %s" % ("The null hypothesis is rejected" if self.fall_in_critical_region() else "Failed to reject the null hypothesis"))
        print(" because %s" % self.spell_reason())
        self.print_extra_conclusions()


class ZTesting(HypothesisTesting):
    """This hypothesis testing is used when we have the population parameters,
    especially the standard deviation. Then given a sample, we would like to
    know if the results happens due to certain cause and not just random chance
    or sampling errors."""
    def __init__(self):
        HypothesisTesting.__init__(self)

    def SEM(self):
        """The standard error"""
        return self.samp0.sd / math.sqrt(self.samp1.n)

    def score(self):
        """The Z-score"""
        return self.mean_difference() / self.SEM()

    def score_title(self):
        """The name for the score"""
        return "Z-score"

    def critical(self):
        """Z-critical value for the specified alpha and test specification"""
        return StatTool.z_critical_value(self.alpha, self.dir)

    def critical_title(self):
        """The name for the critical value"""
        return "z-critical"

    def margin_of_error(self):
        """Margin of error value"""
        z_critical2 = StatTool.z_critical_value(self.alpha, StatTool.TWO_TAILED_TEST)
        return z_critical2 * self.SEM()

    def p_value(self):
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
        HypothesisTesting.__init__(self)

    def df(self):
        """Degrees of freedom"""
        return self.samp1.n - 1

    def SEM(self):
        """Standard error of mean"""
        return self.samp1.sd / math.sqrt(self.samp1.n)

    def critical(self):
        """t-critical value for the specified confidence/alpha.
        Value may be negative!"""
        return StatTool.t_critical_value(self.alpha, self.dir, self.df())

    def t_statistics(self):
        """Returns the t-statistic value"""
        return self.mean_difference() / self.SEM()

    def score(self):
        return self.t_statistics()

    def score_title(self):
        """The name for the score"""
        return "t-statistics"

    def critical_title(self):
        """The name for the critical value"""
        return "t-critical"

    def margin_of_error(self):
        """Get the margin of error value."""
        # Margin of error always uses two tailed
        t_critical2 = StatTool.t_critical_value(self.alpha,
                                                StatTool.TWO_TAILED_TEST,
                                                self.df())
        return t_critical2 * self.SEM()

    def p_value(self):
        """The actual probability value"""
        return StatTool.probability_for_t(self.t_statistics(), self.dir, self.df())

    def r_squared(self):
        t = self.t_statistics()
        return t ** 2 / (t ** 2 + self.df())

    def print_extra_params(self):
        print("df:    %d" % self.df())

    def print_extra_results(self):
        print("Cohen's d                % .3f" % self.cohens_d())
        print("r^2:                     % .3f" % self.r_squared())



if __name__ == "__main__":
    def usage():
        print("Usage:")
        print("  hypothesis_testing.py -z|-t [-i filename] [-o filename]")
        print("")
        print("  -s           Input a sample and get its parameters")
        print("  -z           Z Testing")
        print("  -t           T Testing")
        print("  -i filename  Read parameters from file")
        print("  -o filename  Write parameters to file")

    class MyEncoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__

    t = input_file = output_file = samp = None
    args = ""
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-t":
            t = TTesting()
        elif arg == "-z":
            t = ZTesting()
        elif arg in ["-h", "--help", "/?"]:
            usage()
            sys.exit(0)
        elif arg == "-i":
            i += 1
            input_file = sys.argv[i]
        elif arg == "-o":
            i += 1
            output_file = sys.argv[i]
        elif arg == '-s':
            samp = Sample()
        else:
            args += " " + arg
        i += 1

    if samp:
        samp.input_wizard()
        print(str(samp))
        sys.exit(0)

    if not t and not input_file:
        sys.stderr.write("Error: -t or -z must be specified\n\n")
        usage()
        sys.exit(1)

    if input_file:
        if t:
            sys.stderr.write("Error: -i cannot be used with -z nor -t")
            sys.exit(1)

        with open(input_file) as f:
            body = f.read()
            d = json.loads(body)

        class_name = d['class']
        if class_name == 'ZTesting':
            t = ZTesting()
        elif class_name == "TTesting":
            t = TTesting()
        else:
            sys.stderr.write("Error: invalid class %s\n\n" % class_name)
            sys.exit(1)
        t.load_from_dict(d)
    else:
        t.input_wizard()
        print("End of input wizard")
        print("")

    if output_file:
        with open(output_file, 'wt') as f:
            d = copy.copy(t.__dict__)
            d['class'] = t.__class__.__name__
            f.write(json.dumps(d, indent=4, cls=MyEncoder, sort_keys=True))

    print("REPORT:")
    print('=' * 70)
    t.print_report()
