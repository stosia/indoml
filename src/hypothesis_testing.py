import copy
import json
import math
from scipy import stats
import sys

if sys.version_info >= (3, 2):
    raw_input = input


class Sample:
    """Sample is an observation against a particular subject at one
    point in time."""
    def __init__(self, is_population=None):
        self.title = ''
        self.is_population = is_population
        self.mean = None  # The mean
        self.orig_mean = None  # The original mean as was entered by user
        self.sd = None  # standard deviation, may not be known
        self.orig_sd = None  # Original sd as wes input, may not be known
        self.n = None  # Number of samples, may not be known
        self.members = None  # Individual sample, optional

    def __str__(self):
        s = ''
        if True:
            s += "Title:                %s\n" % self.title
            s += "Treat as population:  %s\n" % self.is_population
        if self.n is not None:
            s += "%s:                   % d\n" % ("N" if self.is_population else "n", self.n)
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

    def wizard(self, require_n=False):
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

            s = raw_input('Standard deviation: ').strip()
            if s:
                self.sd = self.orig_sd = float(s)
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

        # Directionality
        self.two_tailed = True
        self.one_tailed_positive = True

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
        self.two_tailed = d.get('two_tailed', True)
        self.one_tailed_positive = d.get('one_tailed_positive', True)

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
            self.samp0.title += " (mean and sd is reset)"
            self.samp1.mean = Sample._calc_mean(members)
            self.samp1.sd = Sample._calc_sd(members, self.samp1.is_population)
            self.samp1.title += " (mean and sd are difference)"
        elif self.samp0.is_population == False and self.samp1.is_population == False and \
             self.samp0.sd is not None and self.samp1.sd is not None:
            # Lesson 7 problem set 10a
            print("Note: ")
            print("   Looks like we have two samples. Assuming these are dependent samples.")
            print("   Thus we will be using the difference instead.")
            self.samp1.sd = self.standard_deviation_difference()
            self.samp1.title += " (sd is difference)"
            self.samp0.sd = 0
            self.samp0.title += " (after sd reset)"
        elif self.samp1.sd is None and self.samp1.is_population == False and \
             self.samp0.is_population and self.samp0.sd is not None:
            print("Note: ")
            print("   SD for second sample is missing. Calculating the SD as SE from population.")
            self.samp1.sd = self.samp0.sd / math.sqrt(self.samp1.n)
            print("   SE: %.3f." % self.samp1.sd)

    def wizard(self):
        """Wizard to input the parameters from console."""
        print("Input the first sample (samp0)")
        self.samp0 = Sample().wizard()

        print("")
        print("Input the second sample (samp1)")
        self.samp1 = Sample(is_population=False).wizard(require_n=True)

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

        s = raw_input('Two tailed (t), one tailed negative (n), or one tailed positive (p) (t/n/p)? [t] ').strip()
        if s:
            s = s.strip().lower()
            assert s in ['t', 'n', 'p']
            self.two_tailed = s == 't'
            self.one_tailed_positive = s == 'p'

        s = raw_input("Alpha: [%.3f] " % self.alpha).strip()
        if s:
            self.alpha = float(s)

        print("critical value: %.3f" % self.critical())
        return self

    def spell_h0(self):
        """Spell the null hypothesis"""
        if self.two_tailed:
            return "%s DOES NOT change %s" % (self.treatment_title, self.results_title)
        else:
            if self.one_tailed_positive:
                return "%s DOES NOT increase %s" % (self.treatment_title, self.results_title)
            else:
                return "%s DOES NOT reduce %s" % (self.treatment_title, self.results_title)

    def spell_hA(self):
        """Spell the alternate hypothesis"""
        if self.two_tailed:
            return "%s DOES change %s" % (self.treatment_title, self.results_title)
        else:
            if self.one_tailed_positive:
                return "%s DOES increase %s" % (self.treatment_title, self.results_title)
            else:
                return "%s DOES reduce %s" % (self.treatment_title, self.results_title)

    def spell_type_of_test(self):
        """Spell the type of t-test"""
        if self.two_tailed:
            return "two tailed"
        else:
            if self.one_tailed_positive:
                return "one tailed in positive direction"
            else:
                return "one tailed in negative direction"

    def mean_difference(self):
        """The mean difference"""
        return self.samp1.mean - self.samp0.mean

    def standard_deviation_difference(self):
        """Standard deviation of the differences"""
        return math.sqrt(self.samp0.sd ** 2 + self.samp1.sd ** 2)

    def fall_in_critical_region(self):
        """Determine if the standard score falls IN the critical
        region"""
        score = self.score()
        crit = self.critical()
        if self.two_tailed:
            # TODO: with equal or not?
            return score <= -crit or score > crit
        else:
            # TODO: with equal or not?
            if self.one_tailed_positive:
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
        print("Sample-0:")
        print("---------")
        print(str(self.samp0))

        print("Sample-1:")
        print("---------")
        print(str(self.samp1))

        print("Various descriptions:")
        print("---------------------")
        print("The dependent variable is: %s" % self.results_title)
        print("The treatment            : %s" % self.treatment_title)
        print("Null hypothesis          : %s" % self.spell_h0())
        print("Alternate hypothesis     : %s" % self.spell_hA())
        print("Type of test             : %s" % self.spell_type_of_test())
        print("")
        print("Parameters:")
        print("-----------")
        print("alpha: %.3f" % self.alpha)
        self.print_extra_params()
        print("")
        print("Results:")
        print("-----------")
        print("mean difference:         % .3f" % self.mean_difference())
        if self.samp0.sd is not None and self.samp1.sd is not None:
            print("SD difference:           % .3f" % self.standard_deviation_difference())
        print("SEM:                     % .3f" % self.SEM())
        print("z/t-critical             % .3f" % self.critical())
        print("score (z-score/t-stat)   % .3f" % self.score())
        print("p-value:                 % .3f" % self.p_value())
        self.print_extra_results()
        print("Margin of error:         % .3f" % self.margin_of_error())
        print("Confidence interval:     % .3f - %.3f" % self.confidence_interval())
        print("Fall in critical region:  %s" % self.fall_in_critical_region())
        print("Is statistically significant: %s" % self.is_statistically_significant())
        print("")
        print("Conclusions:")
        print("------------")
        print(" - %s" % (self.spell_hA() if self.fall_in_critical_region() else self.spell_h0()))
        print(" - %s" % ("The null hypothesis is rejected" if self.fall_in_critical_region() else "Failed to reject the null hypothesis"))
        print("   because %s" % self.spell_reason())
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

    def critical(self):
        """Z-critical value for the specified alpha and test specification"""
        a = self.alpha / 2.0 if self.two_tailed else self.alpha
        z_critical_ = stats.norm.ppf(1 - a)
        if not self.two_tailed and not self.one_tailed_positive:
            z_critical_ = 0 - z_critical_
        return z_critical_

    def margin_of_error(self):
        """Margin of error value"""
        two_tailed_t_critical = stats.norm.ppf(1 - self.alpha / 2)
        return two_tailed_t_critical * self.SEM()

    def p_value(self):
        """The actual probability value given a z-score"""
        z = self.score()
        pval = stats.norm.cdf(abs(z))  # one sided
        if self.two_tailed:
            pval = pval / 2
        return 1 - pval

    def print_extra_conclusions(self):
        print("")
        print("Note:")
        print("-----")
        print(" - Type I error:  H0 is true, but we reject that")
        print(" - Type II error: H1 is true, but we reject that")


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
        a = self.alpha / 2.0 if self.two_tailed else self.alpha
        t_critical_ = stats.t.ppf(1 - a, self.df())
        if not self.two_tailed and not self.one_tailed_positive:
            t_critical_ = 0 - t_critical_
        return t_critical_

    def t_statistics(self):
        """Returns the t-statistic value"""
        return self.mean_difference() / self.SEM()

    def score(self):
        return self.t_statistics()

    def margin_of_error(self):
        two_tailed_t_critical = stats.t.ppf(1 - self.alpha / 2, self.df())
        return two_tailed_t_critical * self.SEM()

    def p_value(self):
        """The actual probability value"""
        t = self.t_statistics()
        pval = stats.t.sf(abs(t), self.df())  # one sided
        if self.two_tailed:
            return pval / 2
        else:
            return pval

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
        samp.wizard()
        print(str(samp))
        sys.exit(0)

    if not t:
        sys.stderr.write("Error: -t or -z must be specified\n\n")
        usage()
        sys.exit(1)

    if input_file:
        with open(input_file) as f:
            body = f.read()
            d = json.loads(body)
        t.load_from_dict(d)
    else:
        t.wizard()

    if output_file:
        with open(output_file, 'wt') as f:
            f.write(json.dumps(t.__dict__, indent=4, cls=MyEncoder, sort_keys=True))

    print('=' * 50)
    t.print_report()
