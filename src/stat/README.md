These files are small codes that I wrote while taking Udacity's 
[Intro to Inferential Statistics](https://www.udacity.com/course/intro-to-inferential-statistics--ud201). 
Answers to quizes and problem sets are also saved.

I also wrote [review for that course](https://indoml.com/2017/09/09/mooc-review-intro-to-inferential-statistics-udacity/) 
in Indoml.com blog. It's in Bahasa Indonesia though.

# Summary of Problems to be Solved by Inferential Statistics Taught in This Course

* Measuring if a sample is significantly different
  * than the population
    * you have the population parameters (mean, sd)
      * use Z-test
    * you only have population mean and not sd
      * use T-test
  * than the same sample
      * this is called "within subject design"
      * use dependent T-test
      * examples:
        * two conditions
        * longitudinal (same test done after some time)
        * pretest posttest
  * than anoother sample
      * use independent T-test
      * for example:
        * average meal price between two areas
  * you don't know the parameters (non-parametric)
      * use Chi square goodness-of-fit test
      * for example when the sample is just number of yes and no answers
* Determining if at least one group is significantly different than the others if we have three or more groups
  * you just want to know that
    * use ANOVA (F-test)
  * you want to know which one
    * use ANOVA and Tukey's HSD
* Measuring the relationship between two variable
  * parameteric
    * does height have relation to weight?
      * calculate the correlation
    * what's the relation?
      * use linear regression
  * non-parametric
    * use Chi-square independent test
    * for example, does sex have something to do with being admitted to college
