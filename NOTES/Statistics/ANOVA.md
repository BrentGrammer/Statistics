# ANOVA (Analysis Of Variance)

- Examine patterns of variance in a dataset.
  - Ask how much the variance is due to this or that factor.
- The goal is to determine the effects of discrete independent variables (groups, levels) on a continuous dependent variable
  - Example: you want to know the effects of medication type and age group on how well a medication works. (the independent variables are medication type and age group, the dependent continuous var is how well the medication treats the underlying condition)
- The main statistic we are interested in in an ANOVA is the F-statistic (a ratio of Explained Variance (withingroup) to Unexplained Variance (between-group))
  - i.e. the Variance due to the factors in our data divided by the Natural variation in the data (not attributable to experimental factors)
  - We want the F-statistic to be large: When the F statistic is large it means there is a lot of variability related to experimental manipulations, when small it indicates there is more variability just because of natural causes in the data than what we can account for with independent variables.

### Setting up an ANOVA

#### 1. Review the experiment design and ensure ANOVA is appropriate

Example:

- Research Goal: Test whether Covid-19 medications are effective for different groups
- Experiment: Randomly assign patients to receive medication A,B or placebo. Measure the disease severity after 10 days. Separate older (>50 years old) from younger (50 years old) patients.

#### 2. Identify independent and dependent (a.k.a. outcome) variables

- The dependent variable is the var you are trying to explain variance in
- The independent variables (explanatory variables) are what you hope will explain the dependent variable.
  - They are either variables you manipulate in the experiement or they are naturally occurring.
- In the above example, the Independent Variables are: Medication and age group, and the Dependent Variable is Disease severity after 10 days.

#### 3. Create a table of factors and levels (when possible)

- Factors: The dimensions of the independent variables
  - In the example the factors would be: Medication and the age group
- Levels: Specific groups or manipulations within each factor.
  - In the example the levels would be the medication type (A,B, placebo) and the age level (younger, older)
    - note that we manipulate the medication type level and the age level is naturally ocurring.

| -       | A   | B   | placebo |
| ------- | --- | --- | ------- |
| Younger | x   | x   | x       |
| Older   | x   | x   | x       |

- Note that this is not THE ANOVA TABLE (that is something different, it contains all the details of the results)

#### 4. Compute the model and interpret the resullts.

3 ways to interpret the ANOVA results:

- Main Effect: One factor influences the Dependent Variable even when ignoring all other factors.
  - Example: Young people's symptoms improve faster than older people's symptoms, regardless of the medication type.
- Interactions between Factors: The effect of one factor depends on the levels of another factor.
  - Example: Medication A works better in older people, and medication B works better in younger people. (the effect of the med (one factor level) depends on the level of age group (another factor level))
- Intercept: The average of the Dependent Variable is different from zero. The intercept of the ANOVA
  is usually ignored. (it's reported in the anova table, but usually not relevant)
  - Example: Symptoms improve for almost everyone after 10 days.
  - Another example: if the dependent variable were height, then the intercept of the ANOVA term would be highly significant: it means that people on average are taller than 0 cm.

### When not to use ANOVA

- When using an ANOVA you need categorical factors.
- Example: Testing whether people with more Facebook friends have higher self-reported extraversion.
  - Variables: DV is number of Facebook Friends, IV is scores on a personality questionnaire
  - This is not appropriate for an ANOVA because we do not have any categorical factors
  - Not appropriate for a T-test either since we have no groups to compare
  - Correlation is the correct method to analyze this scenario because we are looking for a linear relationship between two continuous variables.
- Example 2: Test whether RSI (repetitive stress injury) is decreased for a group of meditators vs. non-meditators
  - Variables: IV is the group (meditators or non-meditators), and DV is scores on an RSI index.
  - An ANOVA is possible, but is overkill, because we have only one factor with two levels (a categorical independent variable and a continuous dependent variable )
  - A correlation would not be appropriate because the Independent Variables are discrete groups and not continuous variables.
  - It would be much simpler to run a t-test (we have two groups and a continuous random measure on the dependent variable)

### <number>-way ANOVAs and Other types of ANOVAs

- the <number> is the number of factors
- One Way ANOVA: There is just one factor.
  - Note: does not restrict the number of levels within that factor.
  - Ex: determine the influence of the day of week on iPhone purchases.
    - This has one factor of the day of the week, and seven levels (7 days of the week)
- Two Way ANOVA: has two factors
  - Ex: Determine the influence of day of week AND gender on iPhone purchases. Days have seven levels and gender factor has two levels.
- rmANOVA (Repeated Measures ANOVA): Has at least one factor that involves taking multiple measurements from the same individual.
  - Ex: Determine effects of snack type on mood. The experiment would be to have volunteers eat chocolate for 2 days and potato chips for two days (order is randomized).
    - Note that the same individuals participate in all the levels (eating chocolate and potato chips for two days). The factor is snack type here while the levels are chocolate and potato chips. the DV is mood. Each person in the experiment participates in all the levels of the factor (so we get repeated measures from each individual).
- Balanced ANOVA: Same number of data points in each cell
  - Can used simplified formulas in balanced ANOVAs
  - Ex:

| -       | A   | B   | placebo |
| ------- | --- | --- | ------- |
| Younger | 20  | 20  | 20      |
| Older   | 20  | 20  | 20      |

- Unbalanced ANOVA: Different number of data points across cells. Could happen because of data collection or cleaning.
  - Formulas are different and more complex than balanced ANOVAs in this type.
  - Ex:

| -       | A   | B   | placebo |
| ------- | --- | --- | ------- |
| Younger | 20  | 23  | 21      |
| Older   | 18  | 20  | 20      |

### Dummy-coding variables

- Converting categorical variables into numbers.
  - Best applied to two cases with the numbers 0 or 1.
- Ex: entering gender as a dummy-coded variable: (Male=0, Female=1)
  - Interpretation: The effect of the factor shows the change for the "1" variable compared to the "0" variable ("0" acts like a baseline). The ANOVA would tell us how the dependent variable changes for whatever level is labeled as "1".
  - If studying the effect of gender on lipstick usage, a positive result would indicate the effect of being female is more usage (we cannot claim anything about men using lipstick independently of women).
    - NOTE: the dummy coding effects how we interpret the results. I.e. if we set Females=0 and Males=1, then we would say that the effect is that Males use lipstick less than Women. (same finding, we just change the interpretation based on which dummy code is 0 or 1)

### ANOVA vs. MANOVA

- ANOVA: Only one Dependent Variable (as may Independent Variables as appropriate)
- MANOVA: Multivariate ANOVA - Multiple Dependent Variables (as many Independent Variables as appropriate)
  - Ex: The effects of medication type and age on Covid-19 symptoms and total medical expenses.
    - The 2 dedendent variables are disease symptoms/severity and medical expenses.

### Fixed vs. Random Effects

- Fixed Effects/Fixed Factor: Number of levels in a factor is fixed (ex: home type of dorm, apartment or house, or factor of Operating System: Windows, Mac or Linux)
- Random Effects: The levels of the factor are random or continuous in the population.
  - Ex: age (random factor), salary (random factor), effect of nurses (nurses are individuals and differ in empathy and personality etc., so if the effect is found in nurses, then we know there are important differences in individual nurses, the random factor, that are important)
- Mixed Effects ANOVA: Some factors are fixed and others are random.
  - Ex: Determine whether subjective happiness is influenced by home type (fixed effect) and age (random factor)
  - Ex 2: Determine whether an experimenter (random factor, 2 people giving meds to patients with different random personalities that could influence the effects of the meds) influences the effects of medication (fixed, i.e. two types of medications and a placebo).

### Assumptions needed to use ANOVAs

- ANOVAs are widely used because they are generally robust.
- Independence: The data must be sampled independently of each other in the population to which you want to generalize. Data should be RANDOMLY sampled.
  - i.e. if doing an experiment on effects of medication, you would not want to take patients who were all in the same family and related to or lived with each other.
- Normality: The data must be drawn from a population that is normally (Gaussian) distributed.
  - "The residuals (unexplained variance after fitting the model) are normally/Gaussian distributed."
    - This means that the model is not biased in any way towards some data points against other data points.
- Homogeneity of Variance is assumed:
  - a.k.a. Heteroscedasticity: it is the opposite of homogeneity of variance.
  - The variance in the cells (different levels and different factors) are roughly the same.
  - If some of your cells are much more variable and others are much less variable, that can be a problem for an ANOVA because the way an ANOVA works is dividing the total amount of vairance in the data according to different factors and levels.
    - If you start off with different variants in different parts of the data space that will introduce bias.
- Note that ANOVAs are robust to some violations of these assumptions (normality and homogeneity of variance) as long as they are not too far off or bad.

### Alternatives to ANOVAs (non-parametric ANOVAs)

- If the above assumptions are not met, there are other analyses to use
- NOTE: Very infrequently used and not favored or preferred for analysis. They are difficult and it is prefereble to transform your data into a state where you can use a parametric ANOVA instead.
- Kruskal-Wallis test (KW-ANOVA): Non-parametric alternative to the One-Way ANOVA. Works by looking at rank transformed data instead of the data in the original scale.

### NULL and ALTERNATIVE Hypotheses of an ANOVA:

- The Null Hypothesis ANOVAs test against:
  $$H_0 : {\mu}_1 = {\mu}_2 = ... = {\mu}_k$$
- The NULL Hypothesis in an ANOVA is that all the means in all of the cells (all the groups) are equal to each other or statistically indestinguishable.
  - ${\mu}_1$ is the mean of the data points in cell one, mu2 is the mean of data points in cell 2 and so on (as an ANOVA deals with a table of cells with data of the levels/factors involved). Where $k$ is the number of cells in the table.
- The Alternative Hypothesis is that at least the mean of one group is different from at least one other group mean.
  $$H_A : {\mu}_i \ne {\mu}_j$$
  - This is a generic alternative test. We need further investigation into exactly which means differ after running an ANOVA.

### The Sum of Squares (Math of the ANOVA)

$$SS = \sum_{i=1}^n(x_i - \bar{x})^2$$

- This shows that each individual data point minus the mean squared and then sum all of those up.
- There are different variants, but all are based on this formula.
- This formula is very similar to the variance formula
  - We leave out the $1\over{n-1}$ because we deal with ratios in ANOVAs and the term cancels out.
- **The main idea is to partition the total sum of squared errors (the total variance in the data)**
  - The total variation in the dataset is the sum of variation across individuals within each group AND the variation across the different levels in each factor.
  - IOW, the total variance in the signal is equal to the within group plus the between group sum of squares: $\text{Total SS} = \text{Within-group (a.k.a. Error) SS} + \text{Between-group SS}$
    - Within-group means the individual variability inside each ANOVA cell, each table, each experimental group, and the between-group is how much variance there is between all the members of one group and all members of a different group.
    - Example within-group would mean if you have different people in your group it just means that there is variability between individuals or people (it's not really an "error")
- Different types discussed in [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025142#questions) around the 14 minute mark.

### Types of SS

- Note: the following apply to One Way ANOVAs (One factor with k number of levels). This is just an introduction to the concepts.

#### Sum of Squares Total

$${SS}_{Total} = \sum_{j=1}^{levels}\sum_{i=1}^{individuals}(x_{ij} - \bar{x})^2$$

- We take the variability of all the data across all levels. $\bar{x}$ is the global mean of the entire dataset.

##### Degrees of Freedom for SS Total

${df}_{Total} = N - 1$

- $N$ is the total number of subjects
- We have one mean $\bar{x}$ and therefore one parameter (remember deg of freedom is individuals minus parameters as the general formula for it.)

#### Sum of Squares Between

$${SS}_{Between} = \sum_{j=1}^{levels}(\bar{x}_{j} - \bar{x})^2n_j$$

- Tells us the variability considering all levels (it does not tell us about the variability in relation to any one or specific level) - it tells total variance across all the levels!
- Variability/Variance in dataset between different groups that we can attribute to different levels within each factor (based on the thing we are manipulating in the experiment or the different groups we've organized based on some characteristic)
- We are still dealing with the global mean of the entire dataset with $\bar{x}$, but we are ignoring individuals and looking at the mean within each group or of each level with $\bar{x}_{j}$
- $n_j$ at the end is the number of individuals within each level.
- So again note that we ignore individual variability when using Sum of Squares Between. We are averaging all the individuals together from one cell in the ANOVA table (all the data points within each cell/within each level get averaged together)
- $\bar{x}_{j}$ is the group means (within level) and $\bar{x}$ is the mean of the total population when ignoring all of the factors. We care about the difference between these two metrics.

##### Degrees of Freedom for SS Between

${df}_{Between} = k - 1$

- $k$ is the number of levels that we have in our factor
  - The reason is we are dealing with the averages of the levels (so N turns into k)

#### Sum of Squares Within (a.k.a. SS Error)

$${SS}_{Within} = \sum_{j=1}^{levels}\sum_{i=1}^{individuals}(x_{ij} - \bar{x}_j)^2$$

- Sometimes also called the Sum of Squares Errors (this is just a term, the differences between the data are not literally errors.)
- The main difference here is that we are subtracting the mean within each specific group $\bar{x}_j$
  - The individuals are going to be subtracting from the mean of a group
  - Dealing with individual variability within each group.
- $x_{ij}$ is each individual ($i$) within each cell/level ($j$) and we compute difference against the cell (note same as $\bar{x}_{j}$ in SS Between above)

##### Degrees of Freedom for SS Within

${df}_{Within} = N - k$

- $N$ is the total number of subjects/data points since we are considering all the individuals (within the data and within the level) in SS Within
- We use $k$ because we have $k$ different means in the term $\bar{x}_j$ (the means of the levels where we have k number levels, instead of just 1 parameter with $\bar{x}$ as in SS Total)

### F-Statistic

- The main statistic we are interested in in an ANOVA is the F-statistic (a ratio of Explained Variance (withingroup) to Unexplained Variance (between-group))
  - i.e. the Variance due to the factors in our data divided by the Natural variation in the data (not attributable to experimental factors)
  - We want the F-statistic to be large: When the F statistic is large it means there is a lot of variability related to experimental manipulations, when small it indicates there is more variability just because of natural causes in the data than what we can account for with independent variables.
- See example of interpreting an F-statistic in [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025152#content) at timestamp 9:27
  - A small p-value means the there is a low probability of getting that F-statistic on the F-distribution, so it means that there is an effect that one or more of the levels has.

#### Mean of Squares

Divide the Sum of Squares quanitites by their degrees of freedom:

$${MS_{Between}} = {SS_{Between}\over{df_{Between}}} = {{\sum_{j=1}^{levels}(\bar{x}_{j} - \bar{x})^2n_j}\over{k-1}}$$

- $df_{Between}$ is the degrees of freedom for the Sum of Squares Between formula.

$${MS_{Between}} = {SS_{Between}\over{df_{Between}}} = {{\sum_{j=1}^{levels}\sum_{i=1}^{individuals}(x_{ij} - \bar{x}_j)^2}\over{N-k}}$$

#### F-statistic Ratio:

Ratio of Mean of Squares Between to Mean of Squares Within
$$F_{k-1,N-k} = {MS_{Between}\over MS_{Within}}$$

- Represents a ratio of all the variability we can attribute to experimental factors (the levels within each factor) to the variability we attribute to the individual variability which includes individual differences, natural differences, sampling variability, sampling noise, etc.
- The F-statistic above is with two degrees of freedom ($k-1$ and $N-k$) - the numerator degrees of freedom and the denominator degrees of freedom

### THE ANOVA TABLE

- This is not to be confused with the other tables we create with cells.
- A table that shows all the above computed metrics/information in a formatted way in a table.
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025144?start=30#content) around 4:50 timestamp
    | Source of Variance | Sums of Squares | Degrees of Freedom | Mean square | F | P-value |
    | ------- | --- | --- | ------- | ---- | ---- |
    | Between Groups | SS(b) | k-1 | MS(b) | MS(b)/MS(w) | p
    | Within Groups | SS(w) | N-k | MS(w) | | |
    | Total | SS(t) | N-1 | | | |
- Note that the p value deals with the statistical significance if the mean of at least one group is significantly different from the mean of at least one other group.
- Note that it's possible to derive the total number of data points and the number of levels in a factor from the above table - use the degrees of freedom col - N-1 (N is total data points), N-k (k is number of levels)

#### Correct Interpretation of the p-value in The ANOVA Table:

- **The p-value and F-statistic ratio only tells you that there is a difference between the groups somewhere - it does NOT tell you which groups are different or how many differences there are**
  - It says that the mean of at least one level/group is statistically significantly different from the mean of at least one other group/level.
  - You must do subsequent visualizations and follow up T-tests to determine exactly which groups/levels are causing the differences shown in The Anova Table.
    - i.e. Remember that the SS Between tells us total variance across ALL levels.
  - Example: A p-value of .0002 (because it is less than o.5 or 5%) means that the mean of at least one level/group is statistically significantly different from the mean of one of the other groups.

#### "Omnibus" F-Test

- Refers to the F statistic and in an indicator that there is something in the conclusion that means we should reject the null hypothesis (the means of levels are not all equal and somwhere is a statistically significant difference)
- After seeing that there is some difference, next step is to make visualizations of the data to investigate where the difference is:
  - You could make a bar plot with levels of factor on the x-axis and the dependent variable on the y-axis
  - This is a start, but not conclusive, but note that you cannot just run t-tests pairwise on all the levels (this leads to an unacceptably high family error rate leading to type I errors).
    - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025146?start=15#content) at timestamp 6:05
  - The way to correct for t-test comparisons is to run the Tukey test

### The Tukey Test

- Allows for post-hoc comparisons after a F test reveals there is some difference in an ANOVA in the levels while controlling the familywise error rate
  - Comes into play when wanting to test individual conditions after we have tested that the ANOVA is signficant.
- NOTE: You should only use post-hoc comparisons/tukey test when the omnibus F-test is significant! (otherwise the ANOVA is telling us there is nothing significantly different between levels/groups and there is no reason to compare.)

  - i.e. if the F term shown in The ANOVA Table has a p-value that was greater than .05 or 5%, then there is no significant difference in the levels. iow, there is no evidence of the effect of the factor

#### Formula

$$q = {\bar{x}_b - \bar{x}_s \over {\sqrt{{MS}_{\text{Within}}} \sqrt{2/n}}}$$

- $q$ is a statistic value
- The numerator is the differnce of the means of the two conditions you are comparing (condition $b$ and condition $s$), i.e. the two means of two different groups you're comparing.
- The denominator involves multiplying a variance term by a factor of $n$ (the total number of data points)
- **The tukey test is conceptually similar to a t-test**(we're looking at the difference of means and scaling that by some variance term considering the total number of data points)
  - The way we evaluate the signficance depends on the number of comparisons of the levels we want to make.

##### Q-Distribution

- $q$ has it's own distribution, the Q-Distribution.
  - Evaluated with j,n-j degrees of freedom where j is the total number of comparisons you plan on doing (not the total possible number, for ex. if comparing 4 levels, j = 6 to get all the pairwise comparison permutations). n is the total number of values.

## Two-way ANOVA (A 2 Factor/Factorial ANOVA)

- Dealing with 2 factors
  - Example: Factors are Medication and age, Levels are A,B, placebo (medication) and younger/older (age)
- The total variation in the dataset is the sum of the variation across individuals within each group and the variation across the different levels **within each factor AND the variation at the interaction between the factors**
  - The effect of one fator depends on the levels of the other factor.
    - Example: If the way the medication works differs depending on whether patient is young or old, that means there is an interaction (if the way it worked was the same whether young or old there is no interaction present)
- See formulas updated for a 2-way ANOVA in [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025150#content) at timestamp 2:36

  - Formulas are for:

    - Sum of Squares Total
    - Sum of Squares Between group A (Pool across cells in columns for one factor and across the rows for the other factor, i.e. Medication or Age)
    - Sum of Squares Between group B
    - Sum of Squares for group A x B (variability attributed to factors interacting with each other)
    - Sum of Squares Within (variances within each cell pooled together)

    MEDICATION

| AGE     | A   | B   | placebo |
| ------- | --- | --- | ------- |
| Younger | 20  | 20  | 20      |
| Older   | 20  | 20  | 20      |

### The ANOVA Table for a Factorial ANOVA:

| Source of Variance | Sums of Squares | Degrees of Freedom | Mean square | F             | P-value |
| ------------------ | --------------- | ------------------ | ----------- | ------------- | ------- |
| Factor A           | SS(a)           | a-1                | MS(a)       | MS(a)/MS(w)   | p       |
| Factor B           | SS(b)           | b-1                | MS(b)       | MS(b)/MS(w)   | p       |
| AxB Interact.      | SS(axb)         | (a-1)(b-1)         | MS(axb)     | MS(axb)/MS(w) | p       |
| Within (error)     | SS(w)           | N-ab               | MS(w)       |               |         |
| Total              | SS(t)           | N-1                |             |               |         |

#### Interpreting the p-values

- As shown, we have multiple p-values, but each p-value tells us only about that specific Factor

### Interpreting Main effects and interactions

- Example: The effects of vitamin supplements (vs. placebo) and age(30-40,50-60) on cardiovascular health. (does the supplement improve health and does that depend on the age of participants?)
  - Factor A: Supplement, levels: real vs. placebo
  - Factor B: Age group, levels: 30-40 vs. 50-60

### Main Effects

- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025150#questions) at timestamp 13:11
- Main Effects refer to the effect in a particular factor ignoring the other factor
  - Example: The main effect of Age in the above example means you take the average between the two levels young and old, ignoring the medication type level (placebo vs. supplement) on cardiovascular health.
    - The main effect of Medication factor means you take the average of the Placebo total vs. Supplement totals ignoring the age factor on health.
- Note that when you have a significant interaction, it could mean that the main effects are significant but difficult to interpret - caution is needed when interpreting main effects given a significant interaction and you also have to decide if there is an interaction whether to interpret the main effects.

## Code: One-way ANOVA (including multiple comparisons with Tukey test)
- Use the function from `pingouin` package to compute a one way anova
- [See notebook](../../statsML/anovas/stats_anova_1wayANOVA.ipynb)

## Code: Repeated Measures ANOVA
- [See notebook](../../statsML/anovas/stats_anova_1way_rm_ANOVA.ipynb)