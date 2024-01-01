# ANOVA (Analysis Of Variance)

- Examine patterns of variance in a dataset.
  - Ask how much the variance is due to this or that factor.
- The goal is to determine the effects of discrete independent variables (groups, levels) on a continuous dependent variable
  - Example: you want to know the effects of medication type and age group on how well a medication works. (the independent variables are medication type and age group, the dependent continuous var is how well the medication treats the underlying condition)

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
