# DATA

- Singular: Datum (a single point)
- Plural: Data
- To reduce complexity of the universe into something we can try to understand, physical and biological data can be reduced to numbers that can be stored on a computer.

### Limitations of Data

- Be careful about whether the data is actually measuring what you think it is
  - some data is difficult to measure (ex: Covid cases - some people had minor symptoms and did not report - so finding the number of total covid cases was impossible and the number represented only reported cases.)
- **Data does NOT come from the universe, they come from our MEASUREMENT DEVICES**

  - if the devices measurements are not accurate, noisy, imprecise, biased or flawed then our data will have those problems as well.

## DATA TYPES IN STATISTICS

- Data can be divided into categories and types - this matters because certain statistical procedures and analyses are applied depending on the type of data.

### 3 Categories:

#### NUMERICAL DATA:

- **Interval**: Numeric scale with meaningful intervals and arbitrary precision (1.2112,1.2322), and zero does not mean an absense of or is not meaningful in that way. The relationship between datum is not constant necessarily either.
  - Ex: Temperature in Celsius
  - Ex: IQ - an IQ of zero does not mean absence of intelligence and an IQ of 100 is not twice as smart as an IQ of 50
  - Note that 0 degrees does not mean the absence of temperature (should switch to ratio scale)
- **Ratio**: similar to an interval, precision can be less than 1 (decimals, 1.5 etc.), but has a meaningful zero and consistent relationship between values (you do not have this in interval type data)
  - Arbitrary precision - the more precise your ruler, the more precise the measurement is. i.e. not discrete
  - Ex: Height in centimeters (zero cm represents an absense of height)
  - Money: 0 money represents an absence of wealth
  - Ex: Land area of forests - you can have zero land area, no forests
- **Discrete**: No arbitrary precision (integers only)
  - Ex: Population (you can't have 2.5 people, either 1 or 2 or 3 people etc.)
  - Ex: number of widgets purchased in Jan. 2019

#### CATEGORICAL DATA (LABELED DATA):

Data that do not have intervals or important numerical props as above

- **Ordinal**: Sortable and discrete data
  - Key difference between ordinal and discrete (numerical) is that the differences are not meaningful or consistent (difference between a Bachelor's and a Masters is not consistent with difference between Bachelor's and PHD for ex.)
  - Ex: Level of Education (elementary school, junior high, high school, university, etc.)
  - Ex: Favorite 3 countries (Vietnam, Fiji, New Zealand sorted in order of favoriteness)
  - Ex: Cities sorted by population (note that if you wanted the populations themselves then that would be numerical discrete data. In this case the ratio of populations between the countries is not constant)
  - Ex: Satisfaction with this course so far (rating 1-5) - the difference of rating between a 1 and 2 is not the same thing as rating between a 4 and a 5
- **Nominal**: Non-sortable, but discrete
  - Ex: Movie Genres (sci-fi, romantic comedy, etc.) - you can't sort these meaningfully (alpha sorting does not count)
  - Ex: Last three countries you visited
  - Ex: Gender (male, female)

Note: the type of data can change depending on the question asked. For ex:

- last 3 countries visited can either be Nominal data (unsortable but discrete) or Ordinal (sortable) if the date is included/asked for (you can sort by date).

### SAMPLE vs. POPULATION DATA

Many statistical procedures are designed for population or sample data.
Most data you work with in practice is sample data. (it's usually not feasible to measure every single point of data in large sets)

- **Population**: Data from all, every single, members of a group
  - Ex: Salaries of all people in a department (lower finite number - feasible to get all of them), # Lions in a Zoo, all group members in a Facebook group
- **Sample**: Data from some, but not every single, member of the group. **Should be randomly selected data**.
  - Samples are necessary when it is not feasible or easy to measure the entire population.
  - Ex: average height of all Italians, Salaries of all teachers, Brightness of stars in the Milky Way

#### Data Parameter Notation:

- Population Parameters:
  - mean: $\mu $
  - Beta for regression coefficient: $\beta$
  - Sigma Squared for variance: $\sigma^2$
- Sample Parameters (place a hat over the symbol):
  - mean: $\hat{\mu}$ or $\bar{\mu}$
  - regression coefficient: $\hat{\beta}$ or $\beta$
  - variance: $\hat \sigma^2$ or $s^2$

### Samples, Case Reports and Anecdotes

#### N = 1 Studies

- Interesting stories that may inspire future research, but that more often should not be trusted or used as a generalized finding and **not trusted or weighted too much**.
  - Statistical inference is difficult or impossible, why is it being reported?
  - highly likely to be noise or outlier and non-representative data
  - reported on things like the news - something unusual, something exceptional and not representative of the norm.
- Case Study: one Patient
  - rare treatment that worked for one individual or person with rare disease etc.
- Anecdote: One person's experience

#### N > 1 Studies

- N could be 2 or more than 2 (like a million etc.).
- Actual research worthy of important and generalizable findings.
- Pilot Studies: sample of a small subset of individuals - to test that the experiment is working well
- Proof of Principle: a little larger than a pilot study, gather just enough data to see if your experiment is on to something
- Small-scale Studies: small N number studies (could include Pilot or Proof of principle studies)
- Large-scale Studies: large N number studies (can be really large and consist of a very large quantity of data) 


### Simulating Data
- Allows for validating analysis methods
- Learn pros and cons of different analysis methods
- **Always tell the audience that data is simulated or faked if you are working with simulated data**
