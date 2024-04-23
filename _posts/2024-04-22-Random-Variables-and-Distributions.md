# Random Variables [R.V]


$$ \text{Random variable} \, \mathbb{X}: \Omega \rightarrow \mathbb{R}$$

##### Definition:

Random Variable (X) on a sample space is a function that assigns each sample point $\omega \isin \Omega$ to a real number X($\omega$).

X := number of heads in n-coin tosses.

X is random variable.

(There should be a bound for random variables, for eg., for n coin tosses, the possible values are 0, 1, ..., n)


--- 

* *X axis* = random variables.

* *Y axis* = $\mathbb{P}(x_i)$ = distribution of ($x_i$)

* *Expectation, E* = center of mass of distribution graph / average value of X axis. 

---

#### Example 1:

![alt text](image-8.png)

1. Define $X_i$ = number of i fix points.
2. Get **Distribution**
    * $\mathbb{P}(X_0 = \frac{2}{6}) $
   * $\mathbb{P}(X_1 = \frac{3}{6}) $
   * $\mathbb{P}(X_3 = \frac{1}{6}) $
3. Note that $\mathbb{P}(X_0) + \mathbb{P}(X_1) + \mathbb{P}(X3) = 1$ 
> $\mathbb{P}(X_2)$ is not possible/ $\mathbb{P}(X_2) = 0$. If two letters out of three letters are in their correct place $\rightarrow $third letter must also be in correct place.

---

## Bernoulli Distribution

![alt text](image-10.png)

> Bernoulli distribution is a discrete distribution in which the **random variable** has only **two possible outcomes** and **one trial**

> The Bernoulli distribution calculates the probability of each outcome of an event with two possible outcomes. It's a model for the possible outcomes of a single experiment that asks a yes-no question, where the probability of success is p and the probability of failure is 1 - p. 

> Bernoulli distribution accesses only one trial.

* the random variable must be either 0 or 1.
* $\mathbb{P}(X = 1) = p$
* $\mathbb{P}(X = 0) = 1 - p$


$$ X \sim Bernoulli(p)$$

<small> A random variable with Bernoulli distribution is called Bernoulli random variable </small>

---

## Binomial Distribution [independent Bernoulli]

![alt text](image-11.png)

> It calculates the likelihood of a certain number of successes in a set number of trials

$$ \mathbb{P}(X = i) = \binom{n}{i} p^{i} (1- p)^{n - i} $$

* Think about a set of number of tosses of coins. For example, toss three coins.
* if $\mathbb{P}(head) = p$ Probability of getting 2 heads in the sequence will be $\binom{3}{2} p^2 \cdot (1-p)^{1}$



> **Bernoulli deals with the outcome of a single trial of an event, while Binomial deals with the outcome of multiple trials of the same event.**

$$ X \sim Binomial(n, p) $$

n = number of trials
p = probability of success

<small> A random variable with Binomial distribution is called Binomial random variable </small>

---


### Hypergeometric Distribution

So, in *Binomial Distribution*, the events were independent and without replacement. In other words, all sequence of events have the same probability.

What if the events are dependent? For example, in the case of choosing balls into bins without replacement, what's the distribution gonna be?

> Note that if the balls are replaced, the distribution will just be Binomial Distribution.

Example:
N = B(lack) + W(hite) "balls"
X := number of black balls in a sample.
What is the probability distribution of X?

**Replacement Case**
+ probability of getting black ball on 1st choice = $\frac{B}{N}$
+ probability of getting black ball on 2nd choice = $\frac{B}{N}$
+ ...
Therefore, if *k := number of black balls and n := number of balls picked for a sample*

$$ \mathbb{P}(X = k) = \binom{N}{k} (\frac{B}{N})^{k}(\frac{W}{N})^{n - k}$$

Another interpretation would be:

$$ X \sim \text{Binomial } (N, p=\frac{B}{N}) $$

**Without Replacement Case** [Poker Hand]
+ probability of getting black ball on 1st choice = $\frac{B}{N}$
+ probability of getting black ball on 2nd choice = $\frac{B - 1}{N - 1}$


$$ \mathbb{P}(X = k) = \frac{|E_k|}{|\Omega|} $$

$$ |\Omega| = \binom{N}{n}$$

$$ |E_k| = \text{cardinality of events where there are k black balls and n-k white balls}$$

$$ |E_k| =  \binom{B}{k} \cdot \binom{N - B}{n - k}$$

Therefore, probability distribution of X:

$$ \mathbb{P}(X = k) = \frac{\binom{B}{k} \cdot \binom{N - B}{n - k}}{\binom{N}{n}} $$

Another Interpretation would be:

$$ X \sim \text{Hypergeometric } (N, B, n) $$

---

### Multiple Random Variables and Independence

##### Joint Distributions

Assume we have two random variables X and Y. Then the joint distribution is:

$$ {a, b, \mathbb{P}[X = a,(and) Y = b]) ; a \isin A, b \isin B} $$


If the **Random Variables** are **independent**:

$$ \mathbb{P}[X = a, Y = b] = \mathbb{P}[X] \times \mathbb{P}[Y]$$

**Marginal Distribution** is given by:

$$ \mathbb{P}[X] = \sum_{b\isin B} \mathbb{P}[X = a, Y = b \,] $$

---

#### Examples on Joint Distributions

Random Variables:
```
X = score on first die
Y = score on second die
Z = summation of the scores, Z = X + Y

```

$$ \mathbb{P}[X = 3, Y = 5] = \frac{1}{36} $$

$$ \mathbb{P}[X = 3, Z = 9] = \frac{1}{36} [\text{Y must be 6}]$$

> X and Y are independent random variables.
> X and Z are **not** independent random variables.


---

### Expectation [Typical Value of Random Variable]

Expectation is a summarized / compact version of distribution and is easier to compute!

Expectation is also the center of mass / mean / average of the Random Variable X.

As the value of N increases, the average gets closer and closer to our expectation value.

$$ \mathbb{E}[X] = \sum_{a\isin A} a \times \mathbb{P}[X = a] $$

*A = set of possible values of X.*

---

### Examples on Expectation

####  1. Single Die.

X := score that comes up after rolling the dice.
X = {1, 2, 3, 4, 5, 6}

What's  $\mathbb{E}[X]$?

$$ \mathbb{P}[X = 1 or ... 6]  = 1/6$$

Therefore:

$$ \mathbb{E}[X] = (1 \times \frac{1}{6}) + (2 \times \frac{1}{6}) + (3 \times \frac{1}{6}) + (4 \times \frac{1}{6})+ (5 \times \frac{1}{6})+ (6 \times \frac{1}{6}) $$

$$ \mathbb{E}[X] = 3.5 $$

---

#### 2. Two Die
X := sum of the two scores.
X = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

$$ \mathbb{P}[X = 2] = \frac{1}{36} $$
$$ \mathbb{P}[X = 3] = \frac{2}{36} $$
$$ \mathbb{P}[X = 4] = \frac{3}{36} $$
$$ ...$$

Therefore,

$$ \mathbb{E}[X] = (2 * \frac{1}{36}) + (3 * \frac{2}{36}) + ...  $$
$$ \mathbb{E}[X] = 7 $$

---

#### 3. Roulette.
38 slots: 36 unique numbers + 0 + 00
18 blacks and 18 reds

*X := net winnings of one game*
*X = {-1, +1}*

Find the expected value of X.

$$ \mathbb{P}[X = -1] = \frac{20}{38} \\ \, \\ \mathbb{P}[X = +1] = \frac{18}{38}  $$

<br>

$$ \text{Therefore, Expectation,  } \mathbb{E}[X] = (1 \times \frac{18}{38}) + (-1 \times \frac{20}{38})\\ \, \\ \mathbb{E}[X] = -0.0526 \\ \\ \text{This basically means that you will lose 5 cents for every dollar you play on average}$$

---


## Linearity of Expectation

**Linearity of Expectation holds true for both independent and dependent values.**

Property 1:
$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$$

Property 2:
$$\mathbb{E}[cX] = c\mathbb{E}[X]$$ 

---

### Examples on Linearity of Expectation

#### 1. Two Die
X := sum of the scores on two dices.
$\mathbb{E}[X]$ = Expectation of dice 1's score + Expectation of dice 2's score.

Therefore, 

$$ \mathbb{E}[X] = \frac{7}{2} + \frac{7}{2} = 7 $$

![alt text](image-12.png)

---


#### 2. Rounds of Roulette.

$X_n$ := expected winning after playing n times.

$\therefore X_n$ = sum of each game net expectation 

let $Y_i$ = net winning of $i^{th}$ game.

$$ \mathbb{E}[X_n] =  \mathbb{E}[Y_1] + \mathbb{E}[Y_2] + ... + \mathbb{E}[Y_n]$$


$$ \mathbb{E}[X_n] =  \frac{-1}{19} + ... + \frac{-1}{19}$$

$$ \mathbb{E}[X_n] =  \frac{-n}{19}$$

---

### :exclamation: Methods of Indicators [Important]

when do you use indicator random variable?

+ The main idea lies in the observation that the number of **good** results can be counted by first encoding each **good** as **1** and **bad** as **0** [$\mathbb{I}$].

+ If N = number of total **good** results in *n trials*.

+ $ N = I_1 + I_2 + ... + I_n, where \, I \isin {(1, 0)}$

Therefore:

$$ \mathbb{E}[N] = \mathbb{E}[I_1] + \mathbb{E}[I_2] + ... + \mathbb{E}[I_n] $$

> :exclamation: :warning: The additivity works regardless of whether the trials are dependent or independent.

> :exclamation: :warning: Binomial Random Variable can be thought of as the sum of Bernoulli Random Variables.

---

### Expectation of the Binomial

X = binomial variable
I's = bernoulli variables

$$ X = I_1 + I_2 + ... + I_n, where \, I \isin {(1, 0)}$$

$$ \mathbb{E}[N] = \mathbb{E}[I_1] + \mathbb{E}[I_2] + ... + \mathbb{E}[I_n] $$

$$ \text{Since I's are bernoulli variables, } \mathbb{E}[I's] = \mathbb{P}[I_j = 1] = p $$

$$ \mathbb{E}[N] = p + p + ... + p $$

$$ \mathbb{E}[N] = n \cdot p$$

Applications of Expectation of Binomial Distribution

1. the expected number of heads in 100 tosses of a coin is 100 * 0.5 = 50
2. The expected number of heads in 25 tosses is 12.5.
3. The expected number of times green pockets win in 20 independent spins of roulette spins is $20 \times \frac{2}{38}$

---
#### Fixed point permutation example [Binomial + Indicator Variables]

$X_n$ = number of students that got their homework after shuffling.

Let $I_i = 1$ if $i^{th}$ student got their own homework 
Let $I_i = 0$ if $i^{th}$ student did not get their own homework 

$X_n = I_1 + I_2 + ... + I_n$

Calculate Expectation of each I.

$$\mathbb{E}[I = j] = 1 \times \mathbb{P}[\text{student j got their homework}] + 0 \times \mathbb{P}[\text{student j did not get their homework}] $$

$\mathbb{E}[I = j] = \mathbb{P}[\text{student j got their homework}]$

$\mathbb{E}[I = j] = \frac{1}{n}$

$$\therefore \mathbb{E}[X_n] = n \times \frac{1}{n} = 1 $$

---

#### Think for n tosses of coins.

In this case, $X$ = number of heads in the sequence.
$I's$ = 1 if head
$I's$ = 0 if tail

$ \mathbb{E}[I] = 1 \times \mathbb{P}[\text{head}] + 0 = 1 \times \mathbb{P}[\text{head}] = p$

$$ \therefore \mathbb{E}[X] = n \cdot p $$

---

#### Example: Balls and Bins

Throw m balls into n bins.

$Y_n$ = number of empty bins.

Find the $\mathbb{E}[Y_n]$. On average, how many number of bins will be empty?

$I's = 1$ if the bin is empty.
$I's = 0$ if the bin is not empty.

$$ \therefore Y_n = I_1 + I_2 + ... + I_n $$

$$ \mathbb{E}[I = j] = \mathbb{P}[\text{bin j is empty}] $$

$$ \mathbb{E}[I = j] = \frac{(n-1)^m}{n^m} $$
$$ \mathbb{E}[I = j] = \left(\frac{(n-1)}{n}\right)^m $$

$$ \therefore \mathbb{E}[Y_n] = n \times \left(\frac{(n-1)}{n}\right)^m $$

---

<div style="page-break-after: always;">

## Note 19 Geometric and Poisson Distributions.

So far, we have explored two Distributions above.

1. Binomial Distribution : n **independent** trials with **replacement**
$$ \mathbb{P}[X = k] = \binom{n}{k} p^k (1-p)^{n - k} $$

$$ \mathbb{X} \sim \text{Binom}(n, p)$$ 

$$ \text {Expected Value of Binomial Random Variable} = \mathbb{E}[X] = n \cdot p $$

2. HyperGeometric Distribution : n **dependent** trials with **no-replacement**
$$ \mathbb{P}[X = k] = \frac{\binom{B} {k} \cdot \binom{N-B}{n - k} } {\binom{N}{n}} $$

$$ \mathbb{X} \sim \text{HyperGeometric}(N, B, n)$$ 


$$ \text {Expected Value of HyperGeometric Random Variable} = \mathbb{E}[X] = n \cdot \frac{B}{N} $$ [==need reviews==]

## Geometric Distribution [Exponenetial Decays]

![alt text](image-13.png)

> :warning: :exclamation: **Definition**:
    Geometric Distribution represents the probability of getting the **first** success after a **consecutive** number of failures.

##### One example: 
Toss a coin until the coin turns out to be head. In this case,

success = event that the coin is head.
failure = event that the coin is tail.
$ \mathbb{P}[\text{head}] = p$
$\therefore Event = \text{TTTTTTH} $ for example.

Let X := number of trials needed to get the first success.

To calculate the distribution of X:

$\mathbb{P}[X = 1] = p $                --- *TH*
$\mathbb{P}[X = 2] = (1 - p)p $ --- *TH*
$\mathbb{P}[X = 3] = (1 - p)^{2}p $     ----    *TTH*
$\mathbb{P}[X = 4] = (1 - p)^{3}p $     ----    *TTTH*
...
$\mathbb{P}[X = k] = (1 - p)^{k - 1}p $  

$$ \mathbb{X} \sim \text{Geom}(p)$$

<small> parameters = `p` = success probability </small>

---

### Memorylessness of Geometric Distribution
> :warning: Note: Geometric Distribution is Memoryless. "Waiting time" until an event occurs does not depend on how much time has passed already although it may look like it depends. (Similar to Gambling Fallacy).

$$ \mathbb{P}(X > x + n | X > n) = \mathbb{P}(X > x) $$

---
### Expectation of Geometric Random Variable

$$ \mathbb{E}[X] = \sum_{i = 1}^{\infin} i \times \mathbb{P}[X = i] =  \sum_{i = 1}^{\infin} (1 - p)^{i - 1}p = p \sum_{i = 1}^{\infin} (1 - p)^{i - 1}  $$

**Another Easier Interpretation is**:

$$ \mathbb{E}[X] = \sum_{i=1}^{\infin}\mathbb{P}[X \ge i] $$

* $ \mathbb{P}[X \ge i]$ . This equation is the same as probability that number of trials needed to succeed >= i.

* In other words, you must fail i - 1 times before succeeding no matter what.

* Therefore, $ \mathbb{P}[X \ge i] = (1-p)^{i -1} $

* If we use Taylor Expansion: $\mathbb{E}[X] = \frac{1}{1 - (1 - p)}  = \frac{1}{p}$







</div>

















<div style="page-break-after: always;">

## Poisson Distribution

Poisson Distribution is in fact came from **Binomial Distribution**, particularly, when $n \rightarrow \infin \, and \, p \rightarrow 0$

Usage: Rare events (radioactive source, typos on a page, number of phone drops, number of chocolates in cookies)

if we know the average -> predict how many times an event will occur during a specific period.

Formula:

$$ \mathbb{P}[X = k] = e^{-\lambda} \frac{\lambda^k}{k!}$$

if we sum up all the events of distributions -> probability should sum up to 1.

$$ \sum_{k=0}^{\infin} \mathbb{P}[X = k] = \sum_{k=0}^{\infin}e^{-\lambda} \frac{\lambda^k}{k!}$$

$$ \sum_{k=0}^{\infin} \frac{\lambda ^ k}{k!} = e^{-\lambda} $$

![alt text](image-14.png)

--- 

#### Examples:
Goals in a world soccer match
$\lambda$  = 2.5 [average number of goals per match] 

**Key Idea**: *Now, we want to find the probability of getting "X" amount of goals in one match.*


$$\text{no goal,} \; \mathbb{P}[K = 0] = e^{-2.5}\frac{2.5^0}{0!} $$
$$\text{one goal,} \; \mathbb{P}[K = 1] = e^{-2.5}\frac{2.5^1}{1!} $$
$$\text{two goal,} \; \mathbb{P}[K = 2] = e^{-2.5}\frac{2.5^2}{2!} $$
$$\text{three goal,} \; \mathbb{P}[K = 3] = e^{-2.5}\frac{2.5^3}{3!} $$
$$\text{more than three goal,} \; \mathbb{P}[K > 3] = 1 - [\mathbb{P}[K = 0] +\mathbb{P}[K = 1] + \mathbb{P}[K = 2] + \mathbb{P}[K = 3]] $$

---

## Expectation of Poisson Random Variable

$$\mathbb{E}[X] = \lambda$$

*That's it. This makes sense, since $\lambda$ is a form of average, and expected value is the value we should get on average if we have enough trials.*

---

### Sum of Independent Poission Random Variables. [Must be Independent]

$\text{Given } X \sim Pois(\lambda) \text{ and Y} \sim Pois(\mu)$:

$$ \mathbb{P}[X + Y = k] = e^{-(\lambda + \mu)} \cdot \frac{{(\lambda + \mu)}^k}{k!} $$

---


## Poisson vs Binomial

What's the relationship between Poisson distribution and Binomial Distribution?

* The binomial distribution tends toward the Poisson Distribution as $n \rightarrow \infin\; ,p \rightarrow 0$ and $n\cdot p$ stays **constant**. 

* Remember that the Expectation of Binomial Distribution is $E[X] = n \cdot p$

* The posson Distribution with $\lambda = np$ closely approximates the binomial distribution if n is large and p is small. (like in the radioactive decay case).

#### Examples:

Balls and Bins
Let X:= number of balls in bin1
Then X ~ Binom($ n, \frac{1}{n}$)

$\mathbb{E}[X] = x \cdot \frac{1}{n} = 1$

**if n is very large:**

$X \sim Pois(1)$

---

#### More Generally: 
$$ \text{Binom} (n,\; \frac{\lambda}{n}) \rightarrow \text{Pois}(\lambda) \text{ ;  as n} \rightarrow \infin$$


</div>



---

Footnote:
+ Study Lecture 18 + Note 18 *[Applications]*
+ Coupon Collecting Examples
