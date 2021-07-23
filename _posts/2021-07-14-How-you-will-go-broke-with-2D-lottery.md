---
layout: post
title:  "How You Will Go Broke with 2D-Lottery"
date:   2021-07-18 11:37:07 +0630
categories: jekyll update
---
Several weeks ago, [2D Lottery](https://en.wikipedia.org/wiki/Thai_lottery#Demographics) was increasingly trending on my Facebook newsfeed. While some people were mourning because of losing, others were being **House** themselves. And of course, there would also be a couple of people who won the lottery because the chances of winning are really high: 1 in 100. Incidentally, one of my friends in my Facebook account is a **House** and posted about rules and regulations. Here is the simple rule: 

> <font size="4">If you win the lottery, you will get 80x of what you had paid.</font>

This rule stuck out to me, and I thought a lot about this. I asked one of my friends to give it a try; he said *"Yes"*, and sent me 10 random numbers. I reversed all of his numbers - a total of 20 numbers - and gave it a chance. In retrospect, I reckon this was **really stupid**, and I will explain why in the following paragraphs.

---
<br>
### Superstitions
Most people pick numbers from dreams, news, events or oddities they observe: the number of bananas on a stalk or the appearance of a two-headed fish, cats with strange markings, the pattern a snake leaves in the dirt, bla bla..

> Various websites and soothsayers help interpret these events, with rats viewed as a one, water a two, snakes a five (large) or six (small), and anything related to royalty a nine. Many Thais (*including Myanmar*) believe that calamity can beget good fortune, and that tragedy may give rise to powerful ghosts who offer guidance on winning numbers. - Wikipedia

In fact, this kind of information increases the odd of the **House** winning. Since the lottery numbers are [stastically independent](https://en.wikipedia.org/wiki/Independence_(probability_theory)). A lot of people try to figure out some form of a pattern in lottery ticket. In reality, there is no absolute pattern to it and winning always happen to be **random**. 

Let's talk about *the fortune tellers*; aka *the number givers*. From my poor research, they usually announce 16 numbers everyday. Let's think about this properly. The probability of he/she getting the right number is already ![](https://latex.codecogs.com/gif.latex?%5Cfn_cm%20%5Cfrac%7B16%7D%7B100%7D). Instead of following his numbers, you can also choose some random 16 numbers, which would lead to the same probability of winning.

---
<br>
### Probability and Gambling Mathematics
Here is an example to demonstrate the maths behind the 2D lottery. Let's say you play the game with 100 MMK. The probability of winning 80x is 0.01; and the probability of lossing 100 is 0.99. To calculate the [expected value](https://en.wikipedia.org/wiki/Expected_value): 

**Expected Value** = ![](https://latex.codecogs.com/gif.latex?%5Cfn_jvn%20%5Csmall%208000%20*%5Cfrac%7B1%7D%7B100%7D%20-%20100*%5Cfrac%7B99%7D%7B100%7D) = **- 19** MMK

Here, you might say "Well, this is just the expected value. How can I know that my bet will exactly turn out to be like this? I could be so *lucky* and the luck turns into a *winning streak*." 

However, according to [Law Of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers):

> The average of the results obtained from a large number of trials should be close to the expected value and will tend to become closer to the expected value as more trials are performed.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Lawoflargenumbers.svg/600px-Lawoflargenumbers.svg.png)

To make it easier to understand:
> When tossing a fair coin for 5 times, it is feasible to get a 5 head or 5 tail straight. But, when it comes to 1000 times, it is almost impossible to get a 1000 head or 1000 tail straight.

**Simulation of a coin that is red on one side and blue on the other side** | ![](https://upload.wikimedia.org/wikipedia/commons/4/49/Lawoflargenumbersanimation2.gif)

---
<br>

Besides, let's calculate the **House Edge**, which is the average tendency of the House "**Winning**" or You "**Loosing**". Using Expected Value (19 MMK) from the previous calculation:

**House Edge** =  ![](https://latex.codecogs.com/gif.latex?%5Cfn_jvn%20%5Csmall%20%5Cfrac%7BExpectedValue%7D%7BAmountSpentByGambler%7D%20*%20100%25)  =  ![](https://latex.codecogs.com/gif.latex?%5Cfn_jvn%20%5Csmall%20%5Cfrac%7B19%7D%7B100%7D%20*%20100%25) ---> **19%**

*House Edge of 19% is ridiculous!!!*; that's why in the beginning of this blog, I said it's a stupid bet.

To explain it briefly, if you bet *2000 MMK* everyday for a month (30-days), you will be losing approximately *11400 MMK* every month! For a year? It will be *136800 MMK*! Of course, this value is not accurate enough since the number of trials is fairly small and billions away from approaching infinity. 






*My opinion here is if you win big accidentally, just quit and leave the game*.

---
<br>

If you still believing that 2D lottery is a cash cow, you must take a look at other types of casino games and their house advantage

* [Blackjack](https://en.wikipedia.org/wiki/Blackjack) - 1.5%
* [Craps](https://en.wikipedia.org/wiki/Craps) - 1.4% - 5%
* [Baccarat](https://en.wikipedia.org/wiki/Baccarat_(card_game)) - 1.5%
* [Roulette](https://en.wikipedia.org/wiki/Roulette) - 5.26%

PS. I am just speculating here, and I'm not fond of gambling.. Furthermore, [about 70% of lottery winners end up bunkrupt](https://www.cleveland.com/business/2016/01/why_do_70_percent_of_lottery_w.html). But this makes my point. Don't bet on something with big house edge value. Or, you would end up going flat broke! At the end of the day, it's just a form of **entertainment**.