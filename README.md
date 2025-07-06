# Predicting Formula One Race Outcomes 
A continuation of the Erdos Institute Data Science Boot Camp project https://github.com/Merlin117/erdos_ds_f1. 

# Table of Contents
1. [Introduction](#Introduction)
2. [Previous Results](#Previous-Results)
3. [Changes to the Model](#Changes-to-the-Model)
4. [Potential Model Features](#Model-Features)
5. [Feature Analysis and Selection](#Feature-Analysis-and-Selection)
6. [Model Testing](#Model-Testing)

## Introduction
Formula One races are exciting and many people come from all over in order to witness the high-speed stakes on the racet track! Our goal in the summer boot camp was to try quantify what 'excitment' could mean for a formula one race, and then use the recorded data from past races to try and come up with a model for predicting this excitement value. This information could be valuable to stakeholders in the formula one world: knowing which races could be exciting or not can lead to better marketing strategies, and can also give significant insights to each team and their drivers on what to expect from a race. 

The first inclination was to try and predict the number of overtakes in a race; more overtakes = more exciting races, right? However, we quickly found some simple counterexamples to this idea. For the rest of our discussion, we consider a permutation $\sigma$ to be a finishing grid where each number in the permutation represents the position of that driver in the starting grid of a race. Now consider the two permutations (finishing positions of a race) $\omega = 15234$ and $\tau = 13524.$ Both have the same number of overtakes: in $\omega$ the driver in fifth position overtook the drivers in second, third, and fourth, while in $\tau$ the driver in third overtook the driver in second, while the driver in fifth overtook the drivers in second and fourth. Yet, the race $\omega$ is clearly not as exciting as the race $\tau,$ since only one driver made progress while the rest of the drivers maintained their *local positions*. This lead us to the conclusion that simply using the number of overtakes may not lead to a good metric for measuring the excitement of a race. We needed something that would best account for the way positions were changing both globally and locally. 

This leads us to define the Average Local Position Change (ALPC) of a race with finishing positions $\sigma$:

$$ALPC(\sigma)=\frac{1}{N(N-1)}\sum_{i=1}^N\sum_{j\neq i} |(i-j)-(\sigma^{-1}(i)-\sigma^{-1}(j))|,$$

where $N$ is the number of drivers that finish the race, and $ALPC = 0$ if $N=1.$ This metric looks at what we call the *local position change* of each driver and then averages this value out. The local position change of each driver is given by the inner sum divided by one less than the number of finsihing drivers. Then the ALPC is achieved by averaging out the local position change across all finshing drivers. Notice that the local position change of a driver avoids the issue that we ran into with the number of overtakes. The value $(i-j)-(\sigma^{-1}(i)-\sigma^{-1}(j)) = 0$ if and only if the two racers which started out in positions $i$ and $j,$ respectively, are the same positions away from each other in the finsihing grid. We do allow for negative values since if the two racers in positions $i$ and $j$ end up in positions $j$ and $i,$ we don't want this to add a value of 0. This new metric does exactly what we want and distinguishes between the two finishing positions $\omega$ and $\tau$ since $ALPC(\omega)=1.8$ while $ALPC(\tau)=2,$ and gives to them the appropriate values which we would expect since $\omega$ is not as exciting as $\tau.$ Also notice that $ALPC=0$ only if the starting grid is the same as the finishing grid (i.e. there are no overtakes), just as desired. 

Previously, we had the absolute value bars outside the inner sum instead of inside. The motivation behind this change is that inclusing the absolute value bars inside the sum better captures how much the grid is changing, while also yielding better correlation with the chosen features. Potential issues with this metric, is that it automtically excludes DNFs and that it heavily relies on how many drivers finish the race.

The goal now is to improve upon the existing model by:
1. adding more features
2. combining features to yield the optimal model
3. testing different types of models and tuning hyperparameters to see which one yields the best results

## Previous Results
The previous model ws able predict with an $r^2\approx 0.13$ and a $MSE\approx 0.86.$ We used only linear regression with this model. We also noticed a strong correlation between the number of DNFs and ALPC (as seen below), which makes sense since DNFs are automatically excluded whencomputing ALPC. One would also expect that the more DNFs there are, the less drivers will finish the race heavily lowering the possible ALPC value. 

![image](https://github.com/user-attachments/assets/a11d6c8d-bec0-44cf-a9c5-d2a3091f9781)

## Changes to the Model
One of the downfalls of the previous model was that it only used two features, making it an overly simplistic model. The features chosen also had very small correlation coefficients, meaning that the features did not have add very strong predictive power to the model. We also found out too late how much the number of DNFs impacted the target. 

Here we begin by exploring more possible features for prediction, and then implement a better process for feature selection, only allowing for features which improve the predictive power of the model. We also want to create two models: one to predict the number of DNFs in a race, and then another to predict the ALPC of a race. We will use the predicted values from the DNF model as a feature for the ALPC model. We also explore how different modeling approaches will habdle the data and then choose the approach which will yield the best predictive power. 

## Potential Model Features
Below we list out the chosen potential features for our model and then give an in depth survey for each. The first six features are all continuous data features, while the last three are all categorical data features. 

1. [Average Driver Experience on the track (weighted sum)](#Average-Driver-Experience-on-the-track)
2. [Average Pit Stop Lap, Number, and Time (weighted sum)](#Average-Pit-Stop-Lap,-Number,-and-Time)
3. [Pre Race ALPC (weighted sum)](#Pre-Race-ALPC)
4. [Total Cumulative Constructor Average Points Earned](#Total-Cumulative-Constructor-Average-Points-Earned)
5. [Time Gap Cluster Square Mean (weighted sum)](#Time-Gap-Cluster-Square-Mean)
6. [Time Gap Statistics (weighted sum)](#Time-Gap-Statistics)
7. [Top Ten Diversity](#Top-Ten-Diversity)
8. [Circuit ID](#Circuit-ID)
9. [Country of the Race](#Country-of-the-Race)

### Average Driver Experience on the track
This feature consists of three subfeatures which are then given weights which optimize their sum for correlation: (1) the average driver race count (ADRC), the average cumulative racer points (ACRP), and the average driver experience in years (ADEY). Each is computed for each race, and is also exactly what it sounds like it would be. ADRC is computed by getting first the total number of races a driver has participated in over their formula one career up to but not including the current race, and then averaging over all drivers participating in the current race. ACRP is achieved in a similar manner, by getting first the total number of points a driver has earned over their formula one career up to but not including the current race, and then averaging out over all drivers participating in the current race. Lastly, ADEY is computed by garnering the total number of years a driver as been racing over their formula one career up to but not including the current year of the current race, and then averaging out over all drivers in the current race. Then to get the Average Driver Experience on the track, we use `Optimizing Weighted Sum for Correlation.ipynb` to get the following formula 

$$-0.87595933\cdot ADRC+0.03330413\cdot ACRP+0.48123393\cdot ADEY.$$

On their own each subfeature yields a correlation to the DNF data $<0.7$ while the wieghted sum yields correlation to the number of DNFs $0.7474226769941237.$

### Average Pit Stop Lap, Number, and Time
The subfeatures are exactly what they are in the name of this feature, but are calculated with a little more nuance than the name would imply. Really, the name should be the average average pit stop lap, number, and time, but the above name was chosen for simplicity. The average driver pit stop lap (ADPSL) is computed by getting the average lap each driver takes a pit stop over their formula one career up to but not including the current race and then averaging that value over all drivers in the current race. The average driver pit stop number (ADPSN) is computed by getting the average number of pit stops a driver takes over their formula one career up to but not including the current race, and then averaging that value over all drivers in the current race. The average driver pit stop time (ADPST) is computed by getting the average time in milliseconds a driver as spent in a pit stop over their formula one career up to but not including the current race, and then averaging that that value over all drivers in the current race. To get the weighted sum for this feature, we used `Optimizing Weighted Sum for Correlation.ipynb` to get the following weights 

$$-0.24214759\cdot ADPSL+0.31940806\cdot ADPSN-0.91615667\cdot ADPST.$$

On their own each subfeature yields a correlation to the number of DNFs $<0.65$ but the wieghted sum yields a correlation to the number of DNFs $0.6730485988533443,$ which may only be marginally better but also help to add to the predictive power of the model. 

### Pre Race ALPC
There are six subfeatures, two for each of the free practice (FP), qualifiying (Q), and absolute pace postion grids (pace). The free practice and qualifying grids were obtained by ordering the drivers by their fastest free practice and qualifying time, respectively. The absolute pace psoition grid is obtained by ordering the drivers by theie fastest time from both the qualifying and free practice times. Then the ALPC and the average absolute position change (PC) are taken for each. The absolute position change for a driver is the absolute value of how many positions they gained/lost and then the average absolute position change for a race is the average of each drivers absolute position change. We used `Optimizing Weighted Sum for Correlation.ipynb` to get the following weights 

$$0.00514474\cdot FALPC+0.01086802\cdot FPC-0.69783557\cdot QALPC+0.58691877\cdot QPC+0.29529604\cdot paceALPC-0.28496937\cdot pacePC.$$

On thaeir own, each subfeature yields a correlation to the number of DNFs $<0.48$ but the weighted sum yields a correlation to the number of DNFs $0.5397808558195485.$ 

### Total Cumulative Constructor Average Points Earned
The total cumulative constructor average points earned (TCCAPE) is computed by first calculating the total number of points that a constructor has earned through all their drivers in the current season up to but not including the current race, then for each race the TCCAPE is gotten by averaging out that value over all constructors participating in the current race. The TCCAPE exhibits a correlation value of $-0.513920$ to the number of DNFs in a race. 

### Time Gap Cluster Square Mean
This feature is obtained by getting what we call the *time clusters* for each of the free practice (F), qualifying (Q), and absolute pace position (pace) grids. To find the time cluster of a grid, we must first choose a *time gap interval number* in milliseconds, $I$. This number is then applied to each driver by taking their time in each grid and lumping in all drivers who have a time within $I$ milliseconds of the driver. This creates some number of clusters of drivers who all are within $I$ milliseconds of at least one driver in the cluster. If there are $M$ clusters labeled $C_i$ for $1\leq i\leq M,$ then we can compute the Time Gap Cluster Square Mean (TGCSM) with the following formula: 

$$TGCSM = \frac{\sum_{i=1}^M |C_i|^2-N}{N(N-1)},$$

where $TGCSM=0$ if $N=1.$ The choice of square mean over regular mean was done to somewhat normalize the value, and then the choice to subtract $N$ from the top and bottom of the fraction was done to place values between $0$ and $1.$ If $D\neq 1$ then $TGCSM=0$ if $M=N$ and $TGCSM=1$ if $M=1.$ In otherwords, TGCSM measures how 'close' the drivers are expected to be when finishing the race. 

To choose the best time gap interval number, we plotted the time gap against the correlation value of TGCSM with that gap to DNFs and ALPC. Using `Time Gap Correlation Testing.ipynb`, we found the best time gap intervals which would optimize the correlation values. These are the results. 

![image](https://github.com/user-attachments/assets/f857646c-6e7c-4846-b003-f29546f19984)
![image](https://github.com/user-attachments/assets/ea4de4df-9de8-45e5-a8b0-a10f3f48a5e2)
![image](https://github.com/user-attachments/assets/e492911c-0b86-4d7b-bc7e-1646a6fd5d59)

For the free practice grid we have $I_{ALPC} = 200$ and $I_{DNF} = 220.$ Then, surprisingly, for the qualifying grid we have $I_{ALPC} = 7780$ while $I_{DNF} = 140.$ Finally, for the absolute pace position grid we have $I_{ALPC} = 360$ and $I_{DNF} = 380.$ These are the time gaps which maximize the correlation values, respectively, for each grid. 

We used `Optimizing Weighted Sum for Correlation.ipynb` to get the following weights 

$$-0.93340429\cdot FTGCSM-0.34852292\cdot QTGCSM-0.08537102\cdot paceTGCSM.$$ 

Alone each subfeature has a correlation of $<.4$ to the total number of DNFs but the weighted sum has a correlation value of $0.4009029995634536$ to the total number of DNFs. Again, the improvement is only marginal, but it still adds to the total predictive power of the model.

### Time Gap Statistics
Again we use the free practice (FP), qualifying (Q), and absolute pace position (pace) grids and look at the actual time gaps between each driver in these grids. Once we get a list of the time gaps for each race, we find the statistics for each grid (mean, median, standard deviation, variance) and the sum of the gaps between the first ten drivers in the grid. We then get a weighted sum of the statistics for each grid which maximized correlation: free practice stats weighted sum (FPSWS), qualifying stats weighted sum (QSWS), and the absolute pace position weighted sum (APPSWS). Once we get each of these values for each race, we then take another weighted sum of the weighted sums to maximize correlation value. Using `Optimizing Weighted Sum for Correlation.ipynb` we get the following weights 

$$0.51820936\cdot FPSWS-0.3706651\cdot QSWS+0.77075706\cdot APPSWS.$$

On their own each subfeature yields correlation values to the number of DNFs $<0.5003$ while the weighted sum yeilds a correlation value of $0.5387701704389778$.

### Top Ten Diversity
Which is exactly what it sounds like. We take the top ten drivers in the starting grid of a race and count how many different constructors are represented within the top ten starting drivers. We used `predictor testing.ipynb` to garner the following statistics when comparing agains the number of DNFs:

* $F\text{-statistic} = 3.602885940319581$
* $p\text{ value} = 0.0064373789382687506$
* $\eta^2 = 0.0201726076336946.$

Although the $\eta^2$ implies a relatively small effect size, the $F$-statistic shows moderate variance between groups and the $p$ value shows strong statistical significance. This is the reason we chose to include this as a feature to the DNF model. 

### Circuit ID
Looking at the stats for each circuit, we get the following results using `predictor testing.ipynb` when comparing against the number of DNFs:

* $F\text{-statistic} = 5.533569261680306$
* $p\text{ value} = 1.0887569928328795\times 10^{-25}$
* $\eta^2 = 0.2882031909571808.$

Although the $F$-statistic only shows moderate variance between groups, the $p$ value and $\eta^2$ show just how important this feature is when it comes to predicting DNFs. The $p$ value is way beyond the threshold for major statistical significance, while the $\eta^2$ shows that this feature alone can account for over $25\%$ of the variance. These strong statistical values heavily imply that this categorical feature is a necessity for any model trying to predict the number of DNFs in a race. 

### Country of the Race
This categorical feature is, again, self-explanatory. We use `predictor testing.ipynb` to get the following statistics when comparing against the number of DNFs:

* $F\text{-statistic} = 4.215860892494897$
* $p\text{ value} = 1.9813247453103664\times 10^{-12}$
* $\eta^2 = 0.1580008072004169.$

Again, the $F$-statistic only shows moderate variance between groups. Yet, similar to the Circuit ID feature, the Country of the Race exhibits a massively strong $p$ value which is again way beyond the threshold for statistical significance while also exhibiting a strong $\eta^2$ value which shows that this feature accounts for roughly $15\%$ of the variance. 

## Feature Analysis and Selection
First we look at the correlation of the features between themselves to make sure no features are colinear. Below is a heatmap of the correlation values when compared against each other.

![image](https://github.com/user-attachments/assets/6c9a225f-0334-4463-9057-f0b09a8b667d)

The biggest culprit for colinearity is the Average Driver Experience on the track. This feature correlates heavily with the Average Pit Stop feature, the Time Gap Statistics feature, and the TCCAPE feature. The following VIF chart also confirms this, showing that our Average Driver Experience feature is the feature with the most dangerous VIF value. 

![image](https://github.com/user-attachments/assets/f171141f-8a4d-4b9f-9b86-4cba6530898b)

However, we do not want to drop this feature, since it is our feature with the strongest correlation to the number of DNFs. When testing the features with a Random Forest model, we also see that the Average Driver Experience contributes heavily to the model with the strongest SHAP values as seen in the table below. 

![image](https://github.com/user-attachments/assets/1d1364ae-5aae-44cf-915c-fd174fb46f20)

Thus, out best bet is to combine this feature with another feature. Using the `Optimizing Weighted Sum for Correlation.ipynb`, we find that the best way to increase our predictive power while also avoiding colinearity is to combine the Average Driver Experience (ADE) with Time Gap Statistics (TGS). This gets us what we will call the *Complete Average Driver Experience*. The formula is as follows:

$$0.77186024\cdot ADE+0.63579224\cdot TGS.$$

This gets a new correlation value of $0.7857751712228391$ with the total number of DNFs. However, it's here that we run into a major issue. When testing with different predictive models, using this new feature actually diminishes predictive power. In the previous models, we were getting $r^2$ values consistently $>0.66$. After introducing the new feature, the model consistently had $r^2$ values of $<0.64$. Therefore, we conclude that it is actually better to sacrifice less collinearity for more predictive power, and we keep the original features even though they may correlate heavily with one another. 

## Model Testing
The pattern for model testing went like this: we plug in the features, generate the model without any tuned hyperparameters, and then use the `GridSeach()` method to optimize the model with the best tuned hyperparameters. In each model type, we look at the learning curves for training and validation sets and then compare. Our goal is to get a model which will consistently learn better as more data is included, and which minimizes the train-validation learning gap. These are the different models we tested:

1. [DNF Model](#DNF-Model)
   * [Random Forest](#Random-Forest)
   * [Light GBM](#Light-GBM)
   * [Gradient Boosting](#Gradient-Boosting)
   * [Stacked with MLP Meta](#Stacked-with-MLP-Meta)
   * [Stacked with XGBoost Meta](#Stacked-with-XGBoost-Meta)
   * [Stacked with Elastic Net Meta](#Stacked-with-Elastic-Net-Meta)
   * [Stacked with Ridge CV Meta](#Stacked-with-Ridge-CV-Meta)
   * [Conclusion](#Conclusion)
2. [ALPC Model](#ALPC-Model)

### DNF Model
For the DNF model we use all nine features. We begin with single model methods: Random Forest, Light GBM, and Gradient Boosting. After tuning the hyperparameters of each and optimizing for the best $r^2$ value, we then stack all three models for the best possible predictive power and then test with different meta models. After tuning the hyperparameters of each meta model, we end on one optimized stacked model. 

Each chart for the lunring curves has the validation curve plotted against the training curve, with the shaded area representing the variance of the $r^2$ of the model for that amount of data. 

#### Random Forest
The Random Forest model as the first model we tested for demonstrated promise, as the $r^2$ values were relatively high with the training $r^2$ reaching $\approx 0.85$ and having a narrowing variation band, while the validation $r^2$ values reached $\approx 0.69$ stopping just shy of $0.7$. However, the variance band for the validation curve remained high all throughout the curve, and even seems to increse with more data. Another downfall of this model is the training-validation learning gap which remained at a steady $0.16$ throughout no matter the training set size. 

![image](https://github.com/user-attachments/assets/78f53fd8-f15d-44a3-b8e8-079de3d7d8b4)

#### Light GBM
The Light GBM model shows massive improvements over the Random Tree model. The first notable difference is the drop in trainingn $r^2$ values which decrease from $0.85$ to $\approx 0.775$ with a narrowing variance band. However, the validation curve stays roughly the same exhibiting similar performance plateauing at an $r^2$ value of around $\approx 0.69$, again just shy of $0.7$, with a relatively large variance band which persists throughout the training set size. This model shines with the smallest learning gap being $<0.1$, showing that the Light GBM model performs well to unseen data. 

![image](https://github.com/user-attachments/assets/96ebd8e6-7625-42c0-be3f-9437d32ba7a1)

#### Gradient Boosting 
The Gradient Boosting model performed similarly to the Light GBM model but with a few key differences. The training curve begins at $0.9$ and only decreases to a value of $\approx 0.8$, which leaves us with a larger learning gap since the validation curve behaves roughly the same as the one in the Light GBM model. 

![image](https://github.com/user-attachments/assets/294bf819-3b77-421d-b633-1e92040293c5)

#### Stacked with MLP Meta
First we tried stacking the previous models with the MLP as the meta model. This however, was not the ideal. The learning curves ended up jagged which indicates some instability with this model. There are also no real improvements to the $r^2$ values. 

![image](https://github.com/user-attachments/assets/377956f4-c638-4958-9d81-d3fa8c633fe4)

#### Stacked with XGBoost Meta
The XGBoosted model performed somewhat better than the MLP model, still exhibiting a jagged like curve behavior, but not as severe. The learning gap size is also a key benefit to this model, being the second smallest gap afer the Light GBM model. However, the variatiance bands are still quite large here.

![image](https://github.com/user-attachments/assets/32307a1f-7afe-43c9-b58d-48e31570ddab)

#### Stacked with Elastic Net
Here we have the best performing model so far, with the highest validation $r^2$ value and the third smallest learning gap. The training curve also deceases showing that the model doesn't overfit to the training data. The only pitfall of this model is how jagged the curve remains to be, indicating some slight instability with the learning.

![image](https://github.com/user-attachments/assets/35f372e6-5f60-46ec-a89d-04a676b7d606)

#### Stacked with Ridge CV
The stacked model with Ridge CV as the meta model performs excellently, smoothing out those jagged edges that were prominent in every other stacked model. This model learns consistently, and ends with a high validation $r^2$ of $\approx 0.705$ and a decreasing learning gap $<0.1$. This model shows promise, asthe validation $r^2$ values are still increasing with more data. 

![image](https://github.com/user-attachments/assets/d7a53dc7-2c58-46dc-9d13-2f2bae29d927)

#### Conclusion 
Out of all the models, the stacked model with Ridge CV as the meta model shines as the key model to use for predicting DNFs. This model had the highest validation $r^2$, shows minimal signs of overfitting, exhibits learning stability, and will close the learning gap as more data is gathered.

