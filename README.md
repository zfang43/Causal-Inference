# Project 4: Causal Inference

### [Project Description](doc/project4_desc.md)

Term: Spring 2021

+ Team #3
+ Projec title: Causal Inference Algorithms Evaluation
+ Team members
	+ Fang, Zi zf2258@columbia.edu
	+ Gao, Catherine cyg2107@columbia.edu
	+ Sang, Siyuan ss6165@columbia.edu
	+ Washington, Eve esw2175@columbia.edu

+ Project summary: In this project, we evaluate three causal inference algorithms to compute the average treatment effect (ATE) on two distict datasets and compare their computational efficiency and performance. One dataset contains high dimensional data and another contains low dimensional data. We will use L1 penalized logistic regression to estimate the propensity scores for these two datasets then apply the three methods to calcualte ATE for each dataset. Below is a summary of the results:

| **Algorithm** | **Data Type** | **Run Time** | **ATE** | **Accuracy** | 
|:-------------|:-------:|:-------:|:-------:|:-------:|
| Propensity Scores Matching (full)    | Low| 1.7| 0.36| 17%|
| Doubly Robust Estimation    | Low| 0.11| 2.09| 100%|
| Stratification   | Low| 0.03| 2.38| 86%|
| Propensity Scores Matching (full)    | High| 11.81| -11.71| 21%|
| Doubly Robust Estimation    | High| 0.23| -57.04| 96%|
| Stratification   | High| 0.03| -59.83| 91%|

	
**Contribution statement**: 

Catherine Gao: coordinated and attended all group meetings, performed propensity score estimation and reviewed results from all models, created Main Report and explained model comparisons, updated Github homepage and folders, put together presentation file and is the presenter of the group.

Zi Fang: attended all group meetings, performend doubly robust estimation, reviewed evaluation results, and debugged the doubly robust codes in the main report.

Eve Washington: attended all group meetings, performed stratification, helped to validate doubly robust algorithm, and reviewed evaluation results.

Siyuan Sang: attended all group meetings, performed propensity score matching and reviewed evaluation results.


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.


```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
