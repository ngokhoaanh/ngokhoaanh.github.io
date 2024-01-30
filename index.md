# Portfolio
---
## M2 ISDS - ISUP 

Complete projects for the  [***M2 ISDS: Ingénieur Statistique et Data Science***](https://isup.sorbonne universite.fr/formations/filiere-ingenierie-statistique-et-data-science-isds) at Sorbonne University (2022-2023).

---
### Machine Learning: Diabetes classification

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/NGO_LAFFINEUR.html)

<div style="text-align: justify">In this comprehensive project, our objective is to classify diabetes using advanced Machine Learning techniques. The project initiates with a thorough analysis of the dataset, emphasizing the understanding of key variables and their relationships. This phase includes detailed data visualization and correlation studies to identify patterns and interactions critical for diabetes classification.

Subsequently, the project incorporates unsupervised learning, specifically using a K-means clustering approach, to discern distinct data groups. This methodology aims to enhance our analytical perspective and contribute to more nuanced data interpretation.

The pivotal aspect of the project involves the systematic training and evaluation of various classifiers. These include Decision Trees, Logistic Regression, Random Forest, MLP, SVM, QDA, LightGBM, Gradient Boosting, and Neural Networks. Our approach is methodical, comparing each model's performance to ascertain the most effective algorithm for our specific dataset.

The project is not only a pursuit of the optimal classification model but also an endeavor in deep learning and discovery in the field of Machine Learning.</div>

<br>
<center><img src="images/diabetes.png"/></center>
<br>


---
### Time series: Wikipedia Traffic Forecast

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/Projet_Stat_Prevision.html)


<div style="text-align: justify">This project presents a thorough investigation into the dynamics of user interactions and server performance optimization by forecasting Wikipedia page traffic. The significance of this study lies in its dual focus: gaining insights into user behavior on Wikipedia and enhancing the server performance and availability of this vast online encyclopedia. Our objective is to achieve precise predictions of Wikipedia page traffic through the application of three distinct machine learning models, each contributing uniquely to our forecasting accuracy.


***ARIMA Model***: Specializing in autocorrelation, the ARIMA model leverages time-series data, enabling us to make more nuanced predictions.
***XGBoost***: As a decision-tree-based ensemble machine learning algorithm, XGBoost excels in identifying complex, non-linear patterns within the data.
***Random Forest Algorithm***: Comprising multiple decision trees, this model adds an extra dimension of accuracy to our predictions.
This study provides an in-depth exploration of these machine learning models. We will demonstrate their individual capabilities and how their integrated application can significantly enhance the accuracy of traffic forecasting on Wikipedia.</div>
<br>
<center><img src="images/wikipedia.png"></center>
<br>

---
### Industrial statistics

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/Rapport_Stats_indus_Remy.pdf)


<div style="text-align: justify">This project represents a comprehensive exploration at the nexus of industrial statistics, environmental science, and economics, targeting a pragmatic challenge in flood management. Our focus is on determining the optimal embankment heights to mitigate flood risks effectively. The foundation of our study is a robust dataset derived from real-world flood events, water flow dynamics, and economic impact assessments. The goal is to ascertain the most appropriate embankment height that not only offers adequate flood defense but also maintains economic viability. We adopt a three methodological approach, each offering a distinct perspective to analyze the data and shape our conclusions:

***Historical Measurement-Based Approach***: Leveraging historical flood data, this approach aims to utilize past measurement records as a guide for future embankment design, providing a practical, experience-based perspective.

***Hydraulic Model-Based Approach***: Here, we employ hydraulic modeling techniques to simulate water flow patterns and predict flood probabilities, offering an engineering-centric viewpoint for optimal embankment height determination.

***Economic Model-Based Approach***: This facet of our study integrates economic modeling to assess the financial implications of various embankment heights, ensuring our flood protection strategies are economically sustainable.

Through the integration of these three distinct research methodologies, our project aims to arrive at a multi-faceted solution that encapsulates a broad range of considerations, thereby ensuring a robust and comprehensive approach to flood management.</div>
<br>
<center><img src="images/fiabilite.png"></center>
<br>

---
### Parallel Computing: Parallel Implementation of Conway's Game of Life


<div style="text-align: justify">John Conway came up with the Game of Life in 1970. The game demonstrates the fact that some simple local rules can lead to interesting large-scale life behavior(birth, reproduction and death). The game is played in a 2 dimensional grid N x N, made of cells, that can be either alive, or dead. The game does not have any players, thus it does not require any input by the user. Each cell has at most 8 neighbours, that determine its state in the next generation. The re-formation of the grid from generation to generation is done simultaneously, meaning that each state in the next generation depends exclusively in the state of the cell and its neighbours. Our goal is to implement this game, using Parallel programming.</div>
<br>
<center><img src="images/game_of_life.gif"></center>
<br>


---
### Statistical Quality Control

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/Projet_CQ.pdf)


<div style="text-align: justify">In this project, we delve into the world of manufacturing with a focus on improving process efficiency through statistical quality control. The objective is to employ robust statistical methods and adequacy tests, thus ensuring processes are streamlined, waste is minimized, and a higher proportion of specification-conforming products are produced.

Our approach involves the construction and utilization of control charts, including the CUSUM (Cumulative Sum Control Chart) and EWMA (Exponentially Weighted Moving Average). These powerful statistical tools allow us to precisely detect moments of rupture in the process and accurately identify false alarm rates.

The CUSUM chart is particularly effective at identifying small shifts from the process target over time, while the EWMA chart is adept at detecting larger, sudden shifts. Together, they provide a comprehensive toolset for monitoring process variability and maintaining control.

The results from this project not only optimize the manufacturing process but also contribute to an overall reduction in waste. The insights and techniques gleaned from this work prove invaluable in promoting more efficient, sustainable, and profitable manufacturing practices.</div>
<br>
<center><img src="images/controle.png"/></center>
<br>

---

### Latent structure models

#### Kmeans and Hierarchical Ascending Classification

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/MSLTP1.html)

<div style="text-align: justify">Objective of this project is to warn against an overly systematic or blind application of PCA in a clustering study, and to explore and compare the behavior of Kmeans and ascending hierarchical clustering.</div>

<br>
<center><img src="images/latent1.png"/></center>
<br>

---

#### Mixing models, Model-Based Clustering, EM algorithm

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/MSLTP2.html)

<div style="text-align: justify">Objective of this project is first to explore the behavior of the EM in the context of a simple Gaussian mixture model at J components in dimension 1
  
$$
  M = \{ \sum_{j=1}^J \pi_j \phi(\cdot; \mu_j, \sigma_j^2) : (\pi_1, \ldots, \pi_J) \in \Pi_J, \mu_1, \ldots, \mu_J \in \mathbb{R}, \sigma_1^2, \ldots, \sigma_J^2 \in \mathbb{R}_{+}^{*} \}
$$
  
  
$$
  \text{and } \Pi_J = \{(\pi_1, \ldots, \pi_J) \in [0, 1]^J : \sum_{j=1}^J \pi_j = 1\} \text{ and } \phi(\cdot; \mu, \sigma^2) \text{ the Gaussian density of expectancy } \mu \text{ and variance } \sigma^2
$$
  
It is then to initiate the model-based clustering in higher dimension with the Rmixmod package, which allows to fit mixing models.
</div>

<br>
<center><img src="images/latent2.png"/></center>
<br>

--- 

#### Bayesian methods, Markov Chain Monte Carlo(MCMC)

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/MSLTP3.html)

<div style="text-align: justify">This project is an introduction to Bayesian methods in the context of Gaussian mixture models models, and in particular to the Gibbs sampler and the Metropolis-Hastings algorithm. We are interested in the following mixture model:

$$
  M = \{ \pi \phi(\cdot; \mu_1, 1) + (1 - \pi) \phi(\cdot; \mu_2, 1) : \mu_1 \in \mathbb{R}, \mu_2 \in \mathbb{R} \},
$$
  
with $\phi$ being the Gaussian density on $\mathbb{R}$ and $\pi \neq \frac{1}{2}$ known.
  
The prior distribution of $(\mu_1, \mu_2)$ is given by: $\mu_1 \sim \mathcal{N}(\delta_1, \frac{1}{\lambda}), \mu_2 \sim \mathcal{N}(\delta_2, \frac{1}{\lambda}), \delta_1, \delta_2 \in \mathbb{R}$ and $\lambda > 0$, with $\mu_1$ and $\mu_2$ independent.
</div>

<br>
<center><img src="images/latent3.png"/></center>
<br>

---
## M1 ISIFAR: Statistical and Computer Engineering for Finance, Insurance, and Risk

Complete projects for the [***M1 ISIFAR: Statistical and Computer Engineering for Finance, Insurance, and Risk***](https://master.math.univ-paris-diderot.fr/annee/m1-isifar/)) at Université de Paris (2021-2022).

---

### Asset Allocation and Portfolio Optimisation

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/Rapport_projet.pdf)

<div style="text-align: justify">In this project, we grapple with a classic challenge in finance: determining an investment strategy that yields optimal performance over a specified time horizon, given a set of traded assets. Our exploration takes us into the core aspects of discrete-time portfolio optimization, where the balance between risk and reward comes to the fore.

We examine three of the principal criteria in portfolio optimization to provide a comprehensive view of the problem at hand. Our methodology revolves around two potent solution strategies:

Dynamic Programming Method: A versatile approach that breaks down the larger problem into smaller, manageable sub-problems, thereby simplifying complex optimization tasks.
Martingale Method: A powerful probabilistic technique, valuable in the context of investment strategies and financial forecasting.
Further enhancing our exploration, we dive into the practicalities by providing explicit examples for both the logarithmic utility function, which captures the investor's level of risk tolerance, and the full binomial case, a method used to model the dynamics of an asset's price over time.

By undertaking this project, we aim to offer profound insights into the world of finance, contributing to more efficient, optimized investment strategies and improved financial outcomes.</div>
<br>
<center><img src="images/optimization.png"/></center>
<br>

---
### SVD Analysis & Life Tables

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/Projet--COMPLET-.html)


<div style="text-align: justify">In this captivating project, we embark on an exploration of demographic data and its intricate patterns using an array of powerful techniques. Our collaboration spans various data structures, including data.frames, tibbles, and data.tables, where we leverage the capabilities of dplyr and other query languages, such as data.table, to process and manipulate the data effectively.

With the invaluable data provided by the esteemed Human Mortality Database organization (https://www.mortality.org), we dive into a world of demographic insights. Our focus is on visualizing and analyzing multivariate datasets, with a special emphasis on life tables as an essential component of demographic research.

To unravel the underlying trends and uncover hidden patterns, we employ a range of powerful techniques, including Singular Value Decomposition (SVD) analysis and Principal Component Analysis (PCA). These matrix-oriented methods provide a comprehensive understanding of the interrelationships and variations within the data.

Moreover, we delve into the application of the renowned Lee-Carter model, a widely accepted methodology for predicting mortality quotients. This model not only aids in forecasting future mortality trends but also plays a crucial role in informing public health policies and insurance industry practices.

By combining statistical analysis, data manipulation, and advanced modeling techniques, this project offers profound insights into demographic dynamics, contributing to our understanding of population trends, mortality patterns, and the intricate interplay between various demographic factors.</div>
<br>
<center><img src="images/life.gif"/></center>
<br>

---
<center>© 2023 Ngo Khoa Anh. Powered by Jekyll and the Minimal Theme.</center>
