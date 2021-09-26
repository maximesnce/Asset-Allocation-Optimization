# Portfolio Optimization

The goal of this project was to discover and become familiar with the world of assets allocation. 
We had to design an optimization tool that we complexified along the whole project. 
We started by only taking into account the diversification ratio. Then, we started incrementing the complexity of the algorithm by adding other elements to the optimization function such as a decorrelated basket (beta_market = 0) or an indicator of estimated future performances (Relative Stregnth Index - RSI).

All of our results can be found in the final report in this repository.
Regarding the coding files, here is how they are organized.

## Tools

Main functions to facilitate the different computations of the project such as rolling windows, returns and other performances indicators. It includes the core otpimization function. 

## Data

Because we used the Yahoo Finance API for our financial data, we quicly realized we would have to store the data rather than accessing it at every iteration of the algorithm. It especially started to get problematic when we worked on the graphic interface which requires new computation every time a parameter was changed. 
This why, you will find in this folder the data we used to solve our optimization problem.

## Optimizer

In this folder you will find the principal results and computations of our project. 

## Dashboard

This file contains the necessary code to build our graphic interface. It was a way to present our work in a clear and comprehensible way.
It was also an efficient solution to compute different charts by tweaking the parameters of the optimization function.
