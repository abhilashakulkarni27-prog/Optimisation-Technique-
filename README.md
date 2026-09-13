# Operations Research Assignment

This project contains Python implementations of two Operations Research
optimization methods.

## Part A - Big-M Simplex Method

The first part implements the Big-M Simplex Method for solving a
constrained Linear Programming Problem (LPP).

### Case Study

The selected case study is an Advertising Media Selection problem.

The company has four types of advertisements:

- TV
- Radio
- Newspaper
- Online

The program finds the number of advertisements to select from each medium
in order to maximize the total advertising benefit while satisfying the
given constraints.

### Methods Used

- Conversion of LPP into standard form
- Slack variables
- Surplus variables
- Artificial variables
- Big-M method
- Simplex iterations
- Optimal solution detection
- Infeasible and unbounded solution detection

### Result

For the given case study, the optimal solution is:

- x1 = 0
- x2 = 6
- x3 = 6
- x4 = 0

Maximum objective function value:

**Z = 360**

## Part B - Transportation Problem

The second part solves a transportation optimization problem using:

1. Vogel's Approximation Method (VAM)
2. MODI (Modified Distribution) Method

### Case Study

The selected case study is an Ambulance Transportation Problem.

There are 3 ambulance stations and 4 hospitals. The program determines
how many ambulance trips should be sent from each station to each hospital
while minimizing the total transportation cost.

### Methods Used

- Transportation problem formulation
- Supply and demand balancing
- Vogel's Approximation Method (VAM)
- Initial Basic Feasible Solution
- MODI method
- Opportunity cost calculation
- Improvement loop
- Optimal allocation
- Minimum transportation cost

### Input Data

Supply:

| Station | Available Trips |
|--------|-----------------:|
| Station 1 | 20 |
| Station 2 | 30 |
| Station 3 | 25 |

Demand:

| Hospital | Required Trips |
|----------|---------------:|
| Hospital 1 | 10 |
| Hospital 2 | 20 |
| Hospital 3 | 25 |
| Hospital 4 | 20 |

Transportation cost matrix:

|          | H1 | H2 | H3 | H4 |
|----------|---:|---:|---:|---:|
| Station 1 | 8 | 6 | 10 | 9 |
| Station 2 | 9 | 7 | 4 | 2 |
| Station 3 | 3 | 5 | 6 | 4 |

### Program Flow

The program first uses VAM to obtain an initial basic feasible solution.

The initial solution is then given to the MODI method. MODI checks the
solution for optimality and improves the allocation if a better solution
is possible.

The program finally displays the optimal ambulance allocation and the
minimum transportation cost.

## Requirements

Python 3.x

NumPy

Install NumPy using:

```bash
pip install numpy
