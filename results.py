#   Proj:   TFG - Evaluating RL algorithms and policies in gFootbal enviroment
#   File:   results.py
#   Desc:   Loads the results from evaluate.py and performs stadistics tests to compare the data
#   Auth:   Enrique Boya Falcón
#   Date:   2021

import os
import sys

import csv

from scipy import stats

import numpy as np

def importResults(file: str, type: str):

    if type == 'algorithms':

        PPO = []
        DQN = []
        A2C = []

        with open(file, mode='r') as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:
                PPO.append(float(row["PPO"]))
                DQN.append(float(row["DQN"]))
                A2C.append(float(row["A2C"]))

            return PPO, DQN, A2C

    elif type == 'policies':

        scoring = []
        checkpoint = []

        with open(file, mode='r') as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:
                scoring.append(float(row["scoring"]))
                checkpoint.append(float(row["scr+chp"]))

        return scoring, checkpoint
        
    else: 
        assert False, "unhandled option"
                

# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
            
def penaltyTest():

    print("Penalty")

    PPO, DQN, A2C = importResults('../Data/Results/Penalty.csv', 'algorithms')

    #   -------------------------------------------------------------

    print("\nShapiro-Wilk Normality test")

    print("\tA2C")
    stat, p = stats.shapiro(A2C)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tDQN")
    stat, p = stats.shapiro(DQN)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tPPO")
    stat, p = stats.shapiro(PPO)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    #   -------------------------------------------------------------

    print("\nBartlett’s test for equal variances")
    stat, p = stats.bartlett(A2C, DQN, PPO)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably equal variances')
    else:         print('\t-> Probably different variances')

    #   -------------------------------------------------------------

    print("\nKruskal-Wallis H-test")
    stat, p = stats.kruskal(A2C, DQN, PPO)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    print("\nMann-Whitney U Test")
    
    print("\tA2C - DQN")
    stat, p = stats.mannwhitneyu(A2C, DQN)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    print("\tDQN - PPO")
    stat, p = stats.mannwhitneyu(DQN, PPO)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    print("\tA2C - PPO")
    stat, p = stats.mannwhitneyu(A2C, PPO)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    print("\nMedian comparison")
    medians = [np.median(A2C), np.median(DQN), np.median(PPO)]
    print("\tA2C {:.2f}".format(medians[0] * 100))
    print("\tDQN {:.2f}".format(medians[1] * 100))
    print("\tPPO {:.2f}".format(medians[2] * 100))


def counterattackTest():

    print("Counterattack")

    scoring, checkpoint = importResults('../Data/Results/Counterattack.csv', 'policies')

    #   -------------------------------------------------------------

    print("\nShapiro-Wilk Normality test")

    print("\tScoring")
    stat, p = stats.shapiro(scoring)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tCheckpoint")
    stat, p = stats.shapiro(checkpoint)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    #   -------------------------------------------------------------

    print("\nBartlett’s test for equal variances")
    stat, p = stats.bartlett(scoring, checkpoint)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably equal variances')
    else:         print('\t-> Probably different variances')

    #   -------------------------------------------------------------

    print("\nMann-Whitney U Test")
    
    print("\tScoring - Checkpoint")
    stat, p = stats.mannwhitneyu(scoring, checkpoint)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    print("\nMedian comparison")
    medians = [np.median(scoring), np.median(checkpoint)]
    print("\tScoring {:.2f}".format(medians[0] * 100))
    print("\tCheckpoint {:.2f}".format(medians[1] * 100))



def iniestaTest():

    print("Iniesta")

    scoring, checkpoint = importResults('../Data/Results/Iniesta.csv', 'policies')

    #   -------------------------------------------------------------

    print("\nShapiro-Wilk Normality test")

    print("\tScoring")
    stat, p = stats.shapiro(scoring)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    print("\n\tCheckpoint")
    stat, p = stats.shapiro(checkpoint)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably gaussian')
    else:         print('\t-> Probably not gaussian')

    #   -------------------------------------------------------------

    print("\nBartlett’s test for equal variances")
    stat, p = stats.bartlett(scoring, checkpoint)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably equal variances')
    else:         print('\t-> Probably different variances')

    #   -------------------------------------------------------------

    print("\nMann-Whitney U Test")
    
    print("\tScoring - Checkpoint")
    stat, p = stats.mannwhitneyu(scoring, checkpoint)
    print("\tStatistics={:.5f}, p={:.5f}".format(stat, p))
    alpha = 0.05
    if p > alpha: print('\t-> Probably the same distribution')
    else:         print('\t-> Probably different distributions')

    #   -------------------------------------------------------------

    # print("\nMedian comparison")
    # medians = [np.median(scoring), np.median(checkpoint)]
    # print("\tScoring {:.2f}".format(medians[0] * 100))
    # print("\tCheckpoint {:.2f}".format(medians[1] * 100))


# to output the results to a file:
# python results.py > ../Data/Results/Results.txt
def main(argv):

    penaltyTest()
    print("\n-------------------------------------------\n")
    counterattackTest()
    print("\n-------------------------------------------\n")
    iniestaTest()


if __name__ == "__main__" :
    main(sys.argv[1:])