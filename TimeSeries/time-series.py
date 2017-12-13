# Nam Phan
# Time Series plots and pair-wise comparisons of states

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.tsa.stattools as sm
import sys

stateList = []


def read_file(filename):
    '''Reads csv file and outputs a dictionary of
    states_year as a key and the counts as a value'''

    with open(filename) as file:
        # Skip header and get column names
        column_names = file.readline().strip().split(',')

        # Column names all lowercase
        column_names = [title.lower() for title in column_names]

        # State and Year index to format key in dictionary
        state_index = column_names.index("state")
        year_index = column_names.index("year")

        # Intialize
        countsDict = {}

        # Iterate through rows
        for line in file:  # each line in file

            # Get all items in row into a list
            line_item_list = line.strip().split(',')

            # Get state as a string
            state = str(line_item_list[state_index])
            # Year as a string
            year = str(line_item_list[year_index])

            # Save in state_year column
            state_year = state + "," + year

            # Save counts as key for specific state_year
            # Count will be last column
            countsDict[state_year] = int(line_item_list[-1])

    return countsDict


def get_state_data(state, countsDict):
    ''' takes input of dictionary with states_years and values
    and returns two vectors with years and values'''
    data = []
    yearsData = {}
    for key in countsDict:
        column_names = key.strip().split(',')
        if column_names[0] == state:
            yearsData[column_names[1]] = countsDict[key]

    years = yearsData.keys()
    years.sort()
    for year in years:
        data.append(yearsData[year])

    years = map(int, years)

    return years, data


def is_state(state):
    # checks if a string is a state
    if state in stateList:
        return True
    else:
        return False


def set_states(countsDict):
    # creates a list of states
    global stateList
    for key in countsDict:
        state = key.strip().split(',')[0]
        stateList.append(state)


def DTWDistance(s1, s2):
    # compute the DTW Distance between two equal length vectors
    DTW = {}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def state_series(gunlawCounts, massshootingCounts):
    # time series on a single state
    inputState = ""
    # user inputs state name
    inputNeeded = True
    # check if input is a proper state
    while inputNeeded:
        inputState = raw_input("Please enter a state (full name and capitalized): ")
        if is_state(inputState):
            inputNeeded = False

    gunLawYears, gunLawValues = get_state_data(inputState, gunlawCounts)
    shootingYears, shootingCounts = get_state_data(inputState, massshootingCounts)

    gunsTs = pd.Series(gunLawValues, index=gunLawYears)
    shootingsTs = pd.Series(shootingCounts, index=shootingYears)

    # plot the number of gun laws
    # plot the number of mass shootings
    gunsTs.plot(label="# Gun Laws", title=inputState,
                legend=True)
    shootingsTs.plot(label="# Mass Shootings", legend=True)
    # shot the plots
    plt.show()


def state_pair(gunlawCounts, massshootingCounts):
    # comparison of time series on two states

    # check if both state inputs are valid
    input1Needed = True
    while input1Needed:
        inputState1 = raw_input("Please enter the first state (full name and capitalized): ")
        if is_state(inputState1):
            input1Needed = False
    input2Needed = True

    while input2Needed:
        inputState2 = raw_input("Please enter the second state (full name and capitalized): ")
        if is_state(inputState2):
            input2Needed = False

    # get vectors of years, counts, for laws and shootings
    # for both states
    gunLawYears1, gunLawCounts1 = get_state_data(inputState1, gunlawCounts)
    shootingYears1, shootingCounts1 = get_state_data(inputState1, massshootingCounts)
    gunLawYears2, gunLawCounts2 = get_state_data(inputState2, gunlawCounts)
    shootingYears2, shootingCounts2 = get_state_data(inputState2, massshootingCounts)

    # create time series for gun laws and shootings for state 1
    gunsTs1 = pd.Series(gunLawCounts1, index=gunLawYears1)
    shootingsTs1 = pd.Series(shootingCounts1, index=shootingYears1)

    # create time series for gun laws and shootings for state 2
    gunsTs2 = pd.Series(gunLawCounts2, index=gunLawYears2)
    shootingsTs2 = pd.Series(shootingCounts2, index=shootingYears2)

    # plot first state's info
    gunsTs1.plot(label=(inputState1 + " " + "# Gun Laws"),
                 title=(inputState1 + " - " + inputState2), legend=True)
    shootingsTs1.plot(label=(inputState1 + " # Mass Shootings"), legend=True)

    # plot second state's info
    gunsTs2.plot(label=(inputState2 + " " + "# Gun Laws"),
                 title=(inputState1 + " - " + inputState2), legend=True)
    shootingsTs2.plot(label=(inputState2 + " # Mass Shootings"), legend=True)

    # compute the DWT Distance for the # of gun laws
    # and # of mass shootings
    print("DWT Distance of number of laws is " + str(DTWDistance(gunLawCounts1, gunLawCounts2))
          + " over " + str(len(gunLawYears1)) + " years.")
    print("DWT Distance of mass shooting counts is " + str(DTWDistance(shootingCounts1, shootingCounts2))
          + " over " + str(len(shootingCounts1)) + " years.")

    # perform a granger casualty test of the first state vs second state
    # with maximum lag of 2 years. Testing to see if changes in gun laws
    # of second state affected gun laws in first state
    sm.grangercausalitytests(np.column_stack((gunLawCounts1, gunLawCounts2)), maxlag=2
                             , verbose=True)
    # show the plots
    plt.show()



if __name__ == '__main__':

    # get data from the main datasets
    gunlaw_counts_file = "../Datasets/num_gun_laws_by_state_per_year.csv"
    massshooting_counts_file = "../Datasets/mass_shootings_2014-2017.csv"
    # gunlaw_counts_file = "num_gun_laws_by_state_per_year.csv"
    # massshooting_counts_file = "mass_shootings_2014-2017.csv"
    gunlawCounts = read_file(gunlaw_counts_file)
    massshootingCounts = read_file(massshooting_counts_file)

    # create the global list of states
    set_states(massshootingCounts)
    # pick whether to view a time series of a single state
    # or a pair-wise comparison
    ansTs = True
    while ansTs:
        print ("""
            1.Single State
            2.State-Pair
            3.Exit/Quit
            """)
        ansTs = raw_input("Choose an option: ")
        if ansTs == "1":
            state_series(gunlawCounts, massshootingCounts)
            ansTs = False
        elif ansTs == "2":
            state_pair(gunlawCounts, massshootingCounts)
            ansTs = False
        elif ansTs == "3":
            sys.exit()
        elif ansTs != "":
            print("\n Please select an option from 1-3")


