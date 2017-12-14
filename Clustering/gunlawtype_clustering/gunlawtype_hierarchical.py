#Isavannah Reyes
#Hierarchical clustering of specific gun law types versus mass shooting events
import numpy as np
import matplotlib.pyplot as plt
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

MAX_ITERATIONS = 20

#Pre: String
#Post: Dict
def read_file(filename):
    '''Reads csv file and outputs a dictionary of
    states_year as a key and the counts as a value'''

    with open(filename) as file:
        #Skip header and get column names
        column_names = file.readline().strip().split(',')

        # Column names all lowercase
        column_names = [title.lower() for title in column_names]

        #State and Year index to format key in dictionary
        state_index = column_names.index("state")
        year_index = column_names.index("year")

        # Intialize
        countsDict = {}

        # Iterate through rows
        for line in file:  # each line in file

            #Get all items in row into a list
            line_item_list = line.strip().split(',')

            #Get state as a string
            state = str(line_item_list[state_index])
            #Year as a string
            year = str(line_item_list[year_index])

            #Save in state_year column
            state_year = state + "_" + year

            # Save counts as key for specific state_year
            #Count will be last column
            countsDict[state_year] = int(line_item_list[-1])

    return countsDict

#Pre: String
#Post: Dict
def read_file_mc(filename):
    '''Reads csv file and outputs a dictionary of
    states_year as a key and the counts as a value'''

    with open(filename) as file:
        #Skip header and get column names
        column_names = file.readline().strip().split(',')

        # Column names all lowercase
        column_names = [title.lower() for title in column_names]

        #State and Year index to format key in dictionary
        state_index = column_names.index("state")
        year_index = column_names.index("year")
        gunLawTypeDict = {}
        lines = file.readlines()
        # Intialize
        #For each column
        for i in range(2,len(column_names)-1):
            # Iterate through rows
            countsDict = {}
            for line in lines:  # each line in file

                #Get all items in row into a list
                line_item_list = line.strip().split(',')

                #Get state as a string
                state = str(line_item_list[state_index])
                #Year as a string
                year = str(line_item_list[year_index])

                #Save in state_year column
                state_year = state + "_" + year

                # Save counts as key for specific state_year
                #Count will be last column
                countsDict[state_year] = int(line_item_list[i])
            gunLawTypeDict[column_names[i]] = countsDict

    return gunLawTypeDict

#Pre: Dict, Dict
#Post:numpy array, [[gun law counts, mass shooting counts]]
def merge_data(gunlawCounts_perStateYearDict, massshootingCounts_perStateYearDict):
    '''Merges two dictionarys into one by state_year key, value from two dictionarys become a list of counts.
    The first is gun law counts and the second is mass shooting counts'''

    count_xy = []
    labels = []
    for key in gunlawCounts_perStateYearDict:
        count_xy.append([gunlawCounts_perStateYearDict[key], massshootingCounts_perStateYearDict[key]])
        labels.append(key)

    return np.array(count_xy),labels

def plot_dendrogram(*args, **kwargs):
    '''Plots Dendogram'''

    #Maxd and title of plot
    max_d = kwargs.pop('max_d', None)
    title = kwargs.pop('title', None)

    #Clusters
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title(title)
        plt.xlabel('States per Year')
        # From method above ("ward")
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')

    return ddata

if __name__ == '__main__':

    #Options
    np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

    #Files
    gunlaw_counts_file = "../../ToyDatasets/num_gun_laws_by_state_per_year_2016.csv"
    massshooting_counts_file = "../../ToyDatasets/mass_shootings_2016.csv"

    #Read Files into Dictionary
    gunlawTypeCountspercolumn_perStateYearDict = read_file_mc(gunlaw_counts_file)
    massshootingCounts_perStateYearDict = read_file(massshooting_counts_file)

    #Cluster ammunition regulations
    count_xy, state_yrs = merge_data(gunlawTypeCountspercolumn_perStateYearDict["ammunition_regulations"],
                                     massshootingCounts_perStateYearDict)

    # generate the linkage matrix - determine distances and merger clusters that have the smallest distance
    # each indice is the clusters at that iteration

    c = pdist(count_xy, "jaccard")
    Z = linkage(count_xy, 'complete')

    max_d = 10
    plot_dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,
        labels=state_yrs,
        max_d=max_d,  # plot a horizontal cut-off line
        title="Dendogram of Number of Ammunition Regulation Laws vs Number of Mass Shootings"
    )

    #Clusters
    clusters = fcluster(Z, max_d, criterion='distance')
    # print(clusters)#agrees with three clusters
    #Plot according to dendogram clusters
    plt.figure(figsize=(10, 8))
    plt.scatter(count_xy[:, 0], count_xy[:, 1], c=clusters,
                cmap='tab10')  # plot points with cluster dependent colors
    plt.title("Clusters based on Hierarchical Clustering")
    plt.xlabel("Number of Ammunition Regulation Laws")
    plt.ylabel("Number of Mass Shootings")
    plt.show()

    #Cluster concealed_carry_permitting laws
    count_xy, state_yrs = merge_data(gunlawTypeCountspercolumn_perStateYearDict["concealed_carry_permitting"],
                                     massshootingCounts_perStateYearDict)

    c = pdist(count_xy, "jaccard")
    Z = linkage(count_xy, 'complete')

    max_d = 10
    plot_dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,
        labels=state_yrs,
        max_d=max_d,  # plot a horizontal cut-off line
        title="Dendogram of Number of Concealed Carry Permitting Laws vs Number of Mass Shootings"
    )
    #Clusters
    clusters = fcluster(Z, max_d, criterion='distance')
    #Plot according to dendorgram clusters
    plt.figure(figsize=(10, 8))
    plt.scatter(count_xy[:, 0], count_xy[:, 1], c=clusters,
                cmap='tab10')  # plot points with cluster dependent colors
    plt.title("Clusters based on Hierarchical Clustering")
    plt.xlabel("Number of Concealed Carry Permitting Laws")
    plt.ylabel("Number of Mass Shootings")

    plt.show()

    #Plot all gun law type dendograms and clusters
    #lawtype= ['domestic_violence', 'preemption', 'immunity', 'dealer_regulations', 'prohibitions_against_high_risk_gun_owners', 'background_checks', 'ammunition_regulations', 'child_access_prevention', 'concealed_carry_permitting', 'assault_weapons_regulations', 'gun_trafficking', 'no_stand_your_ground', 'buyer_regulations', 'possession_regulations']

    # for law in lawtype:
    #         # generate the linkage matrix - determine distances and merger clusters that have the smallest distance
    #         # each indice is the clusters at that iteration
    #         # ward = ward variance minimization problem
    #         # minimize intra cluster variance in eculidian space (similar to k-means)
    #
    #         c = pdist(count_xy, "jaccard")
    #         Z = linkage(count_xy, 'complete')
    #
    #         max_d = 24
    #         plot_dendrogram(
    #             Z,
    #             truncate_mode='lastp',
    #             p=12,
    #             leaf_rotation=90.,
    #             leaf_font_size=12.,
    #             show_contracted=True,
    #             annotate_above=10,
    #             labels=state_yrs,
    #             max_d=max_d,  # plot a horizontal cut-off line
    #             title = "Dendogram of Number of " + law+ " Laws vs Number of Mass Shootings"
    #         )
    #
    #         clusters = fcluster(Z, max_d, criterion='distance')
    #         # print(clusters)#agrees with three clusters
    #
    #         plt.figure(figsize=(10, 8))
    #         plt.scatter(count_xy[:, 0], count_xy[:, 1], c=clusters,
    #                     cmap='tab10')  # plot points with cluster dependent colors
    #         plt.title("Clusters based on Hierarchical Clustering")
    #         plt.xlabel("Number of " +law+" Laws")
    #         plt.ylabel("Number of Mass Shootings")
    #
    #         plt.show()


