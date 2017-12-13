#Isavannah Reyes
#K-means++ partitioning of mass shooting and gun law data

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

#Isavannah Reyes
#K-medoids partitioning of mass shooting and gun law data

import numpy as np
import matplotlib.pyplot as plt
import random

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

def get_nonmedoids(count_xy, medoids):
    isNonMedoid = np.ones((count_xy.shape[0]), dtype=bool)
    i = 0
    medoids_found = 0
    while medoids_found != len(medoids):
        if count_xy[i, :] in medoids:
            isNonMedoid[i] = False
            medoids_found += 1
        i += 1
    non_medoids = count_xy[isNonMedoid]
    return non_medoids


#Pre: numpy array, numpy array, iterations
#Post: boolean, should k-means stop
def shouldStop(oldmedoids, medoids, iterations):
    '''Returns True or False based on if new medoids dont reduce cost or do'''

    if np.array_equal(oldmedoids, medoids):
        return True
    elif iterations > MAX_ITERATIONS:
        return True
    else:
        return False

# Pre: numpy array, int:
# Post: numpyarray: randomized medoid points with k rows
def randomize_medoids(data, k):
    """returns k randomized medoids from the initial points"""

    medoids = data.copy()

    # Shuffle data randomly
    np.random.shuffle(medoids)
    # Return top k rows of randomized data
    return medoids[:k]

def calculate_cost(nearest_medoid, dist, m ):

    cost = 0
    for i in range(m.shape[0]):
        x = dist[i, nearest_medoid == i]
        cost += x.sum(axis=0)
    return cost

def mask(maskee, check):
        check = check[np.newaxis]
        isInMaskee = np.ones(maskee.shape[0], dtype=bool)
        i = 0
        checks_found = 0
        while checks_found != len(check):
            if maskee[i, :] in check:
                isInMaskee[i] = False
                checks_found += 1
            i += 1
        return isInMaskee

#Pre: nparray, k: [[gun_laws, mass_shootings],[x,y]...]
#Post: nparray k rows 2 columns of centroid points
def k_medoids(count_xy ,k):

    '''

    PAM algorithm

    Uses mass shooting and gun law counts to cluster the data
    into k partitions'''

    #Initially randomize medoids
    medoids = randomize_medoids(count_xy,k)

    # Initialize
    iterations = 0
    oldmedoids = np.zeros((k, 2))
    cost_decreases = True

    #Initially get cost and distances of initial medoid assignments
    # Broadcasting
    medoids_extended = medoids[:, np.newaxis]
    diff = count_xy - medoids_extended
    # Get distance for all points and medoids
    # [[point1 distance to centroid1, point2 distance to centroid1,...]
    # [point1 distance to centroid2, point2 distance to centroid2,...],...]
    all_distances = np.sqrt(((diff) ** 2).sum(axis=2))
    # Get nearest medoid assignments
    nearest_medoid = np.argmin(all_distances, axis=0)
    # Calculate lowest known cost which is initial
    cost = calculate_cost(nearest_medoid, all_distances, medoids)

    while cost_decreases == True:
        best_medoids = medoids
        lowest_cost = cost
        current_nearest_medoids = nearest_medoid
        for medoid in medoids:
#         #Get non_medoids
            non_medoids = get_nonmedoids(count_xy, medoids)
            for non_medoid in non_medoids:
                #Swap non-medoid with medoid
                current_medoids = medoids.copy()
                #Swap False with the new non-medoids
                current_medoids[~mask(medoids,medoid)] = non_medoid

                # Broadcasting
                current_medoids_extended = current_medoids[:, np.newaxis]
                diff = count_xy - current_medoids_extended
                # Get distance for all points and medoids
                # [[point1 distance to centroid1, point2 distance to centroid1,...]
                # [point1 distance to centroid2, point2 distance to centroid2,...],...]
                all_distances = np.sqrt(((diff) ** 2).sum(axis=2))
                # Get nearest medoid assignments
                new_nearest_medoids = np.argmin(all_distances, axis=0)
                # Calculate lowest known cost which is initial
                current_cost = calculate_cost(nearest_medoid, all_distances, medoids)
                # If the swap gives us a lower cost we save the medoids and cost
                if current_cost < lowest_cost:
                    lowest_cost = current_cost
                    best_medoids = current_medoids
                    current_nearest_medoids = new_nearest_medoids
              # If there was a swap that resultet in a lower cost we save the
              # resulting medoids from the best swap and the new cost
        if lowest_cost < cost:
            cost = lowest_cost
            medoids = best_medoids
            nearest_medoid = current_nearest_medoids
        else:
            break

    return medoids, nearest_medoid

def plot_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    title = kwargs.pop('title', None)

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

    lawtype= ['domestic_violence', 'preemption', 'immunity', 'dealer_regulations', 'prohibitions_against_high_risk_gun_owners', 'background_checks', 'ammunition_regulations', 'child_access_prevention', 'concealed_carry_permitting', 'assault_weapons_regulations', 'gun_trafficking', 'no_stand_your_ground', 'buyer_regulations', 'possession_regulations']

    # for law in lawtype:
c    #         # generate the linkage matrix - determine distances and merger clusters that have the smallest distance
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


    count_xy, state_yrs = merge_data(gunlawTypeCountspercolumn_perStateYearDict["ammunition_regulations"],
                                         massshootingCounts_perStateYearDict)
        # generate the linkage matrix - determine distances and merger clusters that have the smallest distance
        # each indice is the clusters at that iteration
        # ward = ward variance minimization problem
        # minimize intra cluster variance in eculidian space (similar to k-means)

    c = pdist(count_xy, "jaccard")
    Z = linkage(count_xy, 'complete')

    max_d = 15
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

    clusters = fcluster(Z, max_d, criterion='distance')
        # print(clusters)#agrees with three clusters

    plt.figure(figsize=(10, 8))
    plt.scatter(count_xy[:, 0], count_xy[:, 1], c=clusters,
                    cmap='tab10')  # plot points with cluster dependent colors
    plt.title("Clusters based on Hierarchical Clustering")
    plt.xlabel("Number of Ammunition Regulation Laws")
    plt.ylabel("Number of Mass Shootings")

    plt.show()

    max_d = 21
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

    clusters = fcluster(Z, max_d, criterion='distance')
        # print(clusters)#agrees with three clusters

    plt.figure(figsize=(10, 8))
    plt.scatter(count_xy[:, 0], count_xy[:, 1], c=clusters,
                    cmap='tab10')  # plot points with cluster dependent colors
    plt.title("Clusters based on Hierarchical Clustering")
    plt.xlabel("Number of Concealed Carry Permitting Laws")
    plt.ylabel("Number of Mass Shootings")

    plt.show()