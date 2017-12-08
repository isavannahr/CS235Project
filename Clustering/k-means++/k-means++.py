#Isavannah Reyes
#K-means++ partitioning of mass shooting and gun law data

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
    for key in gunlawCounts_perStateYearDict:
        count_xy.append([gunlawCounts_perStateYearDict[key], massshootingCounts_perStateYearDict[key]])

    return np.array(count_xy)

#Pre: numpy array
#Post: numpyarray: randomized centroid point
def randomize_first_centroid(data):
    """returns k randomized centroids from the initial points"""

    centroids = data.copy()

    #Shuffle data randomly
    np.random.shuffle(centroids)

    #Return first row of randomized data
    return centroids[:1]

def initialize_centroids(count_xy, k):
    # Intialize 1 random centroid and declare it the first centroid
    centroids = randomize_first_centroid(count_xy)

    # Extend centroid an extra dimension, to allow for efficient operation using numpy broadcasting
    # Essentially going to find the closest distance to centroids by perferming operatons on whole operations
    centroids = centroids[:, np.newaxis]

    # Initialize
    i = 1
    while i < k:
        i += 1

        # Distance
        diff = count_xy - centroids
        # Get distance between all points and centroid one
        all_distances = np.sqrt(((diff) ** 2).sum(axis=2))

        # Square Distances
        all_distances_sq = np.square(all_distances)

        # get minimum distance from all centroids
        all_distances_sq = np.amin(all_distances_sq, axis=0)

        # Select a Random point with probability propotional to the square distance and set it as the next centroid
        probability = all_distances_sq / all_distances_sq.sum()

        # Get cumulative probability of choosing all points
        # Returns an array from [0,10
        cumulative_prob = probability.cumsum()

        # Choose random value from [0,1]
        r = random.random()

        # Find the first index of the point correspnding to r in cumulative prob
        # This is our new random centroid
        ind = np.where(cumulative_prob >= r)[0][0]
        new_centroid = np.array([[count_xy[ind]]])

        # Get new centroid
        centroids = np.append(centroids, new_centroid, axis=0)

    return centroids

#Pre: numpy array, numpy array, iterations
#Post: boolean, should k-means stop
def shouldStop(oldCentroids, centroids, iterations):
    '''Returns True or False if kmeans iterations should stop
    based on if centroids have changed or not'''

    if np.array_equal(oldCentroids, centroids):
        return True
    elif iterations > MAX_ITERATIONS:
        return True
    else:
        return False

#Pre: nparray, k: [[gun_laws, mass_shootings],[x,y]...]
#Post: nparray k rows 2 columns of centroid points
def k_means_pp(count_xy ,k, plot_name):
    '''Uses mass shooting and gun law counts to cluster the data
    into k partitions'''

    centroids = initialize_centroids(count_xy,k)
    #plot_initial_centroids(count_xy, centroids, plot_name)

    # Initialize
    iterations = 0
    oldCentroids = np.zeros((k, 2))

    # Repeat until cluster points dont change
    while not shouldStop(oldCentroids, centroids, iterations):
        oldCentroids = centroids
        iterations += 1

        diff = count_xy - centroids

        # Get distance for all points and centroid
        # [[point1 distance to centroid1, point2 distance to centroid1,...]
        # [point1 distance to centroid2, point2 distance to centroid2,...],...]
        all_distances = np.sqrt(((diff) ** 2).sum(axis=2))

        # Get the nearest centroid for each point (index of centroids)
        # ie find the smallest value in each column
        nearest_centroid = np.argmin(all_distances, axis=0)

        # Assign new centroids based on which centroid data points are close to
        # For each centroid get the mean of the points closest to that centroid and assign that to the new centroid
        centroids = np.array([count_xy[nearest_centroid == i].mean(axis=0) for i in range(centroids.shape[0])])
        #Add axis for broadcasting
        centroids = centroids[:, np.newaxis]

    print("Clutering finished in " + str(iterations) + " iterations.")

    return centroids, nearest_centroid


#####PLOTTING FUNCTIONS#########

def plot_initial_centroids(data, final_centroids,  plot_name):
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    plt.scatter(final_centroids[0:, 0][:,0], final_centroids[0:, 0][:,1], c='r', s=100)
    #plt.savefig(plot_name)
    #plt.close()
    plt.show()

def plot(data,final_centroids, nearest_centroid, plot_name):
    plt.scatter(data[:, 0], data[:, 1], c = nearest_centroid, cmap="tab10")
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    plt.scatter(final_centroids[0:, 0][:,0], final_centroids[0:, 0][:,1], c='r', s=100)
    #plt.savefig(plot_name)
    plt.show()
    #plt.close()

if __name__ == '__main__':

    #Files
    gunlaw_counts_file = "../../ToyDatasets/num_gun_laws_by_state_per_year_2016.csv"
    massshooting_counts_file = "../../ToyDatasets/mass_shootings_2016.csv"

    #Read Files into Dictionary
    gunlawCounts_perStateYearDict = read_file(gunlaw_counts_file)
    massshootingCounts_perStateYearDict = read_file(massshooting_counts_file)

    #Merge two dictionaries into one by key
    #Get data without state_year lablels for clustering
    count_xy = merge_data(gunlawCounts_perStateYearDict, massshootingCounts_perStateYearDict)

    #Plot to pick k
    #initial_plot(count_xy)

    #Visualize where you think clusters are
    #initial_plot_two_clusters(count_xy)
    #initial_plot_three_clusters(count_xy)

    #2 clusters
    final_centroids, nearest_centroid = k_means_pp(count_xy, 2,"Plots/initialcentroids_two_2016.png")
    plot(count_xy, final_centroids, nearest_centroid, "Plots/finalcentroids_two_2016.png")

    # 3 clusters
    #final_centroids, nearest_centroid = k_means_pp(count_xy, 3,"Plots/initialcentroids_three_2016.png")
    #plot(count_xy, final_centroids, nearest_centroid,"Plots/finalcentroids_three_2016.png")


