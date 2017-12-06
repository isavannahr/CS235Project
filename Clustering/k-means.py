#Isavannah Reyes
#K-means of mass shooting and gun law data

import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MAX_ITERATIONS = 20

#Pre: String, int
#Post: Dict
def read_file(filename, sheet = 1):
    '''Reads a specific sheet of an excel file and outputs a dictionary of
    states_year as a key and the counts as a value'''

    #File must be in excel file
    assert filename.endswith(".xlsx") , "File must be an Excel file (.xlsx)!!!!"

    #Intialize
    countsDict = {}

    #Open specific sheet specified
    xl_workbook = xlrd.open_workbook(filename)
    xl_sheet = xl_workbook.sheet_by_index(sheet - 1)

    #Column names all lowercase
    column_names = [title.lower() for title in (xl_sheet.row_values(0, 0, 3))]

    #Stat and Year index to format key in dictionary
    state_index = column_names.index("state")
    year_index = column_names.index("year")

    #Iterate through rows
    for row in range(1, xl_sheet.nrows):

        #Get state as a string
        state = str(xl_sheet.row_values(row,state_index,state_index+1)[0])
        #Year as a string
        year = str(xl_sheet.row_values(row, year_index,year_index+1)[0])[0:4]

        #Save in state_year column
        state_year = state + "_" + year

        #Save counts as key for specific state_year
        countsDict[state_year] = xl_sheet.row_values(row, 2)[0]

    return countsDict

#Pre: Dict, Dict
#Post: Dict, state_year: [gun law counts, mass shooting counts]
def merge_data(gunlawCounts_perStateYearDict, massshootingCounts_perStateYearDict):
    '''Merges two dictionarys into one by state_year key, value from two dictionarys become a list of counts.
    The first is gun law counts and the second is mass shooting counts'''
    state_yearDict = {}
    for key in gunlawCounts_perStateYearDict:
        state_yearDict[key] = [gunlawCounts_perStateYearDict[key], massshootingCounts_perStateYearDict[key]]
    return state_yearDict

#Pre: numpy array, int:
#Post: numpyarray: randomized centroid points with k rows
def randomize_centroids(data, k):
    """returns k randomized centroids from the initial points"""

    centroids = data.copy()

    #Shuffle data randomly
    np.random.shuffle(centroids)
    #Return top k rows of randomized data
    return centroids[:k]

#Pre: numpy array, numpy array, iterations
#Post: boolean, should k-means stop
def shouldStop(oldCentroids, centroids, iterations):
    '''Returns True or False if kmeans iterations should stop'''

    if np.array_equal(oldCentroids, centroids):
        return True
    elif iterations > MAX_ITERATIONS:
        return True
    else:
        return False

#Pre: nparray, k: [[gun_laws, mass_shootings],[x,y]...]
#Post: nparray k rows 2 columns of centroid points
def k_means(count_xy ,k):
    "Uses mass shooting and gun law counts to cluster the data into k partitions"

    #Intialize k random centroids to represent each cluster
    centroids = randomize_centroids(count_xy,k)
    plot_initial_centroids(count_xy, centroids)

    # Initialize
    iterations = 0
    oldCentroids = np.zeros((k,2))

    #Repeat until cluster points dont change
    while not shouldStop(oldCentroids, centroids, iterations):

        oldCentroids = centroids
        iterations += 1

        #Extend centroids an extra dimension, allows for efficient operation using numpy broadcasting
        #Essentially going to find the closest distance to centroids by perferming operatons on whole operations
        centroids_extended =  centroids[:,np.newaxis]

        #Euclidan Distance

        diff = count_xy - centroids_extended

        #Get distance for all points and centroid
        #[[point1 distance to centroid1, point2 distance to centroid1,...]
        #[point1 distance to centroid2, point2 distance to centroid2,...],...]
        all_distances = np.sqrt(((diff)**2).sum(axis=2))

        #Get the nearest centroid for each point (index of centroids)
        #ie find the smallest value in each column
        nearest_centroid = np.argmin(all_distances, axis=0)


        # Assign new centroids based on which centroid data points are close to
        #For each centroid get the mean of the points closest to that centroid and assign that to the new centroid
        centroids =  np.array([count_xy[nearest_centroid == i].mean(axis=0) for i in range(centroids.shape[0])])

    print("Clutering finished in " + str(iterations) + " iterations.")
    return centroids, nearest_centroid

#####PLOTTING FUNCTIONS#########

def plot(data,final_centroids, nearest_centroid ):
    plt.scatter(data[:, 0], data[:, 1], c = nearest_centroid, cmap="tab10")
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='r', s=100)
    plt.savefig("finalclusters_two.png")
    #plt.show()

def plot_initial_centroids(data, centroids):
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
    plt.savefig("initialcentroids_two.png")

    #plt.show()

def initial_plot_two_clusters(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    ax = plt.gca()
    ax.add_patch(mpatches.Circle([17, 8], radius = 4, fill=False, lw=3, color= "r"))
    ax.add_patch(mpatches.Circle([78, 16], radius = 4, fill=False, lw=3, color= "r"))
    plt.savefig("twocentroidguess.png")
    #plt.show()


def initial_plot_three_clusters(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    ax = plt.gca()
    ax.add_patch(mpatches.Circle([17, 8], radius=4, fill=False, lw=3, color="r"))
    ax.add_patch(mpatches.Circle([83, 34], radius=4, fill=False, lw=3, color="r"))
    ax.add_patch(mpatches.Circle([74, 7], radius=4, fill=False, lw=3, color="r"))
    plt.savefig("threecentroidguess.png")

    #plt.show()

def initial_plot(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    plt.savefig("IntitialPlot.png")

    #plt.show()


if __name__ == '__main__':

    #Files
    gunlaw_counts_file = "NumGunLawsByState_Cleaned.xlsx"
    massshooting_counts_file = "MassShootings_Cleaned.xlsx"

    #Read Files into Dictionary
    gunlawCounts_perStateYearDict = read_file(gunlaw_counts_file, 1)
    massshootingCounts_perStateYearDict = read_file(massshooting_counts_file, 1)

    #Merge two dictionaries into one by key
    merged_state_yearDict = merge_data(gunlawCounts_perStateYearDict, massshootingCounts_perStateYearDict)

    #Get data without state_year lablels for clustering
    count_xy = np.array(list(merged_state_yearDict.values()))

    #Plot to pick k
    #initial_plot(count_xy)

    #Visualize where you think clusters are
    #initial_plot_two_clusters(count_xy)
    #initial_plot_three_clusters(count_xy)

    #2 clusters
    final_centroids, nearest_centroid = k_means(count_xy, 2)
    plot(count_xy, final_centroids, nearest_centroid)

    # 3 clusters
    # k_means(count_xy, 3)


