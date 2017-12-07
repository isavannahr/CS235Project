#Isavannah Reyes
#K-means++ partitioning of mass shooting and gun law data

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


#Pre: nparray, k: [[gun_laws, mass_shootings],[x,y]...]
#Post: nparray k rows 2 columns of centroid points
def k_means_pp(count_xy ,k):
    '''Uses mass shooting and gun law counts to cluster the data
    into k partitions'''

    #Intialize 1 random centroid and declare it the first centroid
    first_centroid = randomize_first_centroid(count_xy)

    # Extend centroid an extra dimension, to allow for efficient operation using numpy broadcasting
    # Essentially going to find the closest distance to centroids by perferming operatons on whole operations
    first_centroid_extended = first_centroid[:, np.newaxis]

    #Initialize
    i = 1
    while i < k:
        i += 1

        #Distance
        diff = count_xy - first_centroid_extended
        # Get distance between all points and centroid one
        all_distances = np.sqrt(((diff)**2).sum(axis=2))


        #Square Distances
        all_distances_sq = np.square(all_distances)
        print(all_distances_sq)


    ####################
    #oldCentroids = np.zeros((k,2))

    #Repeat until cluster points dont change
    #while not shouldStop(oldCentroids, centroids, iterations):

        #oldCentroids = centroids


        # Get the nearest centroid for each point (index of centroids)
        # ie find the smallest value in each column
        # nearest_centroid = np.argmin(all_distances, axis=0)


        # Assign new centroids based on which centroid data points are close to
        #For each centroid get the mean of the points closest to that centroid and assign that to the new centroid
        #centroids =  np.array([count_xy[nearest_centroid == i].mean(axis=0) for i in range(centroids.shape[0])])

    #return centroids, nearest_centroid
    return 0

#####PLOTTING FUNCTIONS#########

def plot(data,final_centroids, nearest_centroid ):
    plt.scatter(data[:, 0], data[:, 1], c = nearest_centroid, cmap="tab10")
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Number of Mass Shootings")
    plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='r', s=100)
    #plt.savefig("Plots/finalclusters_two_2016.png")
    plt.show()
    #plt.close()

if __name__ == '__main__':

    #Files
    gunlaw_counts_file = "../../Datasets/NumGunLawsByState_Cleaned.xlsx"
    massshooting_counts_file = "../../Datasets/MassShootings_Cleaned.xlsx"

    #Read Files into Dictionary
    gunlawCounts_perStateYearDict = read_file(gunlaw_counts_file, 1)
    massshootingCounts_perStateYearDict = read_file(massshooting_counts_file, 1)

    #Merge two dictionaries into one by key
    #Get data without state_year lablels for clustering
    count_xy = merge_data(gunlawCounts_perStateYearDict, massshootingCounts_perStateYearDict)

    #Plot to pick k
    #initial_plot(count_xy)

    #Visualize where you think clusters are
    #initial_plot_two_clusters(count_xy)
    #initial_plot_three_clusters(count_xy)

    #2 clusters
    final_centroids, nearest_centroid = k_means_pp(count_xy, 2)
    #plot(count_xy, final_centroids, nearest_centroid)

    # 3 clusters
    #final_centroids, nearest_centroid = k_means(count_xy, 3)
    #plot(count_xy, final_centroids, nearest_centroid)


