#below are imports
import numpy as np
import re
import math

# import ratings
ratingsFile = "/home/dekhtyar/data/jester-data-1.csv"

## constants
NUM_USERS = 24983   ## number of users in the dataset
NUM_JOKES = 100     ## number of jokes in the dataset

EPSILON = 0.0001    ## useful for assertions over floats

rawRatingsTable = []  ## use rawRatings variable to store the NumPy array of
                 ## ratings from the data file
userActivity = []  ## put the first column of the rawRatingsTable array here
rawRatings = []  ## put the rest of the rawRatingsTable array here

### loading the array of ratings
def load_ratings():
    with open(ratingsFile, 'r') as file:
        rows = []

        for line in file:
            ratings = line.split(",")
            rows.append(np.asarray(ratings, dtype = 'float'))

        rawRatingsTable = np.asarray(rows)

        # SPLIT RAW RATINGS MATRIX
        userActivityList = []
        rawRatingsList = []
        for row in range(rawRatingsTable.shape[0]):
            userActivityList.append(rawRatingsTable[row, 0])
            rawRatingsList.append(rawRatingsTable[row, 1:])

        userActivity = np.asarray(userActivityList)
        rawRatings = np.asarray(rawRatingsList)


# mean utility of a joke for all users who rated it
def mean_utility(person, jokeId):
    # get all ratings for jokeId
    ratings = rawRatings[:, jokeId-1]

    for i in range(ratings.shape[0]):
        if ratings[i] != 99:
            rSum += ratings[i]
            count += 1

    return rSum/count

# Collaborative predictions

# Item-based predictions

# Nearest Neighbor Collaborative predictions

# Nearest Neighbor Item-based predictions

