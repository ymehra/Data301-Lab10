#below are imports
import numpy as np
import re
import math

# import ratings
ratingsFile = "jester-data-1.csv"

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


# Collaborative predictions

# mean utility of a joke for all users who rated it
def coll_average(person, jokeId):
    # get all ratings for jokeId
    ratings = rawRatings[:][jokeId-1]
    count = 0

    for i in range(ratings.shape[0]):
        # if valid rating and not person in question's rating
        if ratings[i] != 99 and i != person+1:
            rSum += ratings[i]
            count += 1

    return rSum/count

# Item-based predictions

# avg rating a user gave
def item_average(person, jokeId):
    # get all ratings from this user
    user = rawRatings[person-1]
    count = 0

    for i in range(user.shape[0]):
        if user[i] != 99:
            rSum += user[i]
            count += 1

    return rSum / count

# Nearest Neighbor Collaborative predictions

# Nearest Neighbor Item-based predictions

