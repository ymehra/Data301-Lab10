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

        for row in range(rawRatings.shape[0]):
            for col in range(rawRatings.shape[1]):
                if rawRatings[row, col] == 99:
                    rawRatings[row, col] = 0

        return userActivity, rawRatings

def cosine_sim(ratings1, ratings2):
    xSqSum = 0.0
    for r in range(ratings1.shape[0]):
        xSqSum += float(ratings1[r] ** 2)
    xSqSum = xSqSum ** .5

    ySqSum = 0.0
    for r in range(ratings2.shape[0]):
        ySqSum += float(ratings2[r] ** 2)
    ySqSum = ySqSum ** .5

    xy = xSqSum * ySqSum
    xySum = 0.0
    for r in range(ratings1.shape[0]):
        xySum += float(ratings1[r] * ratings2[r])

    return float(xySum)/float(xy)




# Collaborative predictions

# mean utility of a joke for all users who rated it
def coll_average(person, jokeId):
    # get all ratings for jokeId
    ratings = []
    for row in range(rawRatings.shape[0]):
        ratings.append(rawRatings[row, jokeId-1])
    count = 0
    rSum = 0
    ratings  = np.asarray(ratings)
    for i in range(ratings.shape[0]):
        # if valid rating and not person in question's rating
        if ratings[i] != 0 and i != person+1:
            rSum += ratings[i]
            count += 1

    return rSum/count


def computeK(person, jokeID):
    u_c = np.array(rawRatings[person])
    others = []
    for user in range(NUM_USERS):
        if user != person:
            others.append(np.array(rawRatings[user]))

    total = 0
    for oUser in others:
        total += math.fabs(cosine_sim(u_c, oUser))

    return 1 / total





def coll_weighted_sum(person, jokeID):
    u_c = rawRatings[person]
    k = computeK(person, jokeID)
    





# Item-based predictions

# avg rating a user gave
def item_average(person, jokeId):
    # get all ratings from this user
    ratings = []
    for joke in range(rawRatings.shape[1]):
        ratings.append(rawRatings[person-1, joke])
    rSum = 0
    count = 0
    ratings = np.asarray(ratings)
    for i in range(ratings.shape[0]):
        if ratings[i] != 0:
            rSum += ratings[i]
            count += 1

    return rSum / count

def item_weighted_sum(person, jokeId):
    k = 0.0
    absSimSum = 0.0
    simSum = 0.0

    for joke in range(rawRatings.shape[1]):
        if joke != jokeId - 1:
            absSimSum += abs(cosine_sim(rawRatings[jokeId - 1],
                                     rawRatings[joke]))
            simSum += cosine_sim(rawRatings[jokeId - 1],
                        rawRatings[joke]) * \
                        rawRatings[person-1, joke]

    k = 1 / absSimSum

    return simSum * k

# Nearest Neighbor Collaborative predictions

# Nearest Neighbor Item-based predictions

userActivity, rawRatings = load_ratings()
print (coll_average(1, 1))