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
    u_c = np.array(rawRatings[person - 1])
    others = []
    for user in range(NUM_USERS):
        if user != person - 1:
            others.append(np.array(rawRatings[user]))

    total = 0
    for oUser in others:
        total += abs(cosine_sim(u_c, oUser))

    return 1.0 / total





def coll_weighted_sum(person, jokeID):
    u_c = rawRatings[person]
    k = computeK(person, jokeID)

    simSum = 0
    for user in range(rawRatings.shape[0]):
        if user != person - 1:
            sim = cosine_sim(rawRatings[person - 1], rawRatings[user])

            simSum += sim * rawRatings[user, jokeID - 1]

    return k * simSum






def coll_adjusted_sum(person, jokeID):
    #YASH IS WORKING ON THIS
    userAvg = item_average(person, jokeID)
    k = computeK(person, jokeID)
    total = 0
    sim = 0
    for user in range(rawRatings.shape[0]):
        if (user != person - 1):
            sim = cosine_sim(rawRatings[person - 1], rawRatings[user])

            total += sim * (rawRatings[user ,jokeID - 1] - userAvg)

    adjusted = userAvg + k * total
    return adjusted



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


def computeOtherK(person, jokeId):
    simSum = 0.0
    jokes = np.asarray(np.hsplit(rawRatings, rawRatings.shape[1]))

    for joke in range(jokes.shape[0]):
        if joke != jokeId - 1:
            simSum += abs(cosine_sim(jokes[jokeId - 1],
                                        jokes[joke]))

    return 1/simSum

def item_weighted_sum(person, jokeId):
    simSum = 0.0
    jokes = np.asarray(np.hsplit(rawRatings, rawRatings.shape[1]))
    k = computeOtherK(person, jokeId)

    for joke in range(jokes.shape[0]):
        if joke != jokeId - 1:
            simSum += cosine_sim(jokes[jokeId - 1],
                        jokes[joke]) * \
                        rawRatings[person-1, joke]

    return simSum * k

def item_adjusted_sum(person, jokeId):
    mean = coll_average(person, jokeId)



# Nearest Neighbor Collaborative predictions

# Nearest Neighbor Item-based predictions

userActivity, rawRatings = load_ratings()
print (coll_average(2, 1))
print (rawRatings[2])
print (coll_adjusted_sum(2,20))
result = coll_weighted_sum(2, 20)
print (result, "    ", result + item_average(2, 20))
print (item_weighted_sum(2,20))