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
    jokes = []
    for col in range(rawRatings.shape[1]):
        jokes.append(rawRatings[:,col])
    jokes = np.asarray(jokes)
    k = computeOtherK(person, jokeId)

    for joke in range(jokes.shape[0]):
        if joke != jokeId - 1:
            simSum += cosine_sim(jokes[jokeId - 1],
                        jokes[joke]) * \
                        rawRatings[person-1, joke]

    return simSum * k



def item_adjusted_sum(person, jokeId):
    userAvg = coll_average(person, jokeId)
    k = computeOtherK(person, jokeId)
    total = 0.0
    jokes = []
    for col in range(rawRatings.shape[1]):
        jokes.append(rawRatings[:,col])
    jokes = np.asarray(jokes)
    for joke in range(jokes.shape[0]):
        if (joke != jokeId - 1):
            sim1 = cosine_sim(jokes[jokeId - 1], jokes[joke])
            total += sim1 * (rawRatings[person - 1, joke] - userAvg)

    adjusted = userAvg + k * total
    return adjusted



# Nearest Neighbor Collaborative predictions

# returns list of n nearest user IDs
def nNN_users(n, person):
    sims = []
    neighbors = [] # list of nearest user neighbor IDs and their similarities
    for user in range(rawRatings.shape[0]):
        if user != person - 1:
            sims.append((cosine_sim(rawRatings[person - 1], rawRatings[user]), user))

    # elements are (sim, userID)
    sims.sort()

    for n in range(n):
        neighbors.append(sims[n][1])

    return neighbors

# returns list of n nearest jokeIDs
#def nNN_jokes(n):



def nn_coll_average(person, jokeId):
    sum = 0
    N = 10
    nearestNeighbors = nNN_users(N, person)

    for n in range(len(nearestNeighbors)):
        sum += rawRatings[nearestNeighbors[n], jokeId-1]

    return sum / N



# Nearest Neighbor Item-based predictions
def nn_item_average(person, jokeId):
    sum = 0
    N = 10
    nearestNeighbors = nNN_jokes(N)

    for n in range(len(nearestNeighbors)):
        sum += rawRatings[person-1, nearestNeighbors[n]]

    return sum / N




userActivity, rawRatings = load_ratings()
#print (coll_average(2, 20))
#print (item_average(2, 20))
#print (coll_weighted_sum(2,20))
#print (coll_adjusted_sum(2,20))
#print (item_weighted_sum(2,20))
#print (item_adjusted_sum(2,20))
print (rawRatings[1, 19])
print (nn_coll_average(2, 20))



def reserved_set():
    users = np.random.choice(rawRatings.shape[0], 3, False)
    jokes = np.random.choice(rawRatings.shape[1], 3, False)

    for i in range(len(users)):
        print("Real rating: " + rawRatings[users[i], jokes[i]])
        # Do we need to remove rating from rawRatings?
        print("Collaborated mean utility: %.11f" % coll_average(users[i], jokes[i]))
        print("Collaborated weighted sum: %.11f" % coll_weighted_sum(users[i], jokes[i]))
        print("Collaborated adjusted weighted sum: %.11f" % coll_adjusted_sum(users[i], jokes[i]))
        print("Item-based mean utility: %.11f" % item_average(users[i], jokes[i]))
        print("Item-based weighted sum: %.11f" % item_weighted_sum(users[i], jokes[i]))
        print("Item-based adjusted weighted sum: %.11f" % item_adjusted_sum(users[i], jokes[i]))


