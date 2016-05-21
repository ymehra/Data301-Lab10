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


## this will create the nparrays like dekhtyar said in lab - we can now use np.operations to speed it up
def commonUsers(ratings1, ratings2):
    ## length of shorter array
    N = min (ratings1.size, ratings2.size)
    rt1 = []
    rt2 = []
    for i in range(N):
        if ratings1[i] != 99 and ratings2[i] != 99:
            rt1.append(ratings1[i])
            rt2.append(ratings2[i])

    return np.array(rt1), np.array(rt2)


def cosine_sim(ratings1, ratings2):
    r1, r2 = commonUsers(ratings1, ratings2)

    xSqSum = float(np.sum(np.square(r1)))
    xSqSum = xSqSum ** .5

    ySqSum = float(np.sum(np.square(r2)))
    ySqSum = ySqSum ** .5

    xy = xSqSum * ySqSum
    xySum = np.sum(r1 * r2)

    return float(xySum)/float(xy)


def pearson_sim(ratings1, ratings2, avg):
    r1, r2 = commonUsers(ratings1, ratings2)

    x = r1 - avg
    y = r2 - avg
    xSq = np.square(x)
    ySq = np.square(y)

    numerator = float(np.sum(x * y))
    denominator = float(np.sqrt(np.sum(xSq) * np.sum(ySq)))

    return numerator / denominator




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

    usrAvg = item_average(person, jokeID)
    simSum = 0
    for user in range(rawRatings.shape[0]):
        if user != person - 1:
            # sim = cosine_sim(rawRatings[person - 1], rawRatings[user])
            sim = pearson_sim(rawRatings[person - 1], rawRatings[user], usrAvg)
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

    jokeAvg = coll_average(person, jokeId)
    for joke in range(jokes.shape[0]):
        if joke != jokeId - 1:
            # simSum += cosine_sim(jokes[jokeId - 1], jokes[joke]) * rawRatings[person-1, joke]
            simSum += pearson_sim(jokes[jokeId - 1], jokes[joke], jokeAvg) * rawRatings[person-1, joke]

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
def nNN_users(n, person, jokeId):
    sims = []
    neighbors = [] # list of nearest user neighbor IDs
    avg = item_average(person, jokeId)
    for user in range(rawRatings.shape[0]):
        if user != person - 1:
            # sims.append((cosine_sim(rawRatings[person - 1], rawRatings[user]), user))
            sims.append((pearson_sim(rawRatings[person - 1], rawRatings[user], avg), user))

    # elements are (sim, userID)
    sims.sort()

    for n in range(n):
        neighbors.append(sims[n])

    return neighbors


# returns list of n nearest jokeIDs
def nNN_jokes(n, person, jokeId):
    neighbors = [] # list of nearest neighbor joke IDs
    sims = []
    avg = coll_average(person, jokeId)
    for joke in range(rawRatings.shape[1]):
        if joke != jokeId - 1:
            # sims.append((cosine_sim(rawRatings[:,jokeId - 1], rawRatings[:,joke]), joke))
            sims.append((pearson_sim(rawRatings[:, jokeId - 1], rawRatings[:, joke], avg), joke))

    sims.sort()

    for n in range(n):
        neighbors.append(sims[n])

    return neighbors


def nn_coll_average(person, jokeId):
    sum = 0.0
    N = 10
    nearestNeighbors = nNN_users(N, person, jokeId)

    for n in range(len(nearestNeighbors)):
        sum += rawRatings[nearestNeighbors[n][1], jokeId-1]

    return float(sum) / float(N)


def nn_coll_weighted(person, jokeId):
    simSum = 0.0
    sum = 0.0
    N = 10
    nearestNeighbors = nNN_users(N, person, jokeId)

    for n in range(len(nearestNeighbors)):
        simSum += nearestNeighbors[n][0] # computing K
        sum += nearestNeighbors[n][0] * rawRatings[nearestNeighbors[n][1], jokeId - 1]

    k = 1.0 / float(simSum)

    return float(k) * float(sum)

def nn_coll_adjusted(person, jokeId):
    # YASH IS WORKING ON THIS
    N = 10
    sum = 0.0
    simSum = 0.0
    avg = nn_item_average(person, jokeId)
    nearestNeighbors = nNN_users(N, person)
    for n in range(len(nearestNeighbors)):
        simSum += nearestNeighbors[n][0]  # computing K
        sum += nearestNeighbors[n][0] * (rawRatings[nearestNeighbors[n][1], jokeId - 1] - avg)

    k = 1.0 / float(simSum)
    adjusted = avg + (k * sum)
    return adjusted


# Nearest Neighbor Item-based predictions
def nn_item_average(person, jokeId):
    sum = 0.0
    N = 10
    nearestNeighbors = nNN_jokes(N, person, jokeId)

    for n in range(len(nearestNeighbors)):
        sum += rawRatings[person-1, nearestNeighbors[n][1]]

    return float(sum) / float(N)


def nn_item_weighted(person, jokeId):
    simSum = 0.0
    sum = 0.0
    N = 10
    nearestNeighbors = nNN_jokes(N, person, jokeId)

    for n in range(len(nearestNeighbors)):
        simSum += nearestNeighbors[n][0] # computing K
        sum += nearestNeighbors[n][0] * rawRatings[person-1, nearestNeighbors[n][1]]

    k = 1.0 / float(simSum)

    return float(k) * float(sum)


def nn_item_adjusted(person, jokeId):
    simSum = 0.0
    sum = 0.0
    N = 10
    nearestNeighbors = nNN_jokes(N, jokeId)
    avg = nn_coll_average(person, jokeId)

    for n in range(len(nearestNeighbors)):
        simSum += nearestNeighbors[n][0]  # computing K
        sum += nearestNeighbors[n][0] * (rawRatings[person - 1, nearestNeighbors[n][1]] - avg)

    k = 1.0 / float(simSum)
    adjusted = avg + (k * sum)
    return adjusted




# just did basic error equation...can do Sum Squared, etc later if we want
def find_error(predicted, rating):
    return abs(predicted - rating)



## compute overall prediction accuracy using SSE
def sse_overall(err_vectors) -> dict:
    results = { vk : -69 for vk in err_vectors.keys() if vk != "ACTUAL" }

    for vk in err_vectors.keys():
        if vk != "ACTUAL":
            sqSum = float(np.sum(np.square(err_vectors[vk])))
            results[vk] = sqSum

    return results


def avg_sse_overall(err_vectors) -> dict:
    results = {vk: -69 for vk in err_vectors.keys() if vk != "ACTUAL" }

    for vk in err_vectors.keys():
        # dont use actual rating
        if vk != "ACTUAL":
            sqSum = np.sum(np.square(err_vectors[vk]))
            results[vk] = float(sqSum) / err_vectors[vk].size[0]

    return results


## computes the proportion of predictions accurate at threshold, epsilon
def threshold_accuracy(err_vectors, epsilon) -> dict:
    try:
        assert epsilon > 0
    except:
        print("epsilon <= 0")

    results = {vk: -69 for vk in err_vectors.keys() if vk != "ACTUAL" }

    for vk in err_vectors.keys():
        if vk != "ACTUAL":
            num_accurate = np.sum(np.where(err_vectors[vk] <= epsilon))
            results[vk] = float(num_accurate) / err_vectors[vk].shape[0]

    return results


def sentiment_accuracy(err_vectors) -> dict:
    raise NotImplementedError



def reserved_set(ratingsCopy):
    sample_size = 3
    users = np.random.choice(rawRatings.shape[0], sample_size, False)
    jokes = np.random.choice(rawRatings.shape[1], sample_size, False)

    ## need ot add kNN-IAWS and kNN-CAWS
    vec_keys = ["ACTUAL", "CMU", "CWS", "CAWS","kNN-CAVG", "kNN-CWS", "IMU", "IWS", "IAWS", "kNN-IAVG", "kNN-IWS"]
    # dict that holds computed value for each iteration to use in overall accuracy calcs
    overall_err_vectors = { vk : np.empty((sample_size,)) for vk in vec_keys }

    for i in range(len(users)):
        print("User ", users[i]+1, ", Joke ", jokes[i]+1)
        print("Real rating: ", rawRatings[users[i], jokes[i]])
        actual = rawRatings[users[i], jokes[i]]
        collAvg = coll_average(users[i], jokes[i])
        collWeighted = coll_weighted_sum(users[i], jokes[i])
        collAdj = coll_adjusted_sum(users[i], jokes[i])
        nnCollAvg = nn_coll_average(users[i], jokes[i])
        nnCollWeight = nn_coll_weighted(users[i], jokes[i])
        #nnCollAdj
        itemAvg = item_average(users[i], jokes[i])
        itemWeighted = item_weighted_sum(users[i], jokes[i])
        itemAdj = item_adjusted_sum(users[i], jokes[i])
        nnItemAvg = nn_item_average(users[i], jokes[i])
        nnItemWeight = nn_item_weighted(users[i], jokes[i])
        #nnItemAdj

        ## needed for sentiment accurracy
        ## I am going to be changing these functions to compute errors external but i wanted yall to have some stuff to work with
        overall_err_vectors["ACTUAL"][i] = rawRatings[users[i], jokes[i]]
        ## update overall i-th val in vectors
        overall_err_vectors["CMU"][i] = collAvg
        overall_err_vectors["CWS"][i] = collWeighted
        overall_err_vectors["CAWS"][i] = collAdj
        overall_err_vectors["kNN-CAVG"][i] = nnCollAvg
        overall_err_vectors["kNN-CWS"][i] = nnCollWeight
        ## add for nnCollAdj
        overall_err_vectors["IMU"][i] = itemAvg
        overall_err_vectors["IWS"][i] = itemWeighted
        overall_err_vectors["IAWS"][i] = itemAdj
        overall_err_vectors["kNN-IAVG"][i] = nnCollAvg
        overall_err_vectors["kNN-IWS"][i] = nnCollWeight
        ## add for nnItemAdj




        print("Collaborative mean utility error: %.11f" %
              find_error(collAvg, actual))
        print("Collaborative weighted sum error: %.11f" %
              find_error(collWeighted, actual))
        print("Collaborative adjusted weighted sum error: %.11f" %
              find_error(collAdj, actual))
        print("K Nearest Neighbors collaborative average error: %.11f" %
              find_error(nnCollAvg, actual))
        print("K Nearest Neighbors collaborative weighted sum error: %.11f" %
              find_error(nnCollWeight, actual))
        # print("K Nearest Neighbors collaborative adjusted weighted sum error: %.11f" %
            # find_error(nnCollAdj, actual))
        print("Item-based mean utility error: %.11f" %
              find_error(itemAvg, actual))
        print("Item-based weighted sum error: %.11f" %
              find_error(itemWeighted, actual))
        print("Item-based adjusted weighted sum error: %.11f" %
              find_error(itemAdj, actual))
        print("K Nearest Neighbors item-based average error: %.11f" %
              find_error(nnItemAvg, actual))
        print("K Nearest Neighbors item-based weighted sum error: %.11f" %
              find_error(nnItemWeight, actual))
        # print("K Nearest Neighbors item-based adjusted weighted sum error: %.11f" %
            # find_error(nnItemAdj, actual))

    sse_results = sse_overall(overall_err_vectors)
    print("SSE")
    for key in overall_err_vectors.keys():
        print(key + ":     ", sse_results[key])


def all_but_one():

    vec_keys = ["CMU", "CWS", "CAWS","kNN-CAVG", "kNN-CWS", "IMU", "IWS", "IAWS", "kNN-IAVG", "kNN-IWS"]
    # dict that holds computed value for each iteration to use in overall accuracy calcs
    overall_err_vectors = { vk : np.empty(rawRatings.shape[0]) for vk in vec_keys }

    for i in range(rawRatings.shape[0]):
        for j in range(rawRatings.shape[1]):
            if rawRatings[i, j] != 0:
                actual = rawRatings[i, j]
                rawRatings[i, j] = 0
                collAvg = coll_average(i + 1, j + 1)
                collWeighted = coll_weighted_sum(i + 1, j + 1)
                collAdj = coll_adjusted_sum(i + 1, j + 1)
                nnCollAvg = nn_coll_average(i + 1, j + 1)
                nnCollWeight = nn_coll_weighted(i + 1, j + 1)
                #nnCollAdj
                itemAvg = item_average(i + 1, j + 1)
                itemWeighted = item_weighted_sum(i + 1, j + 1)
                itemAdj = item_adjusted_sum(i + 1, j + 1)
                nnItemAvg = nn_item_average(i + 1, j + 1)
                nnItemWeight = nn_item_weighted(i + 1, j + 1)
                #nnItemAdj

                ## update overall i-th val in vectors
                overall_err_vectors["CMU"][i] = collAvg
                overall_err_vectors["CWS"][i] = collWeighted
                overall_err_vectors["CAWS"][i] = collAdj
                overall_err_vectors["kNN-CAVG"][i] = nnCollAvg
                overall_err_vectors["kNN-CWS"][i] = nnCollWeight
                ## add for nnCollAdj
                overall_err_vectors["IMU"] = itemAvg
                overall_err_vectors["IWS"] = itemWeighted
                overall_err_vectors["IAWS"] = itemAdj
                overall_err_vectors["kNN-IAVG"] = nnCollAvg
                overall_err_vectors["kNN-IWS"] = nnCollWeight
                ## add for nnItemAdj

                print("Collaborative mean utility error: %.11f" %
                      find_error(collAvg, actual))
                print("Collaborative weighted sum error: %.11f" %
                      find_error(collWeighted, actual))
                print("Collaborative adjusted weighted sum error: %.11f" %
                      find_error(collAdj, actual))
                print("K Nearest Neighbors collaborative average error: %.11f" %
                      find_error(nnCollAvg, actual))
                print("K Nearest Neighbors collaborative weighted sum error: %.11f" %
                      find_error(nnCollWeight, actual))
                #print("K Nearest Neighbors collaborative adjusted weighted sum error: %.11f" %
                      #find_error(nnCollAdj, actual))
                print("Item-based mean utility error: %.11f" %
                      find_error(itemAvg, actual))
                print("Item-based weighted sum error: %.11f" %
                      find_error(itemWeighted, actual))
                print("Item-based adjusted weighted sum error: %.11f" %
                      find_error(itemAdj, actual))
                print("K Nearest Neighbors item-based average error: %.11f" %
                      find_error(nnItemAvg, actual))
                print("K Nearest Neighbors item-based weighted sum error: %.11f" %
                      find_error(nnItemWeight, actual))
                #print("K Nearest Neighbors item-based adjusted weighted sum error: %.11f" %
                      #find_error(nnItemAdj, actual))



# RUN

userActivity, rawRatings = load_ratings()
#print (coll_average(2, 20))
#print (item_average(2, 20))
#print (coll_weighted_sum(2,20))
#print (coll_adjusted_sum(2,20))
#print (item_weighted_sum(2,20))
#print (item_adjusted_sum(2,20))
#print (rawRatings[30, 19])
#print (nn_coll_average(31, 20))
#print (nn_coll_weighted(31, 20))
#print (nn_item_average(31, 20)) # not sure why only 3 decimal points buttttttt
#print (nn_item_weighted(31, 20))
reserved_set(rawRatings)
# all_but_one()