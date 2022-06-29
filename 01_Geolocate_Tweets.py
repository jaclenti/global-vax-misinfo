import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


#list of fancy substrings that often cause mismatches in the geolocation
strange_places = []
with open("/home/jlenti/Codes/strange_places.txt","r") as f:
    for line in f:
        line = line.strip()
        strange_places.append(line)
        
#locdictionary associate each word to the list of location (geonameid) it can be associated to
locdictionary = json.load(open("/home/jlenti/Codes-cp/geonameslocator/locdictionary.json"))

#dataframe with all location we can geolocate, and relative country
#we also need the population, because if a string is associated two locations, we choose the most populated one
locationdata = pd.read_csv("/home/jlenti/Codes-cp/geonameslocator/countriesdatap.tsv", sep = '\t', low_memory = False,
                           index_col = "geonameid").drop([2855707, 2769324])

#other mismatches are given by numbers or stopwords
engstopwords = stopwords.words("english")
esstopwords = stopwords.words("spanish")
ptstopwords = stopwords.words("portuguese")

numbers = {'zero','one','two','three','four','five','six','seven','eight','nine','ten',
           'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen',
           'nineteen','twenty','thirty','fourty','fifty','sixty','seventy','eighty','ninety',
           'hundred','thousand','million','billion','zillion'}

num = [str(n) for n in range(100)]

# Turn tokens into a sequence of n-grams
def word_ngrams(tokens, ngrams):
    min_n, max_n = 1, ngrams
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append(" ".join(original_tokens[i: i + n]))
    return tokens

#transform a string in sequences of names, to find the possible locations inside a longer string
tokenizer = RegexpTokenizer(r'\w+')
def find_names(text):
    tokens = word_ngrams(tokenizer.tokenize(text.lower()), 5)
    m = set()
    for token in tokens:
        if token in locdictionary:
            m.add(token)
    # filter out matched places that are substrings of another matched place
    k_list = list(m)
    for i, k in enumerate(k_list):
        for k2 in k_list[:i]:
            if k in k2 and k in m:
                m.remove(k)
        for k2 in k_list[i+1:]:
            if k in k2 and k in m:
                m.remove(k)
    return m

# Finds the best match for the text
# For speed reasons, returns only the geonamesid which can be used to index into countriesdatap
def find_best_match(text):
    #correct common mismatches
    if "Ile-de-France" in text:
        text = "Paris"
    if "Repùblica Catalana" in text:
        text = "Barcelona"
    if ("Buenos Aire" in text)|("Castelar" in text):
        text = "Buenos Aires"
    if "Islamabad" in text:
        text = "Pakistan"
    if "Brasi" in text:
        text = "Brasil"
    mymatches = find_names(text)
    #if mymatches:
    # Try finding all the matches
    bestmatch = None
    matcharray = []
    matchedids = []
    foundcountry = None
    countrymatches = None
    # Put all the matches in one array
    for foundloc in mymatches:
        matchedids = matchedids + locdictionary[foundloc]
     #remove some bad match
    
    # If there is more than 1 match, and there is a match to a country, constrain all other matches to that country
    if len(matchedids) > 1:
        matchedlocations = locationdata.loc[matchedids]
        if pd.DataFrame.any(matchedlocations.featurecode=="PCLI"):
            foundcountry = matchedlocations[matchedlocations.featurecode=="PCLI"].iloc[0].countrycode
            countrymatches = matchedlocations[matchedlocations.countrycode==foundcountry]
    # For each word match, find the most populous one
    for foundloc in mymatches:
        # Handle special cases
        if len(foundloc) < 2:
            continue
        if (foundloc in engstopwords) | (foundloc in esstopwords) | (foundloc in ptstopwords) | (foundloc in num) | (foundloc in numbers) | (foundloc in strange_places):
            continue
        else:
            matchedlocations = locationdata.loc[locdictionary[foundloc],:]
            if foundcountry: # Remove matches that are not in the country we found
                matchedlocations = matchedlocations[matchedlocations.countrycode==foundcountry]
            if matchedlocations.shape[0] > 0:
                foundmatch = matchedlocations.sort_values(by="population",ascending=False).index[0]
                matcharray.append(foundmatch)
    # Among all the matches, find the least populous one
    if len(matcharray) > 0:
        allmatches = locationdata.loc[matcharray]
        bestmatch = allmatches.sort_values(by="population",ascending=True).index[0]
    return bestmatch

def find_location(text):
    #return [location, countrycode] if possible, otherwise [null, null]
    #if the string seems to be a link or tag return [null, null]
    if text == None:
        return["null", "null"]
    if ".com" in text:
        return ["null", "null"]
    elif "@" in text:
        return ["null", "null"]
    else:
        try:
            #extract the most populous location substring from the string
            loc_match = find_best_match(text)
            #geolocate the location from the geonameid
            location = locationdata.loc[loc_match]
            #keep only location and countrycode
            return location[["name", "countrycode"]]
        except:
            return ["null", "null"]










