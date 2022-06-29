#This code is used to compute the rwc of a given network, with users assigned to novax communities True or False.
#The rwc is a measure of the polarization of the network that is partitioned in sides X and Y.
#It is computed as 
#RWC = P_XX*P_YY - P_XY*PYX
#where P_XY = P(rw started in Y|rw ended in a popular node in X)
#to do this, we need (i) to build the network, (ii) to extract the k most popular nodes at sides X and Y, (iii) simulate random walks (rw) that restart at one side, (iiii) compute the probabilities
#


import pandas as pd
import networkx as nx


def rwc(edges, coms, k = 100):
    #create the weighted directed networkx graph (DiGraph)
    #on this graph i can perform all the random walks
    G = nx.from_pandas_edgelist(edges, source = "user", target = "user_RT", 
                                    edge_attr = "weight", create_using = nx.DiGraph)

    #in-degree dataframe, for selecting the most popular nodes in each side
    in_deg = pd.DataFrame(G.in_degree, columns = ["user", "in_degree"])

    #we call users1 the ones in novax community, and users2 the ones in other side
    users_com1 = coms.query("novax")["user"]
    users_com2 = coms.query("~novax")["user"]

    #for each user we get information on its indegree and on its membership to a novax community
    #then we sorted nodes by indegree and extract the first k ones
    top1 = coms.merge(in_deg).sort_values("in_degree", ascending = False).query("novax").head(k)["user"]
    top2 = coms.merge(in_deg).sort_values("in_degree", ascending = False).query("~novax").head(k)["user"]

    #we will define random walks having uniform probability to restart in one node of side X
    #for this, we define these uniform distribution, assignign probability 1 / N_X to the N_X nodes
    #of side X, and ~0 to the rest
    N1 = len(users_com1)
    N2 = len(users_com2)
    uni_users1 = {u: 1 / N1 for u in users_com1}
    uni_users1.update({u: 1e-12 for u in users_com2})
    uni_users2 = {u: 1 / N2 for u in users_com2}
    uni_users2.update({u: 1e-12 for u in users_com1})


    #i compute the personalized page rank, with restart on the nodes of one side
    #it simulates random walks (until convergence), such that at each step the probability of restart
    #at a node in side X is 0.85
    #in this way we have the probability of ending at each node, having started at side x
    pr1 = nx.pagerank(G, alpha = 0.85,
                      personalization = uni_users1,   
                      dangling = uni_users1, max_iter = 100000)

    pr2 = nx.pagerank(G, alpha = 0.85,
                      personalization = uni_users2,
                      dangling = uni_users2, max_iter = 100000)
    #create a dataframe with users and page ranks, filter users at the top 
    #and measure the total probability of ending the RW in one of these nodes
    #copmute this, for RW starting at side 1 or 2, and ending at side 1 or 2
    #rw_YX is the probability of ending at side X having started at side Y
    rw_11 = pd.DataFrame({"user": pr1.keys(), "pr": pr1.values()}).merge(top1).sum()["pr"]
    rw_12 = pd.DataFrame({"user": pr1.keys(), "pr": pr1.values()}).merge(top2).sum()["pr"]
    rw_21 = pd.DataFrame({"user": pr2.keys(), "pr": pr2.values()}).merge(top1).sum()["pr"]
    rw_22 = pd.DataFrame({"user": pr2.keys(), "pr": pr2.values()}).merge(top2).sum()["pr"]

    #P_XY = P(started Y|end X) = (P(end X|started Y) * P(started Y)) / P(end X)) 
    #P(end X|started Y) = rw_XY
    #P(started Y) = #users in Y / #users
    #P(end X) = P(end X|started X) * P(started X) + P(end X|started Y) * P(started Y) 


    #proportion of nodes per side = P(started X)
    prop1, prop2 = len(users_com1) / len(coms), len(users_com2) / len(coms)

    #P(statred X|end Y) = P_XY
    p11 = rw_11 * prop1 / (rw_11 * prop1 + rw_21 * prop2)
    p12 = rw_12 * prop2 / (rw_12 * prop1 + rw_22 * prop2)
    p21 = rw_21 * prop1 / (rw_11 * prop1 + rw_21 * prop2)
    p22 = rw_22 * prop2 / (rw_12 * prop1 + rw_22 * prop2)

    #RWC = P_XX * P_YY - P_XY * P_YX
    rwc = p11 * p22 - p12 * p21                           

    return rwc                          
                                  
                                  
                                  
                                  
                                  












