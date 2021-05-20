# %% 
import networkx as nx

# %% Karate dataset
G = nx.karate_club_graph()

# G is an undirected graph
type(G)

# Visualize the graph
nx.draw(G, with_labels = True)

# %%
def average_degree(num_edges, num_nodes):
  # TODO: Implement this function that takes number of edges
  # and number of nodes, and returns the average node degree of 
  # the graph. Round the result to nearest integer (for example 
  # 3.3 will be rounded to 3 and 3.7 will be rounded to 4)

  avg_degree = round(2*num_edges/num_nodes)
 
  return avg_degree

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print("Average degree of karate club network is {}".format(avg_degree))

# %%
def average_clustering_coefficient(G):
  # TODO: Implement this function that takes a nx.Graph
  # and returns the average clustering coefficient. Round 
  # the result to 2 decimal places (for example 3.333 will
  # be rounded to 3.33 and 3.7571 will be rounded to 3.76)

  avg_cluster_coef = round(nx.average_clustering(G),2)

  return avg_cluster_coef

avg_cluster_coef = average_clustering_coefficient(G)
print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))

# %%
def one_iter_pagerank(G, beta, r0, node_id):
  # TODO: Implement this function that takes a nx.Graph, beta, r0 and node id.
  # The return value r1 is one interation PageRank value for the input node.
  # Please round r1 to 2 decimal places.

  pr = nx.pagerank(G, beta, max_iter=1)
#   r1 = 

  ############# Your code here ############
  ## Note: 
  ## 1: You should not use nx.pagerank

  #########################################

  return r1

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0
pr = nx.pagerank(G, beta, max_iter=1)
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))