import itertools
import random
from alias import alias_sample, create_alias_table
from based import partition_num

class RandomWalker:
    def __init__(self, graphs, nodes, verbose):
        """

        :param graphs: list of graph
        :param nodes: all nodes in the graphs
        :param verbose: whether print detail information
        """
        self.Graphs = graphs
        self.alias_nodes=[]
        self.nodes=nodes
        self.verbose=verbose

    def walk(self, walk_length, start_node):
        Graphs = self.Graphs
        alias_nodes = self.alias_nodes
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = [list(G.neighbors(cur)) if(G.has_node(cur)) else [] for G in Graphs ]
            cand=[i for i, e in enumerate(cur_nbrs) if len(e) != 0]
            if(len(cand)>0):
                select=random.choice(cand)
                walk.append(
                    cur_nbrs[select][alias_sample(*alias_nodes[select][cur])])
            else:
                break
        return walk

    def simulate_walks(self ,num_walks ,walk_length ,workers=1):

        nodes=self.nodes

        results = [
            self._simulate_walks(nodes, num, walk_length) for num in
            partition_num(num_walks, workers)]

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.walk(
                    walk_length=walk_length, start_node=v))
        return walks

    def preprocess_transition_probs(self,G):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        print("Preprocess node alias")
        alias_nodes = {}
        L=len(G.nodes())/100
        for i,node in enumerate(G.nodes()):
            if(self.verbose!=0 and i%L==0):
                print(i/len(G.nodes()))
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        self.alias_nodes.append(alias_nodes)