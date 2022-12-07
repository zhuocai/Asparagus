import copy
import numpy as np


class CutsetGraph:

    def __init__(self, nodes, start_node=None, end_node=None):
        # name of nodes

        self.nodes = nodes
        self.node_to_index_map = dict(zip(nodes, range(len(nodes))))

        self.edges = {}
        for node in nodes:
            self.edges[node] = []
        self.reverse_edges = {}
        for node in nodes:
            self.reverse_edges[node] = []

        self.start_node = start_node
        self.end_node = end_node

        self.stack = []
        self.visited = set()

        self.assigned = {}

    def add_edge(self, s, t):
        assert s in self.nodes
        assert t in self.nodes
        if t not in self.edges[s]:
            self.edges[s].append(t)
        if s not in self.reverse_edges[t]:
            self.reverse_edges[t].append(s)

    def dfs(self, node, nodes=None):
        if nodes is None:
            nodes_set = self.nodes
        else:
            nodes_set = nodes
        if node not in self.visited:
            self.visited.add(node)
            for nbr in self.edges[node]:
                if nbr in nodes_set:
                    self.dfs(nbr, nodes)
            self.stack.append(node)

    # def dfs_reverse(self, node):
    #     if node not in self.visited:
    #         pass
    def assign(self, node1, node2, nodes=None):
        if nodes is None:
            nodes_set = self.nodes
        else:
            nodes_set = nodes
        if node1 not in self.assigned:
            self.assigned[node1] = node2
            for node3 in self.reverse_edges[node1]:
                if node3 in nodes_set:
                    self.assign(node3, node2, nodes)

    def get_sccs(self, start_node, selected_nodes):
        self.stack = []
        self.visited = set()

        #self.dfs(start_node, selected_nodes)
        if selected_nodes is not None:
            for node in selected_nodes:
                self.dfs(node, selected_nodes)
        else:
            self.dfs(start_node, selected_nodes)
            for node in self.nodes:
                self.dfs(node, selected_nodes)
        #self.stack1 = copy.copy(self.stack)
        #self.stack = []
        #print('stack:', self.stack)
        #print('visited:', self.visited)
        self.visited = set()

        self.assigned = dict()
        for i in reversed(self.stack):
            self.assign(i, i, selected_nodes)
        # print('assigned=',self.assigned)

        sccs = {}
        for i, j in self.assigned.items():
            if j not in sccs:
                sccs[j] = [i]
            else:
                sccs[j].append(i)

        scc_single = []
        scc_multi = {}
        for i, nodes in sccs.items():
            if len(nodes) == 1:
                scc_single.append(i)
            else:
                scc_multi[i] = nodes
        # print('scc_single:', scc_single)
        # print('scc_multi:', scc_multi)
        return {'single': scc_single, 'multi': scc_multi}

    def get_cut_set_rec(self, selected_nodes=None):
        #print('GET CUT SET, selected_nodes = ', selected_nodes)
        if selected_nodes is None:
            start_node = self.start_node
        else:
            if len(selected_nodes) == 0:
                return []
            start_node = selected_nodes[0]
        sccs = self.get_sccs(start_node, selected_nodes)

        cut_set_now = []
        for i, nodes in sccs['multi'].items():
            cut_set_now.append(i)
            new_nodes = copy.copy(nodes)
            new_nodes.remove(i)

            cut_set_now = cut_set_now + self.get_cut_set_rec(new_nodes)
            #print('selected_nodes===', new_nodes, 'cut_set_now ==== ', cut_set_now)

        return cut_set_now

    def get_cut_set(self):
        cut_set_nodes = self.get_cut_set_rec()
        self.fill_connect_matrix()
        select = [n in cut_set_nodes for n in self.nodes]
        cut_set = []
        for node in cut_set_nodes:
            cut_set.append(
                (node, np.sum(self.connect_mat[self.node_to_index_map[node]][select])))
        self.cut_set = cut_set
        return self.cut_set

    def fill_connect_matrix(self):
        adj_mat = np.eye(len(self.nodes))
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if n2 in self.edges[n1]:
                    adj_mat[i][j] = 1
        self.adj_mat = adj_mat

        connect_mat = copy.deepcopy(adj_mat)
        for _ in self.nodes:
            connect_mat = np.matmul(connect_mat, adj_mat) > 0
        self.connect_mat = connect_mat


if __name__ == '__main__':
    nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    cut_set_graph = CutsetGraph(nodes=nodes, start_node=0, end_node=7)
    cut_set_graph.add_edge(0, 1)
    cut_set_graph.add_edge(1, 2)
    cut_set_graph.add_edge(1, 6)
    cut_set_graph.add_edge(6, 3)
    cut_set_graph.add_edge(3, 6)
    cut_set_graph.add_edge(6, 4)
    cut_set_graph.add_edge(4, 5)
    cut_set_graph.add_edge(2, 1)
    cut_set_graph.add_edge(6, 7)
    cut_set = cut_set_graph.get_cut_set()
    print('cut_set = ', cut_set)
    cut_set_graph.fill_connect_matrix()
    print('index map:', cut_set_graph.node_to_index_map)
    print('adj_mat:', cut_set_graph.adj_mat)
    print('connect_mat:', cut_set_graph.connect_mat)
