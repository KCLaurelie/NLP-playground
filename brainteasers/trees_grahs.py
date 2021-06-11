from collections import defaultdict

#region 6. Determine if a binary tree is a binary search tree
"""
6. Determine if a binary tree is a binary search tree
Given a Binary Tree, figure out whether it’s a Binary Search Tree.
In a binary search tree, each node’s key value is smaller than the key value of all nodes in the right subtree,
and is greater than the key values of all nodes in the left subtree
        4
    2       5
1       3

The optimal approach is a regular in-order traversal and in each recursive call,
pass maximum and minimum bounds to check whether the current node’s value is within the given bounds.

https://www.geeksforgeeks.org/a-program-to-check-if-a-binary-tree-is-bst-or-not/

Runtime Complexity: Linear, O(n) where n is number of nodes in the binary tree
Memory Complexity: Linear, O(n)
"""

class TreeNode:
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class Solution6(object):
    def is_bst_rec(self, root, min_value, max_value):
        if root is None: return True
        if root.data < min_value or root.data > max_value: return False

        return self.is_bst_rec(root.left, min_value, root.data-1) and self.is_bst_rec(root.right, root.data+1, max_value)

    def is_bst(self, root, INT_MAX=4294967296):
        return self.is_bst_rec(root, -INT_MAX - 1, INT_MAX)


root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(5)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
Solution6().is_bst(root)

#endregion

#region 5. Level order Binary Tree Traversal (Breadth First Search)
"""
5. Level order Binary Tree Traversal in python

https://stephanosterburg.gitbook.io/scrapbook/coding/python/breadth-first-search/level-order-tree-traversal
Runtime Complexity: Linear, O(n) where n is number of nodes in the binary tree
Memory Complexity: Linear, O(n)
"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution5(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if root is None: return  # Base Case
        queue = [root]  # Create an empty queue for level order traversal

        while len(queue) > 0:
            # Print front of queue and remove it from queue
            print(queue[0].val)
            node = queue.pop(0)

            # Enqueue left child
            if node.left is not None: queue.append(node.left)
            # Enqueue right child
            if node.right is not None: queue.append(node.right)

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
Solution5().levelOrder(root)

#endregion

#region Binary Trees: Lowest Common Ancestor
# https://www.hackerrank.com/challenges/binary-search-tree-lowest-common-ancestor/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=trees
class Node:
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None

def lca(root, v1, v2):
    if not root: return None
    if root.info == v1 or root.info == v2: return root
    left = lca(root.left, v1, v2)
    right = lca(root.right, v1, v2)
    if right and left: return root
    return right or left

#endregion

#region Binary Trees: Height of tree
#https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=trees
class Node:
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None

def height(root):
    if root is None: return 0
    # Compute the depth of each subtree
    lDepth = height(root.left)
    rDepth = height(root.right)

    # Use the larger one
    if (lDepth > rDepth):
        return lDepth + 1
    else:
        return rDepth + 1
#endregion

#region 17 Convert a Binary tree to a Doubly Linked List TODO
#endregion

#region 13. Deep copy of a directed graph  TODO
"""
In a diagram of a graph, a vertex is usually represented by a circle with a label, 
and an edge is represented by a line or arrow extending from one vertex to another
https://www.educative.io/m/clone-directed-graph

Runtime Complexity: Linear, O(n)
Memory Complexity: Logarithmic, O(logn)
"""
class DGNode:
    def __init__(self, d):
        self.data = d
        self.neighbors = []

class Solution13(object):
    def clone_rec(self, root, nodes_completed={}):
        if root == None: return None

        pNew = DGNode(root.data)
        nodes_completed[root] = pNew

        for p in root.neighbors:
            x = nodes_completed.get(p)
        if x is None:
            pNew.neighbors += [self.clone_rec(p, nodes_completed)]
        else:
            pNew.neighbors += [x]
        return pNew

root = DGNode(1)
root.neighbors = DGNode(2)
root.neighbors = DGNode(3)

Solution13().clone_rec(root)
#endregion

# region Breadth First Dearch / Depth First Search of a graph
"""
BFS and DFS are both simple algorithms, and they are quite similar. 
With both algorithms, we explore individual nodes — one by one — until we find a node matching a particular condition
https://medium.com/tebs-lab/breadth-first-search-and-depth-first-search-4310f3bf8416
https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/


DFS:    prioritizes searching deepest node
        we wont try to search other neighbors of start_node until 1st neighbor fully explored
        1. go to the specified start node.
        2. arbitrarily pick 1 neighbor of start_node and go there. 
        3. If that node has neighbors, arbitrarily pick one of those and go there unless we’ve already seen it
        4. When: node with no neighbors/only neighbors seen before -> go back 1 step and try other neighbor

BFS:    prioritizes most shallow nodes
        explore all neighbors of start node before going to other level
        1. go to the specified start_node
        2. explore all neighbors of start_node
        3. explore all nodes that are 2 hops away from start_node, then 3 hops...

unweighted graph: BFS will find shortest path
"""


# Directed graph using adjacency list representation
class Graph:
    def __init__(self):
        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def __repr__(self):
        return str(dict(self.graph))

    def BFS(self, start_node=0):
        # initialize all vertices as unvisited
        visited = [False] * (max(self.graph) + 1)
        queue = []
        res = ''
        # mark start_node as visited and enqueue it
        visited[start_node] = True
        queue.append(start_node)

        while queue:
            visited_node = queue.pop(0)
            res += str(visited_node) + '-->'
            for node in self.graph[visited_node]:  # look at all neighbors
                if not visited[node]:
                    queue.append(node)
                    visited[node] = True
        return res[:-3]

    def DFS(self, start_node=0):
        # recursion
        def DFS_rec(current_node, visited, res=""):
            visited.add(current_node)
            res += str(current_node) + '-->'

            for neighbor in self.graph[current_node]:
                if neighbor not in visited:
                    res = DFS_rec(neighbor, visited, res)
            return res

        visited = set()
        return DFS_rec(start_node, visited)[:-3]


"""
 0 ---------> 1 ----
 |                  |
 |            ^     |
 2 ---------> 3     |
 ^__________________|

"""
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
print(g)
g.BFS(start_node=2)
g.DFS(start_node=2)
# endregion
