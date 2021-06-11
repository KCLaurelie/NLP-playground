import collections
from collections import defaultdict
from collections import Counter
from random import randint

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

        return self.is_bst_rec(root.left, min_value, root.data) and self.is_bst_rec(root.right, root.data, max_value)

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

#region Divide and Conquer (binary search of given element in an array)

"""
EARCH ELEMENT IN A SORTED ARRAY
Binary Search works on a divide-and-conquer approach and relies on the fact that the array is sorted to eliminate half of possible candidates in each iteration. 
More specifically, it compares the middle element of the sorted array to the element it's searching for in order to decide where to continue the search.
Runtime complexity: O(logn)
"""


class SolutionBinarySearch(object):
    def binary_search_rec(self, element, array, start=0, end=None):
        if end is None: end = len(array) # initializing
        # recursive loop
        mid = (start + end) //2
        if element == array[mid]:
            return mid
        elif element <= array[mid]:
            return self.binary_search_rec(element, array, start, mid-1)
        else:
            return self.binary_search_rec(element, array, mid+1, end)

element = 18
array = [1, 2, 5, 7, 13, 15, 16, 18, 24, 28, 29]
SolutionBinarySearch().binary_search_rec(element, array)
#endregion

#region Minimum swaps to sort a list
def minimumSwaps(arr):
    swaps = 0
    n = len(arr)

    for idx in range(n):
        while arr[idx] - 1 != idx:  # check if element in its right place
            ele = arr[idx]  # this is the misplaced element
            arr[ele - 1], arr[idx] = arr[idx], arr[ele - 1]  # we swap it back where it belongs
            swaps += 1  # and increase the swap counter
    return swaps
minimumSwaps([1,2,3,5,4,6,7,8])
#endregion

#region 16. find K largest elements from an array
"""
https://www.geeksforgeeks.org/k-largestor-smallest-elements-in-an-array/
"""
#METHOD1 Using sorting
#Runtime complexity: O(nlogn) (n: size of array)
class Solution16(object):
    def klargest(self, array, k):
        size=len(array)
        array.sort()
        return array[len(array)-k:len(array)]

#METHOD2 Usining min heap (optimization of method 1)
#Runtime complexity: O(k+(n-k)logk) (n: size of array)
class Solution16b(object):
    def klargest(self, array, k):
        size = len(array)

        # create min heap of k elements with priority queue
        minHeap = array[0:k].copy()

        for i in range(k, size):
            minHeap.sort()
            if array[i] > minHeap[0]:
                minHeap.pop(0)
                minHeap.append(array[i])

        return minHeap

Solution16().klargest(array=[1, 23, 12, 9, 30, 2, 50], k=3)
Solution16b().klargest(array=[1, 23, 12, 9, 30, 2, 50], k=3)
#endregion

#region 3. Merge two sorted linked lists
"""3. Merge two sorted linked lists
Given two sorted linked lists, merge them so that the resulting linked list is also sorted. 
Consider two sorted linked lists and the merged list below them as an example.

https://realpython.com/linked-lists-python/
https://dbader.org/blog/python-linked-list

Runtime Complexity: Linear, O(m + n) where m and n are lengths of both linked lists
Memory Complexity: Constant, O(1)
"""

# METHOD 1: using deque
llst = collections.deque()  # create linked list
llst1 = collections.deque(['A', 'C', 'M', 'X'])
llst2 = collections.deque(['B', 'D', 'E', 'Y'])
llst_merged=(list(llst1)+list(llst2))
llst_merged.sort() # surely that's cheating
llst.pop()  # remove last element

def merge_sorted_llst(llst1, llst2):
    merged_llst = collections.deque()
    while llst1:
        if llst2: # if there's still stuff in llst2
            el1 = llst1[0]
            el2 = llst2[0]
            if el1 <= el2:
                merged_llst.append(el1)
                llst1.popleft()
            else:
                merged_llst.append(el2)
                llst2.popleft()
        else:
            merged_llst += llst1

    if llst2: merged_llst += llst2
    return merged_llst

merge_sorted_llst(collections.deque(['A', 'C', 'M', 'X']), collections.deque(['A', 'B', 'D', 'E', 'Y', 'Z']))
merge_sorted_llst(collections.deque(['A', 'B', 'C', 'D','E','F','G','H','I', 'M', 'X']), collections.deque(['A', 'B', 'D', 'E', 'Y', 'Z']))


# METHOD 2: building linked lists objects from scratch
class LLNode:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return self.data


class LinkedList:
    def __init__(self, nodes=None):
        self.head = None
        if nodes is not None:
            node = LLNode(data=nodes.pop(0))
            self.head = node
            for elem in nodes:
                node.next = LLNode(data=elem)
                node = node.next

    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(node.data)
            node = node.next
        nodes.append("None")
        return " -> ".join(nodes)

    def __iter__(self): # to traverse the llist
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def add_first(self, node):  # inserting at the beginning
        node.next = self.head
        self.head = node

    def add_last(self, node):  # inserting at the end
        if self.head is None:
            self.head = node
            return
        for current_node in self:  # traverse the whole list until you reach the end
            pass
        current_node.next = node

    def append(self, llst2):  # inserting at the end
        if self.head is None:
            self = llst2
            return
        for current_node in self:  # traverse the whole list until you reach the end
            pass
        for current_nodellst2 in llst2:
            current_node.next = current_nodellst2
            current_node = current_nodellst2

    def pop_left(self):  # removing first node
        if self.head is None:
            raise Exception("List is empty")
        else:
            self.head = self.head.next
            return

    def pop_right(self):  # removing last node
        if self.head is None:
            raise Exception("List is empty")
        else:
            current_node = self.head
            while current_node.next:
                prev_node = current_node
                current_node = current_node.next
            prev_node.next = None
            return

    def merge_sorted_llst(self, llst2):
        llmerged = LinkedList()
        while self.head is not None:
            if llist2.head is not None and self.head.data <= llist2.head.data:
                llmerged.add_last(LLNode(self.head.data))
                self.pop_left()
            elif llist2.head is not None and self.head.data > llist2.head.data:
                llmerged.add_last(LLNode(llist2.head.data))
                llist2.pop_left()
            else:
                break
        # append remaining bits
        if self.head is not None: llmerged.append(self)
        if llist2.head is not None: llmerged.append(llist2)
        return llmerged


llist1 = LinkedList(["a", "b", "c", "d", "e"])
llist2 = LinkedList(["b", "e", "f", "g", "h"])

llist1.add_last(LLNode('f'))
llist1.append(llist2)
llist1.merge_sorted_llst(llist2)
print(llist1, llist1.head, llist1.pop_left())

#endregion

#region Common sorting algos
class SolutionSorting(object):

    # Runtime complexity: O(n2) on average
    def BubbleSort(self, array):
        n = len(array)
        if n < 2: return array  # nothing to do

        for i in range(n):
            already_sorted = True

            for j in range(n-i-1):
                if array[j] > array[j+1]: # swap elements if not sorted
                    array[j], array[j + 1] = array[j + 1], array[j]
                    already_sorted = False

            if already_sorted:
                break

        return array

    # Runtime complexity: O(n2) on average
    def InsertionSort(self, array):
        for i in range(1, len(array)):
            key_item = array[i]
            left_ix = i - 1

            # go through left portion of matrix
            while left_ix >= 0 and array[left_ix] > key_item:
                array[left_ix+1] = array[left_ix] # shift values
                left_ix -=1
            array[left_ix + 1] = key_item

    # Runtime complexity: O(nlogn)
    def QuickSort(self, array):
        if len(array) < 2: return array # nothing to do

        low, same, high = [], [], [] # elements smaller than pivot go to low, bigger in high etc...
        pivot = array[randint(0, len(array)-1)] # select pivot randomly

        for item in array:
            if item < pivot: low.append(item)
            elif item == pivot: same.append(item)
            else: high.append(item)
        return self.QuickSort(low) + same + self.QuickSort(high)


    # Runtime complexity: O(nlogn)
    # like quicksort but uses middle element as pivot
    def MergeSort(self, array):
        if len(array) < 2: return array  # nothing to do
        midpoint = len(array) // 2
        left = self.MergeSort(array[:midpoint])
        right = self.MergeSort(array[midpoint:])
        # TODO merge left and right
        return 0

array_to_sort = [randint(0, 1000) for i in range(42)]
SolutionSorting().QuickSort(array)

#endregion

#region HeapSort TODO
"""
https://www.geeksforgeeks.org/heap-sort/
A Binary Heap is a Complete Binary Tree where items are stored in a special order 
such that the value in a parent node is greater(or smaller) than the values in its two children nodes. 
The former is called max heap and the latter is called min-heap.

Runtime complexity: O(nlogn) (complexity of heapify: O(logn)
Memory complexity: 
"""
#endregion

#region Breadth First Dearch / Depth First Search of a graph TODO
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
        visited = [False] * (max(self.graph)+1)
        queue =[]
        res = ''
        # mark start_node as visited and enqueue it
        visited[start_node] = True
        queue.append(start_node)

        while queue:
            visited_node = queue.pop(0)
            res += str(visited_node)+'-->'
            for node in self.graph[visited_node]: # look at all neighbors
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
#endregion

