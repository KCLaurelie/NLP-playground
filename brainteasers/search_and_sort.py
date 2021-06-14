import collections
from collections import Counter
from random import randint

#region Divide and Conquer (binary search of given element in an array)

"""
SEARCH ELEMENT IN A SORTED ARRAY
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
            el = arr[idx]  # this is the misplaced element
            arr[el - 1], arr[idx] = arr[idx], arr[el - 1]  # we swap it back where it belongs
            swaps += 1  # and increase the swap counter
    return swaps
minimumSwaps([1,2,3,5,4,6,7,8])
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

#region Minimum time required
"""
https://www.hackerrank.com/challenges/minimum-time-required/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=search

You have a number of machines that each have a fixed number of days to produce an item. 
All the machines operate simultaneously
Determine the minimum number of days to produce an order.

For example, you have to produce 10 items. You have 3 machines that take [2,3,2] days to make an item

Day Production  Count
2   2               2
3   1               3
4   2               5
6   3               8
8   2              10
It takes 8 days to produce 10 items using these machines.
"""
machines =[2,3,2]
goal = 10
def minTime(machines, goal):
    machines.sort()

    low_rate = machines[0] # fastest machines
    lower_bound = (goal // (len(machines) / low_rate))
    high_rate = machines[-1] # slowest machines
    upper_bound = (goal // (len(machines) / high_rate)) + 1

    while lower_bound < upper_bound:
        num_days = (lower_bound + upper_bound) // 2
        # get number of items
        total = 0
        for machine in machines:
            total += (num_days // machine)
        if total >= goal:
            upper_bound = num_days
        else:
            lower_bound = num_days + 1

    return int(lower_bound)


#endregion

