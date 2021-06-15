import timeit
import math
from collections import defaultdict

#region 1. find missing number in array
"""
1. find missing number in array
You are given an array of positive numbers from 1 to n, such that all numbers from 1 to n are present except one number x. Y
ou have to find x.

Runtime Complexity: Linear, O(n)
Memory Complexity: Constant, O(1)
"""

class Solution1(object):
    def find_missing(self,input):
        # calculate sum of all elements
        # in input list
        sum_of_elements = sum(input)

        # There is exactly 1 number missing
        n = len(input) + 1
        actual_sum = (n * (n + 1)) / 2
        return actual_sum - sum_of_elements

starttime = timeit.default_timer()
print(Solution1().find_missing([3,7,1,2,8,4,5]))
print("The time difference is :", timeit.default_timer() - starttime)
#endregion

#region Min Max array
"""
You are given a list of integers arr and a single integer n.
You must create an array of length n from arr such that the difference between its max and min is minimum
"""
arr = [1,2,3,4,10,20,30,40,100,200]
k=4
# solution: selecting [1,2,3,4] gives max diff=3

arr = [100,200,300,350,400,401,402]
k=3
# solution: selecting [400,401,402] gives max diff=2
def maxMin(k, arr):
    arr.sort()
    min_diff=arr[len(arr)-1]-arr[0]
    for idx in range(len(arr)-k+1):
        curr_diff = arr[idx+k-1]-arr[idx]
        print(idx, arr[idx:idx+k], curr_diff)
        if curr_diff<min_diff:
            min_diff=curr_diff
    return min_diff
#endregion

#region MaxArray sum
"""
Given an array of integers, 
find the subset of non-adjacent elements with the maximum sum. 
Calculate the sum of that subset. 
It is possible that the maximum sum is 0, the case when all elements are negative.
Returns:  int: the maximum subset sum
"""
arr = [3, 5, -7, 8, 10] #res=15 (10+5)
arr = [2, 1, 5, 8, 4] # res=11 (4+5+2)
def maxSubsetSum(arr):
    dp = list()
    dp.append(arr[0])
    dp.append(max(arr[:2]))
    ans = dp[-1]
    for i in arr[2:]:
        print(i, dp)
        dp.append(max(i, dp[-2] + i, ans))
        ans = max(ans, dp[-1])
    return ans
#endregion

#region 22. Rotate an array by K

def rotLeft(a, d):
    if d > len(arr):
        d = d % len(arr)
    print(d)
    return arr[d:]+arr[0:d]

arr = [1, 2, 3, 4, 5, 6]
rotLeft(arr, 14)

#endregion

#region min max of windows in array
#https://www.hackerrank.com/challenges/min-max-riddle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=stacks-queues
def riddle(arr):
    stack = []
    arr.append(0)
    d=defaultdict(int)
    for i,j in enumerate(arr):           #making of step 2
        t=i
        while stack and stack[-1][0]>=j:
            val,li = stack.pop()
            d[j]=max(d[j],i-li+1)
            d[val]=max(d[val],i-li)
            t=li
        stack.append((j,t))
    del d[0]
    e=defaultdict(int)
    for i in d:                           #making of step 3
        e[d[i]]=max(e[d[i]],i)
    #print(e)
    l=len(arr)
    ans=[e[l-1]]                          #at the end, "ans" is our resulted list of step 4
    for i in range(len(arr)-2,0,-1):      #making of step 4; step4: we have to add the largest value in ans(i.e. last value in ans) if the current value of e[i] is less than last value in ans, else we have to just append e[i] to ans.
        if e[i]<ans[-1]: ans.append(ans[-1])
        else: ans.append(e[i])
    print(*ans[::-1])                   #step 5: print reverse ans
#endregion

#region Find triplets in arrays
"""
Given 3 arrays a, b, c find all triples (i,j,k) such as i<=j and j>=k
(with i in a, j in b, k in c)
return: number of triples
"""
a=[1, 3, 5, 7]
b=[5, 7, 9]
c=[7, 9, 11, 13]
# expected output: 12
def triplets(a, b, c):
    a = sorted(set(a))
    b = sorted(set(b),reverse=True)
    c = sorted(set(c))

    nb_triplets=0
    for i in a:
        for j in b:
            if j<i: break
            for k in c:
                if k>j: break
                nb_triplets+=1
    return nb_triplets


#endregion

#region 2. Find 2 numbers from array that add up to a total
"""
2. Determine if the sum of two integers is equal to the given value
Given an array of integers and a value, determine if there are any two integers in the array whose sum is equal to the given value. 
Return true if the sum exists and return false if it does not.

Runtime Complexity: Linear, O(n)
Memory Complexity: Linear, O(n)
"""
class Solution2(object):
    def find_sum_of_two(self, A, val):
        found_values = set()
        for a in A:
            if val - a in found_values:
                return True

            found_values.add(a)

        return False

print(Solution2().find_sum_of_two([3,7,1,2,8,4,5],10))
#endregion

#region Count geometric progression triplets in list
"""
You are given an array and you need to find number of tripets of indices  
such that the elements at those indices are in geometric progression of ratio r
https://www.geeksforgeeks.org/number-gp-geometric-progression-subsequences-size-3/
Time complexity: O(n)
"""
def countTriplets(arr, r):
    res = 0
    # keep track of left and right elements
    left = defaultdict(lambda:0) #to store arra[elem]/r
    right = defaultdict(lambda:0) #to store arra[elem]*r

    # count the nb f occurences of each element present in the arrays
    for elem in arr:
        right[elem] +=1

    for elem in arr:
        cl, cr = 0, 0  # initialize counters
        if elem % r == 0: # if divisible by ratio
            cl = left[elem//r]  # count elements in left hash
        right[elem] -= 1  # remove from right hash
        left[elem] += 1  # increase count in left hash
        cr = right[elem*r]  # count candidate elements in right hash
        print(elem, cl, cr)
        res += cl * cr
    return res

arr = [3, 1, 2, 6, 2, 3, 6, 9, 18, 3, 9]
countTriplets(arr, r=3)
#endregion

#region Common elements in lists
# Complexity O(n), worst case O(n^2)
def common_els(l1,l2):
    return set(l1) & set(l2)
common_els([1, 2, 3, 4, 5], [5, 6, 7, 8, 9])
#endregion

#region Finding 2 numbers from given list that add to a total

class SolutionX(object):
    def find_2_nbs_giving_total(self, total, numbers):
        n2 = total//2
        goodnums = {total-x for x in numbers if x<=n2} & {x for x in numbers if x>n2}
        pairs = {(total-x, x) for x in goodnums}
        return pairs
SolutionX().find_2_nbs_giving_total(total=181, numbers= [80, 98, 83, 92, 1, 38, 37, 54, 58, 89])
#endregion

#region Pairs that have a difference equal to the target value
# https://www.hackerrank.com/challenges/pairs/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=search
"""
arr=[1,2,3,6,7]
k=5

"""
def pairs(k, arr):
    a = set(arr)
    # make a set of all a[i] + k
    b = set(x + k for x in arr)
    # return the length of the intersection set
    return len(a&b)

def pairs2(k, arr):
    res=0
    for x in arr:
        if x+k in arr:
            #print(x, x+k)
            res+=1
    return res

print(timeit.timeit('pairs(5, [1,2,3,6,7])', globals=globals()))
print(timeit.timeit('pairs2(5, [1,2,3,6,7])', globals=globals()))
#endregion

#region 10. Find k_th permutation
"""
given a set of n elements, find the k-th permutation (given permutations are in ascending order
e.g. for the set 123
the ordered permutations are: 123, 132, 213, 231, 312, 321
https://www.geeksforgeeks.org/find-the-k-th-permutation-sequence-of-first-n-natural-numbers/


Runtime Complexity: Linear, O(n)
Memory Complexity: Linear, O(n)
"""
set=[1,2,3,4,5,6]
k = 4 # we want the 4th permutation
nb_permutations = math.factorial(len(set)) # number of permutations using 6 numbers
nb_ind_permutations = nb_permutations/(len(set))# nb of permutations for each number (=factorial n-1)
first_nb_perm= math.floor(k/nb_ind_permutations) # the kth permutation starts with that number


class Solution10(object):
    def kth_permutation(self, k, set, res):
        print('set',set,'res',res)
        if not set: # if set is empty we reached the end of the algo
            return res
        n = len(set)
        nb_ind_permutations = math.factorial(n-1) if n > 0 else 1 # nb of permutations starting with each number
        perm_group = (k-1)//nb_ind_permutations # the kth permutation starts with that number
        res = res + str(set[perm_group])
        # now we want to find permutations in reduced set
        set.pop(perm_group)
        k = k - (nb_ind_permutations*perm_group)
        self.kth_permutation(k, set, res)
Solution10().kth_permutation(k=4, set=[1,2,3,4,5,6], res='')

#endregion

#region 11. Find all subsets of a given set of integers
"""
given the set [1,2,3]
the possible subsets are 1, 2, 3, [1,2], [1,3],[2,3],[1,2,3]

Runtime Complexity: Exponential, O(2^n*n)
Memory Complexity: Exponential, O(2^n*n)
"""

# solution using recursion
class Solution11(object):
    def all_subsets(self, _set, res=[], to_return=[]):
        for size_subset in range(len(_set)):
            print('res', res)
            new_res = res.copy()
            new_res.append(_set[size_subset])
            new_set = _set[size_subset+1:]
            print('new_res', new_res)
            to_return.append(new_res)
            self.all_subsets(new_set, new_res)
        return to_return
Solution11().all_subsets(set=[2,3,4])

def powerset(s):
    x = len(s)
    for i in range(1 << x):
        print([s[j] for j in range(x) if (i & (1 << j))])
#endregion

#region 21. Implement a stack with push(), min(), and pop() in O(1)O(1) time
# https://www.geeksforgeeks.org/design-and-implement-special-stack-data-structure/
# https://www.geeksforgeeks.org/design-a-stack-that-supports-getmin-in-o1-time-and-o1-extra-space/
class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)
s=Stack()

print(s.isEmpty())
s.push(4)
s.push('dog')
print(s.peek())
s.push(True)
#endregion

#region 25 Implement a queue using a linked list
"""
https://www.geeksforgeeks.org/queue-linked-list-implementation/

QUEUE: FIFO
In a Queue data structure, we maintain two pointers, front and rear. The front points the first item of queue and rear points to last item.
enQueue() This operation adds a new node after rear and moves rear to the next node.
deQueue() This operation removes the front node and moves front to the next node.

Runtime Complexity: Time complexity of both operations enqueue() and dequeue() is O(1)
"""
# Linked list node
class LLNode:
    def __init__(self, data):
        self.data = data
        self.next = None
class Queue:
    def __init__(self):
        self.front = self.rear = None

class Solution25(object):
    def enQueue(self, queue, item):
        nodeitem = LLNode(item)
        if queue.rear is None:
            queue.front = queue.rear = nodeitem
        else:
            queue.rear.next = nodeitem
            queue.rear = nodeitem
        print("Queue Rear ", str(queue.rear.data))

    def deQueue(self, queue):
        if (queue.rear is None) or (queue.front is None):
            print('queue is empty')
            return
        else:
            tmp = queue.front
            queue.front = tmp.next
        print("Queue Rear ", str(queue.rear.data))

q = Queue()
qobj = Solution25()
qobj.enQueue(queue=q, item=10)
qobj.enQueue(queue=q, item=20)
qobj.deQueue(queue=q)

#endregion

#region Find cycle in a linked list
"""
https://www.hackerrank.com/challenges/ctci-linked-list-cycle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=linked-lists
https://www.geeksforgeeks.org/detect-loop-in-a-linked-list/

Solution using Floyd's cycles
Time complexity: O(n) (Only one traversal of the loop is needed)
Auxiliary Space:O(1)
"""

class Node(object):
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next = next_node

def has_cycle(head):
    slow_p = head
    fast_p = head
    while slow_p and fast_p and fast_p.next:
        slow_p = slow_p.next
        fast_p = fast_p.next.next
        if slow_p == fast_p:
            return True
    return False


head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
head.next.next.next = Node(4)
head.next.next.next.next = Node(5)
# Create a loop for testing(5 is pointing to 3)
head.next.next.next.next.next = head.next.next
#endregion

#region Insert Node in a linked list
class Node(object):
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next = next_node

def insertNodeAtPosition(llist, data, position):
    new = Node(data)
    pointer = head
    counter = 1

    while pointer.next is not None:
        if counter == position:
            new.next = pointer.next
            pointer.next = new
            break
        counter += 1
        pointer = pointer.next
    return head


head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
head.next.next.next = Node(4)
head.next.next.next.next = Node(5)
#endregion
