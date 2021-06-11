
#region 4. Deep copy linked list with arbitrary pointer TODO
"""
4. Copy linked list with arbitrary pointer
You are given a linked list where the node has two pointers.
The first is the regular next pointer.
The second pointer is called arbitrary_pointer and it can point to any node in the linked list.
Your job is to write code to make a deep copy of the given linked list.
Here, deep copy means that any operations on the original list should not affect the copied list.

Runtime Complexity: Linear, O(n)
Memory Complexity: Linear, O(n)
"""

# TODO

#endregion

#region 21. Implement a stack with push(), min(), and pop() in O(1)O(1) time TODO
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

#region Largest Rectangle
# https://www.hackerrank.com/challenges/largest-rectangle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=stacks-queues
# https://www.geeksforgeeks.org/largest-rectangle-under-histogram/
# Runtime: O(n)

def getMaxArea(arr):
    s = [-1]
    n = len(arr)
    area = 0
    i = 0
    left_smaller = [-1] * n
    right_smaller = [n] * n
    while i < n:
        while s and (s[-1] != -1) and (arr[s[-1]] > arr[i]):
            right_smaller[s[-1]] = i
            s.pop()
        if ((i > 0) and (arr[i] == arr[i - 1])):
            left_smaller[i] = left_smaller[i - 1]
        else:
            left_smaller[i] = s[-1]
        s.append(i)
        i += 1
    for j in range(0, n):
        area = max(area, arr[j] * (right_smaller[j] - left_smaller[j] - 1))
    return area


hist = [6, 2, 5, 4, 5, 1, 6]
print("maxArea = ", getMaxArea(hist))
#endregion

