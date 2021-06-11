
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

#region 17 Convert a Binary tree to a Doubly Linked List TODO
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
