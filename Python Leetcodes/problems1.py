from collections import * 
from itertools import * 
import heapq

"""
DESIGN CIRCULAR QUEUE

Design your implementation of the circular queue.

The circular queue is a linear data structure in whic the operations are performed based on FIFO principle, and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

One of the benefits of the circular queue is that we can make use of the spaces in front of the queue.

In a normal queue, once the queue becomes full, we cannot insert the next element even if there is a space in front of the queue. But using the circular queue, we can use the space to store new values.
"""

class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize the data structure here. Set the size of the queue to be k.
        Initialize the attributes here. We have a queue that is a list with a size of k, along with a current head index, a number of current elements in the queue and a total capacity of k.
        """
        self.queue = [0] * k
        self.headIndex = 0
        self.count = 0
        self.capacity = k 

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return True if the operation is successful.
        If the number of elements is already equal to the capacity, return False to indicate an unsuccessful operation.
        """
        if self.count == self.capacity: return False 
        self.queue[(self.headIndex + self.count) % self.capacity] = value 
        self.count += 1
        return True 
    
    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return True if the operation is successful.
        If there are no elements to delete, return False to indicate an unsuccessful operation.
        """
        if self.count == 0: return False 
        self.headIndex = (self.headIndex + 1) % self.capacity 
        self.count -= 1
        return True 

    def Front(self) -> int:
        """
        Retrieve the front item from the queue.
        Return -1 if there are no elements currently present in the queue.
        """
        if self.count == 0: return -1
        return self.queue[self.headIndex]

    def Rear(self) -> int:
        """
        Retrieve the last item from the queue.
        Return -1 if there are no elements currently present in the queue.
        """
        if self.count == 0: return -1
        return self.queue[(self.headIndex + self.count - 1) % self.capacity]

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.count == 0

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.count == self.capacity

"""
DESIGN HIT COUNTER 

Design a hit counter which counts the number of hits received in the past 5 minutes.

Your system should accept a timestamp parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order.

Several hits may arrive roughly at the same time.
"""

class HitCounter:
    """
    Simple logic and implementation. Use a deque to store the hits whenever the hit method is called.
    When we wish to determine the hits at a given timestamp, we pop the earliest hits so long as the time difference is greater than or equal to 300.
    After removing all the invalid past hits, we return the number of hits remaining in the counter.
    """

    def __init__(self):
        self.counter = deque()

    def hit(self, timestamp):
        self.counter.append(timestamp)

    def getHits(self, timestamp):
        while self.counter and timestamp - self.counter[0] >= 300:
            self.counter.popleft()

        return len(self.counter)

"""
ONLINE STOCK SPAN 

Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day.

The span of the stock's price in one day is the maximum number of consecutive days (starting from that day and going backward) for which the stock price was less than or equal to the price of that day.
"""

class StockSpanner:
    """
    Class has a single attribute, which is a list that behaves as a stack.
    """

    def __init__(self):
        self.stack = []

    def next(self, price):
        """
        Method to determine the span of a stock's price - the maximum number of consecutive days for which the stock price was less than or equal to the price of that day.
        Basically, we want to merge all the previous stock prices that are less than or equal to the current price and return the span.
        """
        ans = 1

        while self.stack and self.stack[-1][0] <= price: 
            ans += self.stack.pop()[1]

        self.stack.append([price, ans])

        return ans 

"""
DESIGN FILE SYSTEM

You are asked to design a file system that allows you to create new paths and associate them with different values.

The format a path is one or more concatenated strings of the form: / followed by one or more lowercase English letters.
"""

class FileSystem:
    """
    The File System class is effectively a dictionary that stores paths with a given beginning.
    """

    def __init__(self):
        self.paths = defaultdict()

    def createPath(self, path: str, value: int):
        """
        Creates a new path and associates a value to it if possible and returns True. 
        Returns False if the path already exists or its parent path doesn't exist.
        """

        if path == "/" or len(path) == 0 or path in self.paths: return False 

        parent = path[:path.rfind("/")]

        if len(parent) > 1 and parent not in self.paths: return False 
        
        self.paths[path] = value 
        
        return True 
    
    def get(self, path):
        """
        Returns the value associated with path or returns -1 if the path doesn't exist.
        """

        return self.paths.get(path, -1)

"""
DESIGN A LEADERBOARD 

Design a Leaderboard class, which has 3 functions.
"""

class Leaderboard:

    def __init__(self):
        self.scores = {}

    def addScore(self, playerId, score):
        """
        Update the leaderboard by adding score the given player's score. 
        If there is no player with such id in the leaderboard, add him to the leaderboard with the given score. 
        """

        if playerId not in self.scores: self.scores[playerId] = 0
        self.scores[playerId] += score 

    def top(self, K):
        """
        Return the score sum of the top K players.
        """

        heap = []
        for val in self.scores.values():
            heapq.heappush(heap, val)
            if len(heap) > K: heapq.heappop(heap)

        res = 0
        while heap:
            res += heapq.heappop(heap)
        return res 
    
    def reset(self, playerId):
        """
        Reset the score of the player with the given id to 0 (in other words erase it from the leaderboard).
        It is guaranteed that the player was added to the leaderboard before calling this function.
        """
        self.scores[playerId] = 0

"""
MIN STACK

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

You must implement a solution with O(1) time complexity for each function.
"""

class MinStack:

    def __init__(self):
        """
        Initializes the stack object.
        """
        self.stack = []

    def push(self, x):
        """
        Pushes the element val onto the stack.
        """

        if not self.stack: 
            self.stack.append((x, x))
            return 
        
        current_min = self.stack[-1][1]
        self.stack.append((x, min(x, current_min)))

    def pop(self):
        """
        Removes the element on top of the stack.
        """
        self.stack.pop()

    def top(self):
        """
        Gets the top element of the stack.
        """
        return self.stack[-1][0]
    
    def getMin(self):
        """
        Retrieves the minimum element in the stack.
        """
        return self.stack[-1][1]
    