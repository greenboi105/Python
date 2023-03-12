import math
from collections import *
import heapq

class Solution:

    """
    MAGNETIC FORCE BETWEEN TWO BALLS

    In the universe Earth C-137, Rick discovered a special form of magnetic force between two balls if they are put in his new invented basket. 
    
    Rick has n empty baskets, the ith basket is at position[i]. Morty has m balls and needs to distribute the balls into the baskets such that the minimum magnetic force between any two balls is at a maximum.

    The magnetic force between two different balls at positions x and y is |x - y|. Given the integer array position and the integer m, we want to return the required force.
    """
    
    def maxDistance(self, position, m):

        # Sort the positions in order and determine the possible search space of distances
        position.sort()
        left, right = 1, position[-1] - position[0]

        # Process with a search space of 1 to determine if the a proposed midpoint distance will allows us to place all the balls such that the magnetic force is maximum
        while left <= right:

            # Calculate a new midpoint distance using the remaining search space
            balls_placed, current_position, mid = 1, position[0], (left + right) // 2

            for i in range(1, len(position)):
                if position[i] - current_position >= mid:
                    balls_placed += 1
                    current_position = position[i]

            # Narrow the search space depending on the number of balls that can be placed for the given distance - the goal is to maximize the possible distance
            if balls_placed >= m: 
                left = mid + 1
                max_force = mid
            else: right = mid - 1

        # Return the required force such that the minimum magnetic force between any two balls is maximum (which is effectively the distance)
        return max_force

    """
    MINIMUM SPEED TO ARRIVE ON TIME

    You are given a floating-point number hour, representing the amount of time you have to reach the office.

    To commute to the office, you must take n trains in sequential order. You are also given an integer array dist of length n, where dist[i] describes the distance (in kilometers) of the ith train ride.

    Each train can only depart at an integer hour, so you may need to wait in between each train ride.

        - For example, if the 1st train ride takes 1.5 hours, you must wait for an additional 0.5 hours before you can depart on the 2nd train ride at the 2 hour mark.

    Return the minimum positive integer speed (in kilometers per hour) that all the trains must travel at for you to reach the office on time, or -1 if it is impossible to be on time.

    Tests are generated usch that the answer will not exceed 10e7 and hour will have at most two digits after the decimal point.
    """

    def minSpeedOnTime(self, dist, hour):

        # Establish the search space of possible speeds we can travel - the slowest possible speed for the train to travel is 1, while the fastest possible speed is 1 greater than the test max 10e7
        left, right = 1, int(10e7) + 1

        # Prune the valid search space so long as the left and right pointers do not equal - for each iteration we calculate a midpoint speed using the remaining search space and determine the amount of time needed to visit all the stops with the given travel speed
        while left < right:

            # Determine a speed to travel using the available search space (low and high potential speeds) for the current iteration so that we arrive before the set hour value 
            mid_speed = (left + right) // 2

            # Calculate the amount of time needed to travel to the location given the calculated speed midpoint, knowing that the final point will be a floating point value
            required_time = sum([math.ceil(distance / mid_speed) for distance in dist[:-1]]) + (dist[-1] / mid_speed)

            # If the required speed is greater than the number of hours, increment lo for the next iteration, otherwise set hi to the current prospective speed (narrowing the search interval depending on circumstances for the given iteration)
            if required_time <= hour: right = mid_speed
            else: left = mid_speed + 1 

        # Return -1 if there is no possible speed we can travel at to reach the office on time
        return -1 if right == int(10e7) + 1 else right

    """
    KOKO EATING BANANAS
    
    There are n piles of bananas, the i-th pile has piles[i] bananas. The guards are gone and will come back in h hours. 

    Koko can decide on her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. 

    If the pile has less than k bananas, she eats all of them and will not eat any more bananas this hour.

    Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

    Return the minimum integer k such that she can eat all the bananas within h hours.
    """

    def minEatingSpeed(self, piles, h):

        # The possible eating speeds, ranging from 1 up to the largest of the possible piles, meaning that every pile can be eaten with the minimal possible time
        left, right = 1, max(piles) 

        # Use Binary Search Processing Logic to find the appropriate eating speed where all the bananas can be eaten
        while left < right: 
            
            # Calculate a midpoint value using the remaining search space
            mid = (left + right) // 2

            # Calculate the amount of time Koko needs to consume all the bananas for this speed
            time_spent = sum([math.ceil(pile / mid) for pile in piles])

            # If the time spent eating was valid, set the fast eating speed to our determined value, since know it is valid
            if time_spent <= h: right = mid
            else: left = mid + 1
        
        # We want to return the minimum integer such that Koko is still able to eat all the bananas within h hours, the loop will terminate when slow == fast so the two pointers will be equal when it terminates
        return left

    """
    CAPACITY TO SHIP PACKAGES WITHIN D DAYS

    A conveyor belt has packages that must be shipped from one port to another within days days.

    The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights).

    We may not load more weight than the maximum weight capacity of the ship.

    Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.
    """

    def shipWithinDays(self, weights, days):

        # The range of possible weight capacities
        left, right = max(weights), sum(weights)

        # Prune the possible search space
        while left < right:

            # Variables to be reset at each given iteration
            mid, need, running_weight = (left + right) // 2, 1, 0

            # Perform the intermediate calculation to determine the number of days of days needed with the given weight capacity
            for w in weights:

                if running_weight + w > mid:
                    need += 1
                    running_weight = 0
                    
                running_weight += w 

            # Determine if the current calculated ship weight capacity is valid or not
            if need <= days: right = mid
            else: left = mid + 1

        # Return the least valid weight capacity
        return left 

    """
    MINIMUM NUMBER OF DAYS TO MAKE M BOUQUETS

    You are given an integer array bloomDay, an integer m and an integer k.

    You want to make m bouquets. To make a bouquet, you need to use k adjacent flowers from the garden.

    The garden consists of n flowers, the ith flower will bloom in the bloomDay[i] and then can be used in exactly one bouquet.

    Return the minimum number of days you need to wait to be able to make m bouquets from the garden. If it is impossible to make m bouquets return -1.
    """

    def minDays(self, bloomDay, m, k):

        # Establish the possible search space for the number of days needed
        left, right = 1, max(bloomDay) + 1

        # Prune the search space
        while left < right:
            
            # Determine a midpoint number of days from the remaining interval of possible days and the corresponding number of bouquets we can make
            midpoint_time = (left + right) // 2
            consecutive_flowers, bouquet_number = 0, 0

            for i in range(len(bloomDay)):

                if bloomDay[i] <= midpoint_time: consecutive_flowers += 1
                else: consecutive_flowers = 0

                if consecutive_flowers == k: 
                    bouquet_number += 1
                    consecutive_flowers = 0

            # If the number of bouquets exceeds or meets the number needed, this is a potential valid time, otherwise we must search for a greater number of days
            if bouquet_number >= m: right = midpoint_time
            else: left = midpoint_time + 1

        # Return the number of days needed if it is possible to make the number of bouquets
        return right if right != max(bloomDay) + 1 else -1

    """
    VALID PARENTHESIS STRING

    Given a string s containing only three types of characters: '(', ')' and '*', return True if s is valid.

    The following rules define a valid string:

        - Any left parenthesis '(' must have a corresponding right parenthesis, and vice versa.

        - '*' can be treated as a single right parenthesis ')' or a single left parenthesis '(', or an empty string.
    """

    def checkValidString(self, s: str):

        stack = []

        # Determine if the pairs match for the first pass - if they do consider the reverse option
        for char in s:
            if char == '(' or char == '*': stack.append(char)
            elif char == ')':
                if stack: stack.pop()
                else: return False 

        # Clear the stack for the second iteration
        stack.clear()

        # Determine if the pairs match for the second pass in reverse - if this pass is valid as well, then the string is valid
        for char in s[::-1]:
            if char == ')' or char == '*': stack.append(char)
            elif char == '(':
                if stack: stack.pop()
                else: return False 

        # If the first pass and reverse pass are both good, return True
        return True

    """
    REMOVE ALL ADJACENT DUPLICATES IN STRING II

    You are given a string s and an integer k, a k duplicate removal consists of choosing k adjacent and equal letters from s and removing them, causing the left and the right side of the deleted substring to concatenate together.

    We repeatedly make k duplicate removals on s until we no longer can.

    Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique.
    """

    def removeDuplicates(self, s, k):
        
        # Initialize a stack
        stack = deque([])

        # Iterate through all the characters in the string
        for char in s:

            # If the stack is not empty and the character at the end of the stack is the same as the current character, increment the number of occurrences, otherwise add the new character with a single occurrence
            if stack and stack[-1][0] == char: stack[-1][1] += 1
            else: stack.append([char, 1])

            # If the corresponding number of occurrences for a removal has been reached, pop from the stack
            if stack[-1][1] == k: stack.pop()

        # Return a string with the characters
        return "".join(char * freq for char, freq in stack)

    """
    MINIMUM REMOVE TO MAKE VALID PARENTHESES

    Given a string s of '(', ')' and lowercase English characters.

    Your task is to remove the minimum number of parentheses so that the resulting parentheses string is valid and return any valid string.

    Formally, a parentheses string is valid if and only if:

        - It is the empty string, contains only lowercase characters, or

        - It can be written as AB where A and B are valid strings, or

        - It can be written as (A), where A is a valid string.
    """
    def minRemoveToMakeValid(self, s):
        
        # Utilize a set to store the indices of the values to be removed and declare our stack
        indices_to_remove = set()
        stack = []

        # Iterate through the index-character pairs in the string
        for index, char in enumerate(s):
            
            # If the character is not a parenthesis we are guaranteed to keep it
            if char not in "()": continue 

            # If it is an opening parenthesis add it to the stack, otherwise determine if there is a corresponding pair
            if char == "(": stack.append(index)
            elif char == ")" and not stack: indices_to_remove.add(index)
            elif char == ")" and stack: stack.pop()

        # Update the set of indices to be removed with the remainder of the stack
        indices_to_remove |= set(stack)

        # Generate the characters after removals
        return "".join([char for index, char in enumerate(s) if index not in indices_to_remove])

    """
    MAKE THE STRING GREAT

    Given a string s of lower and upper case English letters.

    A good string is a string which doesn't have two adjacent characters s[i] and s[i + 1] where:

        - 0 <= i <= s.length - 2

        - s[i] is a lower-case letter and s[i + 1] is the same letter but in upper-case or vice-versa.

    To make the string good, you can choose two adjacent characters that make the string bad and remove them. 

    You can keep doing this until the string becomes good. Return the string after making it good. The answer is guaranteed to be unique under the given constraints.
    """
    
    def makeGood(self, s):
    
        # The stack for storing characters
        stack = []

        # Process all the characters
        for char in s:

            stack.append(char)
            
            # If there is a match as described, remove the pair
            if len(stack) > 1 and (stack[-2] == stack[-1].lower() or stack[-1] == stack[-2].lower()) and (stack[-1] != stack[-2]):
                for _ in range(2): stack.pop()

        # Return the string after all removals
        return "".join(stack)
    
    """
    LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS

    Given a string s, find the length of the longest substring without repeating characters.
    """

    def lengthOfLongestSubstring(self, s):

        window_chars, left, res = defaultdict(int), 0, 0

        for right in range(len(s)):

            window_chars[s[right]] += 1

            while window_chars[s[right]] > 1:
                window_chars[s[left]] -= 1
                left += 1

            res = max(res, right - left + 1)

        return res 

    """
    MINIMUM SIZE SUBARRAY SUM

    Given an array of positive integers nums and a positive integer target, return the minimal length of contiguous subarray of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.
    """

    def minSubArrayLen(self, target, nums):
       
        window_sum, left, min_len = 0, 0, float('inf')

        for right in range(len(nums)):
            
            window_sum += nums[right]  

            while window_sum >= target:
                min_len = min(min_len, right - left + 1)
                window_sum -= nums[left]
                left += 1
        
        return min_len if min_len <= len(nums) else 0

    """
    FIND ALL ANAGRAMS IN A STRING

    Given two strings s and p, return an array of all the start indices of p's anagrams in s. 

    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
    """

    def findAnagrams(self, s, p):

        if len(s) < len(p): return []

        p_count, window_count, res = Counter(p), Counter(), []
        
        for right in range(len(s)):

            window_count[s[right]] += 1

            if right >= len(p):
                if window_count[s[right - len(p)]] == 1: del window_count[s[right - len(p)]]
                elif window_count[s[right - len(p)]] > 1: window_count[s[right - len(p)]] -= 1
            
            if p_count == window_count: res.append(right - len(p) + 1)
        
        return res

    """
    MINIMUM OPERATIONS TO REDUCE X TO ZERO

    You are given an integer array nums and an integer x. In one operation, you can either remove the leftmost or the rightmost element from the array nums and subtract its value from x.

    Note that this modifies the array for future operations. Return the minimum number of operations to reduce x to exactly 0 if it is possible, otherwise return -1.
    """

    def minOperations(self, nums, x):
    
        left, window_sum, window_target, maximum_length = 0, 0, sum(nums) - x, -1

        # Process all the possible right pointers in the array to determine the longest possible complement subarray
        for right in range(len(nums)):
            
            # Add the value at the right pointer to current
            window_sum += nums[right]

            # Contract the window if the sum of elements in the window is currently too great and the window is currently valid
            while window_sum > window_target and left <= right:
                window_sum -= nums[left]
                left += 1

            # If we have found a valid complement, record the new length of the current window if it surpasses the previous length 
            if window_sum == window_target: maximum_length = max(maximum_length, right - left + 1)

        # If we found a compatiable consecutive subarray that produced the complement value, return the number of operations otherwise return -1 to indicate that it is not possible
        return len(nums) - maximum_length if maximum_length != -1 else -1
    
    """
    SUBSTRINGS OF SIZE THREE WITH DISTINCT CHARACTERS

    A string is good if there are no repeated characters.

    Given a string s, return the number of good substrings of length three in s.

    Note that if there are multiple occurrences of the same substring, every occurrence should be counted.

    A substring is a contiguous sequence of characters in a string.
    """

    def countGoodSubstrings(self, s):

        window_characters, good_substrings = defaultdict(int), 0

        # Sliding the window rightwards
        for right in range(len(s)):

            # Add the corresponding character to the window
            window_characters[s[right]] += 1

            # Window must be contracted at each iteration once it has been expanded to include three characters, determine if the character should be deleted entirely or if the occurrences should be decremented
            if right > 2: 
                if window_characters[s[right - 3]] == 1: del window_characters[s[right - 3]]
                else: window_characters[s[right - 3]] -= 1

            # If there are three unique characters for the window, increment the count of good substrings
            if len(set(window_characters.keys())) == 3: good_substrings += 1

        # Return the number of "good" substrings after considering the possible windows
        return good_substrings
    
    """
    PERMUTATION IN STRING

    Given two strings s1 and s2, return True if s2 contains a permutation of s1, or False otherwise.

    In other words, return True if one of s1's permutations is the substring of s2.
    """
    def checkInclusion(self, s1: str, s2: str) -> bool:
        
        counts1, window_counts, window_size = Counter(s1), Counter(), len(s1)

        # Slide the window rightwards
        for right in range(len(s2)):

            # Increment the number of occurrences of the character at the right end of the current window
            window_counts[s2[right]] += 1

            # If the right end of the window has exceeded the possible size of the window we must narrow the fixed size sliding window
            if right > window_size - 1:
                if window_counts[s2[right - window_size]] > 1: window_counts[s2[right - window_size]] -=1
                else: del window_counts[s2[right - window_size]]

            # If the occurrence of characters are identical, we have found a permutation
            if counts1 == window_counts: return True 

        return False
            
    """
    MAXIMUM POINTS YOU CAN OBTAIN FROM CARDS

    There are several cards arranged in a row, and each card has an associated number of points. 

    The points are given in the integer array cardPoints. In one step, you can take one card from the beginning or from the end of the row.

    You have to take exactly k cards. Your score is the sum of the points of the cards you have taken.

    Given the integer array cardPoints and the integer k, return the maximum score you can obtain.
    """

    def maxScore(self, cardPoints, k):

        window_length, window_sum, min_sum = len(cardPoints) - k, 0, float('inf')

        for right in range(len(cardPoints)):
            
            # Update the window sum
            window_sum += cardPoints[right]

            # If the right end of the interval has exceeded the length of the interval we must decrement by the value at the left end
            if right > window_length - 1:
                window_sum -= cardPoints[right - window_length]

            # In this case we can update with a new minimum sum
            if right >= window_length - 1:
                min_sum = min(min_sum, window_sum)

        return sum(cardPoints) - min_sum

    """
    LONGEST SUBSTRING WITH AT MOST TWO DISTINCT CHARACTERS

    Given a string s, return the length of the longest substring that contains at most two distinct characters.
    """

    def lengthOfLongestSubstringTwoDistinct(self, s):

        # Use a mapping to store the occurrences of characters in the current window, a variable to store the current longest substring with the conditions and a variable for the left end of the interval
        window_counts, max_length, left = Counter(), 0, 0

        # At each iteration slide the window rightwards
        for right in range(len(s)):

            # Increment the occurrences of the character at the right end of the window
            window_counts[s[right]] += 1

            # So long as there are more than two distinct character entries in the mapping we must narrow the window
            while len(window_counts) > 2:
                if window_counts[s[left]] == 1: del window_counts[s[left]]
                elif window_counts[s[left]] > 1: window_counts[s[left]] -= 1
                left += 1

            # Calculate the size of the current window
            max_length = max(max_length, right - left + 1)
        
        # Return the length of the longest substring with at most two distinct characters
        return max_length
    
    """
    NUMBER OF ISLANDS

    Given an m x n 2D binary grid which represents a map of '1s' and '0s', return the number of islands.

    An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

    You may assume all four edges of the grid are all surrounded by water.
    """

    def numIslands(self, grid):

        def island_search(r, c):

            if not (0 <= r < m) or not (0 <= c < n) or grid[r][c] != '1': return 
            grid[r][c] = '_'
            for nr, nc in [(1, 0), (-1, 0), (0, 1), (0, -1)]: island_search(r + nr, c + nc)

        if not grid or not grid[0]: return 0

        m, n, res = len(grid), len(grid[0]), 0

        for r in range(m):
            for c in range(n):
                if grid[r][c] == '1':
                    island_search(r, c)
                    res += 1

        return res

    def numIslands2(self, grid):

        def mark_island(r, c):

            queue = deque([(r, c)])
            grid[r][c] = '0'

            while queue: 

                r, c = queue.popleft()

                for nr, nc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    if 0 <= r + nr < m and 0 <= c + nc < n and grid[r + nr][c + nc] == '1':
                        grid[r + nr][c + nc] = '0'
                        queue.append((r + nr, c + nc))

        m, n = len(grid), len(grid[0])
        res = 0
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == '1': 
                    mark_island(r, c)
                    res += 1

        return res 

    """
    NUMBER OF CLOSED ISLANDS 

    Given a 2D grid consists of 0s (land) and 1s (water).

    An island is a maximal 4-directionally connected group of 0s and a closed island is an island totally surrounded by 1s.

    Return the number of closed islands.
    """

    def closedIsland(self, grid):

        def mark_tile(r, c):

            if not (0 <= r < m) or not (0 <= c < n) or grid[r][c] != 0: return 
            grid[r][c] = 1
            for nr, nc in [(0, 1), (1, 0), (0, -1), (-1, 0)]: mark_tile(r + nr, c + nc)

        if not grid or not grid[0]: return 0

        m, n, res = len(grid), len(grid[0]), 0

        for r in range(m):
            for c in range(n): 
                if (r in [0, m - 1] or c in [0, n - 1]) and grid[r][c] == 0: mark_tile(r, c)

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0: 
                    mark_tile(r, c)
                    res += 1

        return res 

    def closedIsland2(self, grid):
        """
        BFS implementation.
        """

        def traverse_tiles(r, c):

            queue = deque([(r, c)])
            grid[r][c] = '_'

            while queue: 

                r, c = queue.popleft()

                for nr, nc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if 0 <= r + nr < m and 0 <= c + nc < n and grid[r + nr][c + nc] == 0:
                        grid[r + nr][c + nc] = '_'
                        queue.append((r + nr, c + nc))

        m, n = len(grid), len(grid[0])
        res = 0

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0 and (r in [0, m - 1] or c in [0, n - 1]): traverse_tiles(r, c)

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0:
                    traverse_tiles(r, c)
                    res += 1

        return res

    """
    FIND IF PATH EXISTS IN GRAPH

    There is a bi-directional graph with n vertices, where each vertex is labeled from 0 to n - 1 (inclusive). The edges are represented as a 2D integer array edges. 

    Each vertex pair is connected by at most one edge, and no vertex has an edge to itself. You want to determine if there is a valid path that exists from vertex source to vertex destination. 

    Given edges and the integers n, source, and destination, return True if there is a valid path from source to destination, or False otherwise.
    """

    def validPath(self, _, edges, source, destination):
        """
        DFS solution on an adjacency set graph.
        """

        def find_path(current_node, target_node):
            """
            Recursive DFS-based helper function to determine a potential path from the starting node to the ending node.
            Function takes two parameters, the node for the given function call and an ending node for the destination.
            """
            
            # Recursive base cases - we have either found a possible path or there are loops indicating that it is impossible to determine a valid path
            if current_node == target_node: return True 
            if current_node in seen: return False 

            # Add the node for the given recursive call to the set 
            seen.add(current_node)

            # Determine the neighbors of the current node
            current_neighbors = neighbors[current_node]

            # Explore further using DFS to determine if any of the calls can find the target node
            for neighbor in current_neighbors:
                if find_path(neighbor, target_node): return True

            # Otherwise return False for the current recursive call if none of the explorations found a valid path
            return False 

        # Mapping to store the list of neighbors for a given node along with a caching set as is common with recursive explorations - the mapping needs to have a default value of an empty list
        neighbors, seen = defaultdict(list), set()

        # Construct the actual graph using the nodes and associated connections
        for node1, node2 in edges:
            neighbors[node1].append(node2)
            neighbors[node2].append(node1)

        # Call the recursive DFS helper with the starting node and ending node to recursively explore the possible paths
        return find_path(source, destination)

    """
    KEYS AND ROOMS

    There are n rooms labeled from 0 to n - 1 and all the rooms are locked except for room 0.

    Your goal is to visit all the rooms. However, you cannot enter a locked room without having its key.

    When you visit a room, you may find a set of distinct keys in it. Each key has a number on it, denoting which room it unlocks, and you can take all of them with you to unlock the other rooms.

    Given an array rooms where rooms[i] is the set of keys that you can obtain if you visited room i, return True if you can visit all the rooms, or False otherwise.
    """

    def canVisitAllRooms(self, rooms):
        
        # A list representing the rooms we can possible visit
        seen = [True] + [False for _ in range(len(rooms) - 1)]

        # For any key that hasn't been used yet, add it to the todo list with the stack
        stack = [0]

        # Continue iterating so long as there are keys with rooms to visit
        while stack:
            
            # Retrieve the next key
            key = stack.pop()

            # Consider every other key in the given room
            for nei in rooms[key]:

                # If we haven't visited the room yet, mark it as visited and add it for future exploration
                if not seen[nei]:
                    seen[nei] = True 
                    stack.append(nei)

        # Return True if we've managed to visit every possible room
        return all(seen)
    
    """
    K HIGHEST RANKED ITEMS WITHIN A PRICE RANGE

    Return the k highest-ranked items within the price range sorted by their rank (highest to lowest).
    """

    def highestRankedKItems(self, grid, pricing, start, k):

        m, n = len(grid), len(grid[0])
        queue = deque([(0, start[0], start[1])])
        seen = set([(start[0], start[1])])
        valid_tiles = []
        min_price, max_price = pricing[0], pricing[1]

        while queue:

            steps, r, c = queue.popleft()

            if min_price <= grid[r][c] <= max_price: 
                valid_tiles.append([steps, grid[r][c], r, c])

            for nr, nc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:

                if 0 <= r + nr < m and 0 <= c + nc < n and (r + nr, c + nc) not in seen and grid[r + nr][c + nc] != 0:
                    queue.append((steps + 1, r + nr, c + nc))
                    seen.add((r + nr, c + nc))

        valid_tiles.sort(key = lambda x: (x[0], x[1], x[2], x[3]))

        return [[valid_tiles[i][2], valid_tiles[i][3]] for i in range(min(len(valid_tiles), k))]

    """
    DESTROYING ASTEROIDS

    Return True if all asteroids can be destroyed.

    Otherwise, return False.
    """

    def asteroidsDestroyed(self, mass, asteroids):

        asteroids.sort()

        current_size = mass

        for asteroid in asteroids:

            if current_size < asteroid: return False 
            else: current_size += asteroid

        return True

    """
    PATH WITH MAXIMUM MINIMUM VALUE 

    Given an m x n integer matrix grid, return the maximum score of a path starting at (0, 0) and ending at (m - 1, n - 1) moving in the 4 cardinal directions.

    The score of a path is the minimum value in that path.
    """

    def maximumMinimumPath(self, grid):

        heap = [(-grid[0][0], 0, 0)]
        seen = set([(0, 0)])
        res = float('inf')
        m, n = len(grid), len(grid[0])

        while heap:

            heap_val, r, c = heapq.heappop(heap)

            val = -heap_val

            res = min(res, val)

            if r == m - 1 and c == n - 1: return res

            for nr, nc in [(0, 1), (1, 0), (-1, 0), (0, -1)]:

                if 0 <= r + nr < m and 0 <= c + nc < n and (r + nr, c + nc) not in seen:

                    heapq.heappush(heap, (-grid[r + nr][c + nc], r + nr, c + nc))

                    seen.add((r + nr, c + nc))

    """
    DETECT CYCLES IN A 2D GRID

    Return True if any cycle of the same value exists in grid, otherwise return False.
    """

    def containsCycle(self, grid):

        def dfs(node, parent):

            if node in visited: return True
            visited.add(node)
            nx,ny = node

            neighbours = [(cx, cy) for cx, cy in [[nx + 1, ny],[nx - 1, ny],[nx, ny + 1], [nx, ny - 1]] if 0 <= cx < m and 0 <= cy < n and grid[cx][cy] == grid[nx][ny] and (cx,cy) != parent]

            for x in neighbours:
                if dfs(x, node): return True 

            return False  
    
        m, n = len(grid), len(grid[0])
        visited = set()

        for i in range(m):
            for j in range(n):
                if (i,j) in visited: continue 
                if dfs((i,j), None): return True

        return False 

    """
    FIND THE INDEX OF THE FIRST OCCURRENCE IN A STRING

    Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
    """

    def strStr(self, haystack, needle):
  
        left = 0

        # Slide the right pointer
        for right in range(len(haystack)):
            
            # If the right pointer is sufficiently slid to accomodate a window
            if right >= len(needle) - 1: 
                
                # Determine the current window
                current_slice = haystack[left:right + 1]

                # If the window is equal to the pattern we want to find, return the left index
                if current_slice == needle: return left
                
                # Otherwise we need to slide the left pointer as well
                left += 1

        return -1

    """
    REPEATED SUBSTRING PATTERN

    Given a string s, check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.
    """

    def repeatedSubstringPattern(self, s):
        N = len(s)
        return any([N % i == 0 and s[:i] * (N // i) == s for i in range(1, N // 2 + 1)])

    """
    STRING TO INTEGER (ATOI)

    Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).
    """

    def myAtoi(self, s: str):
    
        s = s.strip()

        res = []
        for char in s:

            if (char == '-' or char == '+') and not res: res.append(char)
            elif char.isdigit(): res.append(char)
            elif not char.isdigit(): break 

        if not res or set(res) == {'-'} or set(res) == {'+'}: return 0
        if res[0] == '+': res.pop(0)
        
        pruned_res = res[:]
        prev_char = None
        for char in res:
            if prev_char == '0' and char in ['-', '+']: return 0
            if char == '0': pruned_res.remove('0')
            elif char != '0': break
            prev_char = char

        if not pruned_res: return 0

        num_res = int("".join(pruned_res))

        if num_res < -(2 ** 31): return -(2 ** 31)
        elif num_res > (2 ** 31 - 1): return 2 ** 31 - 1
        else: return num_res
