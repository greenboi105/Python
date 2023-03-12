import math 
import heapq
from collections import *
from itertools import *

class Solution:
    
    """
    MAXIMUM SPLIT OF POSITIVE EVEN INTEGERS

    You are given an integer finalSum. Split it into a sum of maximum number of unique positive even integers.

    Return a list of integers that represent a valid split containing a maximum number of integers. 
    
    If no valid split exists for finalSum, return an empty list. You may return the integers in any order.
    """

    def maximumEvenSplit(self, finalSum):

        # If the value is odd, we immediately know that it cannot be split into a series of even integers.
        if finalSum % 2 == 1: return []

        # Return container and beginning positive even amount
        ans, current = [], 2

        # Continue iterating so long as i is less than or equal to finalSum
        while current <= finalSum: 
            ans.append(current)
            finalSum -= current
            current += 2

        # Increment the final amount in the ans container by the remainder of finalSum in case we could not form another distinct integer
        ans[-1] += finalSum 

        # Return the ans container
        return ans
    
    """
    FIND THE WINNER OF THE CIRCULAR GAME

    There are n friends that are playing a game. The friends are sitting in a circle and are numbered from 1 to n in clockwise order. 

    More formally, moving clockwise from the i-th friend brings you to the (i + 1)th friend for 1 <= i < n, and moving clockwise from the nth friend brings you to the 1st friend.

    The rules of the game are as follows:

        1. Start at the 1st friend.

        2. Count the next k friends in the clockwise direction including the friend you started at. The counting wraps around the circle and may count some friends more than once.

        3. The last friend you counted leaves the circle and loses the game.

        4. If there is still more than one friend in the circle, go back to step 2 starting from the friend immediately clockwise of the friends who just lost and repeat.

        5. Else, the last friend in the circle wins the game.

    Given the number of friends, n, and an integer k, return the winner of the game.
    """

    def findTheWinner(self, n, k):
   
        circle, lastIndex = [i for i in range(1, n + 1)], 0

        # Implement and repeat the game algorithm until there is only one person left
        while len(circle) > 1:
            
            # Calculate the index of the friend to leave using the prescribed algorithm
            lastIndex = (lastIndex + k - 1) % len(circle)

            # Remove the value of last index from the list of circle elements
            circle.pop(lastIndex)

            # Re-determine the relative index given the deletion in the new circle after removal
            lastIndex = lastIndex % len(circle)

        # Return the lone friend remaining
        return circle.pop()
        
    """
    POWER OF TWO

    Given an integer n, return True if it is a power of two. Otherwise, return False
    """
    
    def isPowerOfTwo(self, n):
        if n == 0 or (n % 2 == 1 and n != 1): return False 
        while n % 2 == 0: n /= 2
        return n == 1
        
    """
    SORT TRANSFORMED ARRAY

    Given a sorted integer array nums and three integers a, b and c, apply a quadratic function of the form f(x) = ax^2 + bx + c to each elements nums[i] in the array, and return the array in a sorted order.
    """

    def sortTransformedArray(self, nums, a, b, c):

        # The polynomial transformation
        def poly(val):
            return a * val ** 2 + b * val + c 
            
        # Return a sorted container of the numbers after the transformation
        return sorted(poly(nums[i]) for i in range(len(nums)))
    
    """
    COUNT INTEGERS WITH EVEN DIGIT SUM

    Given a positive integer num, return the number of positive integers less than or equal to num whose digit sums are even.

    The digit sum of a positive integer is the sum of all its digits.
    """
    
    def countEven(self, num: int):

        def digit_sum(num):
            return sum([int(digit) for digit in str(num)])

        return sum([1 if digit_sum(val) % 2 == 0 else 0 for val in range(1, num + 1)])
    
    """
    DETERMINE COLOR OF A CHESSBOARD SQUARE

    Return True if the square is white, and False if the square is black.
    """

    def squareIsWhite(self, coordinates: str):
        
        # Determine the relative x coordinate in the grid based on the letter character and the y coordinate in the grid
        x_coord, y_coord = ord(coordinates[0]) - ord('a') + 1, int(coordinates[1])
        
        # Two possibilities - the x coordinate is odd or even
        if x_coord % 2 == 1:

            # Tiles with a relative odd x and y coordinate are False, while tiles with an odd x coordinate and even y coordinate are True
            if y_coord % 2 == 1: return False
            else: return True 
        
        elif x_coord % 2 == 0:
            
            # Tiles with a relative even x coorindate and odd y coordinate are True, while the opposite is False
            if y_coord % 2 == 1: return True 
            else: return False
        
    """
    THE KTH FACTOR OF N

    Consider a list of all factors of n sorted in ascending order, return the kth factor in this list or -1 if n has less than k factors.
    """

    def kthFactor(self, n: int, k: int):

        # Generate a list of all the factors of the number in the valid range
        factors = [val for val in range(1, n + 1) if n % val == 0]

        # Return the kth factor
        return factors[k - 1] if len(factors) >= k else -1

    """
    NUMBER OF COMMON FACTORS

    Given two positive integers a and b, return the number of common factors of a and b.
    """

    def commonFactors(self, a: int, b: int):
        
        def getfactors(num):

            factors, i = set(), 1

            while i * i < num:
                if num % i == 0: factors |= {num // i, i}
                i += 1

            if i * i == num: factors.add(i)

            return factors
        
        # Return the number of common elements
        return len(getfactors(a) & getfactors(b))

    """
    FOUR DIVISORS

    Given an integer array nums, return the sum of divisors of the integers in that array that have exactly four divisors.

    If there is no such integer in the array, return 0.
    """

    def sumFourDivisors(self, nums):

        def get_divisors(num):

            divisor_sum, count, i = 0, 0, 1
            
            while i * i < num:
                if num % i == 0: 
                    divisor_sum += (i + num // i)
                    count += 2
                i += 1
                    
            if i * i == num: 
                divisor_sum += i
                count += 1
            
            return divisor_sum, count
        
        # Running variable for the sum of the divisors of the integers in the list that have exactly four divisors
        res = 0 
        
        # Single pass through the list
        for num in nums:
            div_sum, count = get_divisors(num)
            if count == 4: res += div_sum
        
        # Return the sum of the divisors
        return res
    
    """
    BINARY GAP

    Given a positive integer n, find and return the longest distance between any two adjacent 1's in the binary representation of n.

    If there are no two adjacent 1's, return 0.
    """

    def binaryGap(self, n):
        index = [i for i, v in enumerate(bin(n)) if v == '1']
        return max(index[i] - index[i - 1] for i in range(1, len(index))) if len(index) > 1 else 0

    """
    CLOSEST DIVISORS

    Given an integer num, find the closest two integers in absolute difference whose product equals num + 1 or num + 2.

    Return the two integers in any order.
    """

    def closestDivisors(self, num: int):

        def calculateDifferenceProduct(value):

            res, i, diff = [], 1, float('inf')

            while i * i < value:

                if value % i == 0:
                    temp_diff = abs(value // i - i)
                    if temp_diff < diff:
                        diff = temp_diff 
                        res = [value // i, i]

                i += 1

            if i * i == value: return [i, i], 0
            else: return res, diff

        # Call the helper and determine the corresponding results using tuple unpacking
        first_result, first_difference = calculateDifferenceProduct(num + 1)
        second_result, second_difference = calculateDifferenceProduct(num + 2)

        # Return using a conditional check
        return first_result if first_difference <= second_difference else second_result

    """
    PRODUCT OF ARRAY EXCEPT SELF

    Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

    You must write an algorithm that runs in O(N) time and without using the division operation.
    """
    
    def productExceptSelf(self, nums: list):
        
        # Problem variables to track
        zero_count, total_product, no_zero_product = 0, 1, 1
        
        # Iterate through the numbers
        for num in nums: 
            if num == 0: zero_count += 1
            if zero_count >= 2: return [0] * len(nums)
            total_product *= num
        
        # In the case there is only one zero
        for num in nums: 
            if num == 0: continue 
            no_zero_product *= num
        
        # Essentially we have three possibilities - there are more than a single zero, there is a single zero or there are no zeroes
        if zero_count == 0: return [total_product // num for num in nums]
        else: return [0 if num != 0 else no_zero_product for num in nums]
    
    """
    FIND TRIANGULAR SUM OF AN ARRAY

    You are given a 0-indexed integer array nums, where nums[i] is a digit between 0 and 9 (inclusive).

    The triangular sum of nums is the value of the only element present in nums after the following process terminates:

    Let nums comprise of n elements. If n == 1, end the process. Otherwise, create a new 0-indexed integer array newNums of length n - 1.

    For each index i, where 0 <= i < n - 1, assign the value of newNums[i] as (nums[i] + nums[i + 1]) % 10, where % denotes modulo operator.

    Replace the array nums with newNums. Repeat the entire process starting from step 1. Return the triangular sum of nums.
    """

    def triangularSum(self, nums):

        # If there is only a single element to begin, simply return that element
        if len(nums) == 1: return nums[0]

        def calculateTriangular(arr):
            res = []
            for i in range(1, len(arr)):
                addition_sum = arr[i] + arr[i - 1]
                res.append(addition_sum % 10)
            return res 

        # Begin with a copy of nums
        current_list = nums[:]

        while True:
            triangular_sum = calculateTriangular(current_list)
            if len(triangular_sum) == 1: return triangular_sum[0]
            else: current_list = triangular_sum

    """
    PASCAL'S TRIANGLE

    Given an integer numRows, return the first numRows of Pascal's triangle.

    In Pascal's triangle, each number is the sum of the two numbers directly above it.
    """

    def generate(self, numRows):

        triangle = []

        for row in range(numRows):

            current_row = [None for _ in range(row + 1)]

            current_row[0], current_row[-1] = 1, 1

            for column in range(1, len(current_row) - 1):
                current_row[column] = triangle[row - 1][column] + triangle[row - 1][column - 1]

            triangle.append(current_row)

        return triangle

    """
    COUNT NUMBER OF DISTINCT INTEGERS AFTER REVERSE OPERATIONS

    You are given an array nums consisting of positive integers.

    You have to take each integer in the array, reverse its digits, and add it to the end of the array. 

    You should apply this operation to the original integers in nums. 

    Return the number of distinct integers in the final array.
    """

    def countDistinctIntegers(self, nums) -> int: 
        """
        Utilize a helper for reversing numbers, add the reverse numbers to the end of the original list and return the number of unique elements.
        """

        def reverseDigits(number):
            return int("".join([digit for digit in str(number)])[::-1])
        
        res = nums[:]
        for num in nums: res.append(reverseDigits(num))

        return len(set(res))

    """
    MINIMUM OPERATIONS TO MAKE ARRAY EQUAL

    You have an array arr of length n where arr[i] = (2 * i) + 1 for all valid values of i (i.e., 0 <= i < n).

    In one operation, you can select two indices x and y where 0 <= x, y < n and subtract 1 from arr[x] and add 1 to arr[y].

    The goal is to make all the elements of the array equal. It is guaranteed that all the elements of the array can be made equal using some operations.

    Given an integer n, the length of the array, return the minimum number of operations needed to make all the elements of arr equal.
    """

    def minOperations(self, n):
        """
        Determine a midpoint value and calculate the number of operations needed to make all elements equal.
        """

        arr = [i for i in range(1, (2 * n + 1), 2)]
        target, res = sum(arr) // n, 0

        for num in arr: res += abs(num - target)

        return res // 2

    """
    SUM OF SQUARE NUMBERS

    Given a non-negative integer c, decide whether there're two integers a and b such that a ** 2 + b ** 2 = c.
    """

    def judgeSquareSum(self, c: int):
        """
        Determine an interval of numbers that can potentially sum to form c and progressively narrow the interval.
        """

        left, right = 0, math.ceil(math.sqrt(c))
        
        while left <= right:

            current_sum = left ** 2 + right ** 2

            if current_sum == c: return True 
            elif current_sum < c: left += 1
            elif current_sum > c: right -= 1

        return False

    """
    MAXIMUM SUBARRAY

    Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum. A subarray is a contiguous part of an array. 
    """

    def maxSubArray(self, nums):

        running_val, max_val = nums[0], nums[0]

        # Iterate for all the numbers from the second onwards
        for num in nums[1:]:
            running_val = max(num, running_val + num)
            max_val = max(max_val, running_val)
        
        # Return sum of the maximum subarray after considering all updates
        return max_val

    """
    MAXIMUM SIZE SUBARRAY SUM EQUALS K

    Given an integer array nums and an integer k, return the maximum length of a subarray that sums to k. If there is not such an array, return 0.
    """
    
    def maxSubArrayLen(self, nums, k):

        mapping, running_sum, res = defaultdict(int), 0, 0

        # Process all the numbers in the list via the list indices
        for i in range(len(nums)):
            
            # Update the running sum with the current value of nums for the iteration's index
            running_sum += nums[i]

            # Determine if there is a chance to update the length of the subarray if (this occurs when either the running sum is equal to the target or there is an index with a running sum equal to the required complement, value is only updated if the new length exceeds the previous)
            if running_sum == k: res = max(res, i + 1)
            elif running_sum - k in mapping: res = max(res, i - mapping[running_sum - k])

            # If the value of curr_sum does not yet exist in our mapping, store the corresponding (sum, index) pair in the mapping
            if running_sum not in mapping: mapping[running_sum] = i 
        
        # Return the maximum length of a subarray that sums to k
        return res
    
    """
    COUNT NUMBER OF TEAMS

    There are n soldiers standing in a line. Each solider is assigned a unique rating value.

    You have to form a team of 3 soldiers amongst them under the rules:

        - Choose 3 soldiers with index (i, j, k) with rating (rating[i], rating[j], rating[k]).

        - A team is valid if: (rating[i] < rating[j] < rating[k]) or (rating[i] > rating[j] > rating[k]) where (0 <= i < j < k < n).

    Return the number of teams you can form given the conditions. 
    """

    def numTeams(self, rating):

        # If the size of the ratings container is less than 3, return 0 immediately (common immediate logic check and return)
        if len(rating) < 3: return 0 

        # Utilize two hashmaps to store values greater and less for a given index to determine the valid triplets of soldiers 
        greater, less, num_teams = defaultdict(int), defaultdict(int), 0

        # First determine all the values with values less than or greater than the relative index
        for inner in range(len(rating) - 1):
            for outer in range(inner + 1, len(rating)):
                if rating[outer] > rating[inner]: greater[inner] += 1
                elif rating[outer] < rating[inner]: less[inner] += 1

        # Updating the value for the number of teams using the calculated base cases and the necessary ordering
        for inner in range(len(rating) - 2):
            for outer in range(inner + 1, len(rating)):
                if rating[inner] < rating[outer]: num_teams += greater[outer]
                elif rating[inner] > rating[outer]: num_teams += less[outer]

        # Return the number of valid teams after processing all the possible relative orderings
        return num_teams

    """
    FIND AND REPLACE PATTERN

    Given a list of strings words and a string pattern, return a list of words[i] that match pattern. 

    You may return the answer in any order. A word matches the pattern if there exists a permutation of letters p so that after replacing every letter x in the pattern with p(x), we get the desired word.

    Recall that a permutation of letters is a bijection from letters to letters: every letter maps to another letter, and no two letters map to the same letter.
    """

    def findAndReplacePattern(self, words, pattern) -> list[str]:
        
        def match(word):
            map1, map2 = {}, {}

            # Process the corresponding characters between the current word and the pattern word
            for w, p in zip(word, pattern):
                if w not in map1: map1[w] = p
                if p not in map2: map2[p] = w 
                if (map1[w], map2[p]) != (p, w): return False 
            return True 

        # Check to see if the words in the list match the given pattern
        return [word for word in words if match(word)]

    """
    FIND PLAYERS WITH ZERO OR ONE LOSSES

    You are given an integer array matches where matches[i] = [winner_i, loser_i] indicates that the player winner_i defeated player loser_i in a match.

    Return a list answer of size 2 where:

        - answer[0] is a list of all players that have not lost any matches.

        - answer[1] is a list of all players that have lost exactly one match.

    The values in the two lists should be returned in increasing order.
    """

    def findWinners(self, matches):
        """
        Utilize three sets to be continually updated as we process winners and losers.
        We need a set to store winners with no losses, winners with one loss and winners with more than a single loss.
        """
        
        # We require three sets to store players with the corresponding number of losses
        zero_loss, one_loss, more_loss = set(), set(), set()

        # Iterate through the matches with a given winner and loser
        for winner, loser in matches:
            
            # If the winner has not received a loss yet add the player to the set with no losses
            if (winner not in one_loss) and (winner not in more_loss): zero_loss.add(winner)

            # Otherwise update the sets depending on the current number of losses for a given player
            if loser in zero_loss:
                zero_loss.remove(loser)
                one_loss.add(loser)
            elif loser in one_loss:
                one_loss.remove(loser)
                more_loss.add(loser)
            elif loser in more_loss: 
                continue 
            else: 
                one_loss.add(loser)

        # Return the players in sorted order
        return [sorted(list(zero_loss)), sorted(list(one_loss))]

    """
    LONGEST HAPPY STRING

    Given the three integers a, b, and c, return the longest possible happy string. If there are multiple longest happy strings, return any of them. 
    
    If there is no such string, return the empty string "". A substring is a contiguous sequence of characters within a string.
    """

    def longestDiverseString(self, a, b, c):
        character_map, longest_happy = [[a, 'a'], [b, 'b'], [c, 'c']], []

        while True:
            character_map.sort(key = lambda x: x[0])
            character_index = 1 if len(longest_happy) >= 2 and longest_happy[-2] == longest_happy[-1] == character_map[2][1] else 2
            if character_map[character_index][0] > 0:
                longest_happy.append(character_map[character_index][1])
                character_map[character_index][0] -= 1
            else: break 
    
        return "".join(longest_happy)
    
    """
    SURROUNDED REGIONS

    Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'.

    A region is captured by flipping all 'O's into 'X's in that surrounded region. 
    """

    def solve(self, board):
     
        m, n = len(board), len(board[0])
    
        queue = deque([(r, c) for r, c in product(range(m), range(n)) if (r in [0, m - 1] or c in [0, n - 1]) and board[r][c] == "O"])

        while queue:

            row, col = queue.popleft()

            board[row][col] = "E"

            for nr, nc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= row + nr < m and 0 <= col + nc < n and board[row + nr][col + nc] == "O": queue.append((row + nr, col + nc))
        
        for r, c in product(range(m), range(n)):
            if board[r][c] == 'O': board[r][c] = 'X'
            elif board[r][c] == 'E': board[r][c] = 'O'

    """
    01 MATRIX

    Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell. The distance between two adjacent cells is 1.
    """

    def updateMatrix(self, mat):

        m, n = len(mat), len(mat[0])

        queue = deque([])
        for r in range(m):
            for c in range(n):
                if mat[r][c] == 0: queue.append((r, c))
                if mat[r][c] == 1: mat[r][c] = 'U'

        while queue:

            r, c = queue.popleft()

            for nr, nc in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                if 0 <= r + nr < m and 0 <= c + nc < n and mat[r + nr][c + nc] == 'U':
                    mat[r + nr][c + nc] = mat[r][c] + 1
                    queue.append((r + nr, c + nc))

        return mat

    """
    WALLS AND GATES

    You are given an m x n grid rooms initialized with three possible values:
    
        - -1: A wall or an obstacle (this is impassable) 

        - 0: A gate (what we want to find the distance from)

        - INF: Infinity means an empty room (this value is either converted to an integer or left as is if the room cannot be accessed)
    
    Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
    """

    def wallsAndGates(self, rooms):

        m, n = len(rooms), len(rooms[0])
        
        queue = deque([(r, c) for r, c in product(range(m), range(len(n))) if rooms[r][c] == 0])

        while queue:
            
            r, c = queue.popleft()

            for nr, nc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:

                if 0 <= r + nr < m and 0 <= c + nc < n and rooms[r + nr][c + nc] == 2147483647:
                    rooms[r + nr][c + nc] = rooms[r][c] + 1
                    queue.append((r + nr, c + nc))

    """
    SHORTEST PATH IN BINARY MATRIX

    Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is not clear path, return -1.

    A clear path in a binary matrix is a path from the top-left cell to the bottom-right cell such that:

        - All the visited cells of the path are 0.

        - All the adjacent cells of the path are 8-directionally connected (i.e., they are different any they share an edge or a corner).

    The length of a clear path is the number of visited cells in this path.
    """
 
    def shortestPathBinaryMatrix(self, grid):
 
        if grid[0][0] == 1: return -1 

        n, queue, seen = len(grid), deque([(1, 0, 0)]), set()

        while queue:

            dist, r, c = queue.popleft()

            if (r, c) == (n - 1, n - 1): return dist

            for nr, nc in [(-1, 1), (-1, -1), (1, 1), (1, -1), (0, -1), (-1, 0), (1, 0), (0, 1)]:

                if 0 <= r + nr < n and 0 <= c + nc < n and grid[r + nr][c + nc] == 0 and (r + nr, c + nc) not in seen:
                    queue.append((dist + 1, r + nr, c + nc))
                    seen.add((r + nr, c + nc))

        return -1

    """
    SWIM IN RISING WATER

    You are given an n x n integer matrix grid where each value grid[i][j] represents the elevation at that point (i, j).

    The rain starts to fall. At time t, the depth of the water everywhere is t. You can swim from a square to another 4-directionally adjacent square if and only if the elevation of both squares individually are at most t.

    You can swim infinite distances in zero time. You must stay within the boundaries of the grid during your swim. Return the least time until you can reach the bottom right square (n - 1, n - 1) if you start at the top left square (0, 0).
    """

    def swimInWater(self, grid):

        n = len(grid)
        res = 0
        heap = [(grid[0][0], 0, 0)]
        seen = set([(0, 0)])

        for _ in range(n * n):
            
            time, r, c = heapq.heappop(heap)

            res = max(res, time)

            if r == n - 1 and c == n - 1: return res

            for nr, nc in [(0, 1), (1, 0), (-1, 0), (0, -1)]:

                if 0 <= r + nr < n and 0 <= c + nc < n and (r + nr, c + nc) not in seen:
                    heapq.heappush(heap, (grid[r + nr][c + nc], r + nr, c + nc))
                    seen.add((r + nr, c + nc))

    """
    SHORTEST PATH TO GET FOOD

    You are starving and you want to eat food as quickly as possible. You want to find the shortest path to arrive at any food cell.

    You are given an m x n character matrix, grid, of these different types of cells:

        - '*' is your location. There is exactly one '*' cell.

        - '#' is a food cell. There may be multiple food cells.

        - 'O' is free space, and you can travel through these cells.

        - 'X' is an obstacle, and you cannot travel through these cells.

    You can travel to any adjacent cell north, east, south or west of your current location if there is not an obstacle.

    Return the length of the shortest path for you to reach any food cell.
    """

    def getFood(self, grid):

        m, n = len(grid), len(grid[0])

        queue = deque([(0, i, j) for i, j in product(range(m), range(n)) if grid[i][j] == '*'])

        while queue:

            steps, r, c = queue.popleft()

            for nr, nc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:

                if 0 <= r + nr < m and 0 <= c + nc < n and grid[r + nr][c + nc] in '#O':
                    if grid[r + nr][c + nc] == '#': return steps + 1
                    grid[r + nr][c + nc] = '_'
                    queue.append((steps + 1, r + nr, c + nc))

        return -1
    
    """
    NEAREST EXIT FROM ENTRANCE IN MAZE

    You are given an m x n matrix maze (0-indexed) with empty cells (represented as '.') and walls (represented as '+').

    You are also given the entrance of the maze, where entrance = [entrance_row, entrance_col] denotes the row and column of the cell you are initially standing at.

    In one step, you can move one cell up, down, left or right. You cannot step into a cell with a wall, and you cannot step outside the maze. 

    Your goal is to find the nearest exit from the entrance. An exit is defined as an empty cell that is at the border of the maze. The entrance does not count as an exit.

    Return the number of steps in the shortest path from the entrance to the nearest exit, or -1 if no such path exists.
    """

    def nearestExit(self, maze, entrance):

        m, n = len(maze), len(maze[0])

        queue, seen = deque([(0, entrance[0], entrance[1])]), set((0, 0))

        while queue:

            dist, row, col = queue.popleft()

            if (row in [0, m - 1] or col in [0, n - 1]) and (row, col) != (entrance[0], entrance[1]): return dist 

            for nr, nc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:

                if (row + nr, col + nc) not in seen and 0 <= row + nr < m and 0 <= col + nc < n and maze[row + nr][col + nc] != "+":
                    seen.add((row + nr, col + nc))
                    queue.append((dist + 1, row + nr, col + nc))

        return -1 

    """
    MINIMUM KNIGHT MOVES

    In an infinite chess board with coordinates from -infinity to +infinity, you have a knight at square [0, 0].

    A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.

    Return the minimum number of steps needed to move the knight to the square [x, y]. It is guaranteed the answer exists.
    """

    def minKnightMoves(self, x, y):

        visited = set()
        queue = deque([(0, 0, 0)])

        # Standard BFS exploration
        while queue:
            
            # Retrieve a given tile along with the number of steps needed to reach the tile
            dist, curr_x, curr_y = queue.popleft()

            # If the tile is the target, return the number of steps needed
            if (curr_x, curr_y) == (x, y): return dist

            # Process all the possible directions
            for nx, ny in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
                
                # Calculate the new tile to be explored
                next_x, next_y = curr_x + nx, curr_y + ny

                # If the new tile has not been explored, mark it as visited and add it to the queue
                if (next_x, next_y) not in visited:

                    visited.add((next_x, next_y))
                    queue.append((dist + 1, next_x, next_y))

    """
    FIND ALL POSSIBLE RECIPES FROM GIVEN SUPPLIES

    Return a list of all the recipes that you can create. You may return the answer in any order.
    """

    def findAllRecipes(self, recipes, ingredients, supplies):

        res = []
        seen = set(supplies)
        dq = deque([(recipe, ingredient) for recipe, ingredient in zip(recipes, ingredients)])

        prev_size = len(seen) - 1

        while len(seen) > prev_size:

            prev_size = len(seen)
            
            for _ in range(len(dq)):

                rec, ingredient = dq.popleft()

                if all(i in seen for i in ingredient):
                    res.append(rec)
                    seen.add(rec)
                else: 
                    dq.append((rec, ingredient))

        return res
    
    """
    STRING WITHOUT AAA OR BBB

    Given two integers a and b, return any string s such that:

        - s has length a + b and contains exactly a 'a' letters and b 'b' letters,

        - The substring 'aaa' does not occur in s, and 

        - The substring 'bbb' does not occur in s.
    """

    def strWithout3a3b(self, A: int, B: int):

        ans = []

        # So long as there are characters to write
        while A or B:
            
            # Determine if we should write an 'a' character or 'b' character depending on the current result
            if len(ans) >= 2 and ans[-1] == ans[-2]: writeA = ans[-1] == 'b'
            else: writeA = A >= B

            # Add the determined character
            if writeA:
                A -= 1
                ans.append('a')
            else:
                B -= 1
                ans.append('b')

        # Return the result after adding the characters
        return "".join(ans)

    """
    REMOVE COLORED PIECES IF BOTH NEIGHBORS ARE THE SAME COLOR

    There are n pieces arranged in a line, and each piece is colored either by 'A' or by 'B'.

    You are given a string colors of length n where colors[i] is the color of the ith piece.

    Alice and Bob are playing a game where they take alternating turns removing pieces from the line.

    Assuming Alice and Bob play optimally, return True if Alice wins or False if Bob wins.
    """

    def winnerOfGame(self, colors):

        alice_count, bob_count, consecutive_colors = 0, 0, 0

        # Process all the colors in the string
        for current_color in colors:
            
            # If the current character is equal to the previous character
            if current_color == previous_color:
                consecutive_colors += 1
                if consecutive_colors == 3:
                    if current_color == 'A': alice_count += 1
                    elif current_color == 'B': bob_count += 1

            elif current_color != previous_color: consecutive_colors = 1 
    
            previous_color = current_color 
        
        # If Alice and Bob both have an equal number of pieces removed, Alice will lose by default as such Alice makes her move first
        return alice_count > bob_count
    
    """
    MONOTONIC ARRAY

    Given an integer array nums, return True if the given array is monotonic, or False otherwise.
    """

    def isMonotonic(self, nums):
        return all(nums[i] >= nums[i - 1] for i in range(1, len(nums))) or all(nums[i] <= nums[i - 1] for i in range(1, len(nums)))
    
    """
    CHECK IF WORD EQUALS SUMMATION OF TWO WORDS

    The letter value of a letter its position in the alphabet starting from 0.

    The numerical value of some string of lowercase English letters s is the concatenation of the letter values of each letter in s, which is then converted into an integer.

    You are given three strings firstWord, secondWord, and targetWord, each consisting of lowercase English letters 'a' through 'j' inclusive.

    Return True if the summation of the numerical values of firstWord and secondWord equals the numerical value of targetWord, or False otherwise.
    """
    
    def isSumEqual(self, firstWord, secondWord, targetWord):
        def convert_number(word):
            return int("".join([str(ord(word[i]) - ord('a')) for i in range(len(word))]))
        return convert_number(firstWord) + convert_number(secondWord) == convert_number(targetWord)
    
    """
    CHECK IF NUMBERS ARE ASCENDING IN A SENTENCE

    Given a string s representing a sentence, you need to check if all the numbers in s are strictly increasing from left to right. Return True if so, or False otherwise.
    """

    def areNumbersAscending(self, s: str):
        nums = [int(word) for word in s.split() if word.isdigit()]
        return all([nums[i - 1] < nums[i] for i in range(1, len(nums))])

    """
    SECOND LARGEST DIGIT IN A STRING

    Given an alphanumeric string s, return the second largest numerical digit that appears in s, or -1 if it does not exist.

    An alphanumeric string is a string consisting of lowercase English letters and digits.
    """

    def secondHighest(self, s: str):
        num_list = sorted(list(set(int(char) for char in s if char.isdigit())))
        return num_list[-2] if len(num_list) > 1 else -1

    """
    REMOVE ONE ELEMENT TO MAKE THE ARRAY STRICTLY INCREASING

    Given a 0-indexed integer array nums, return True if it can be made strictly increasing after removing exactly one element, or False otherwise. If the array is already strictly increasing, return True. 
    """

    def canBeIncreasing(self, nums: list):

        def checkIncreasing(A):
            return all(A[i] > A[i - 1] for i in range(1, len(A)))

        # If the initial list is already increasing, return True
        if checkIncreasing(nums): return True 

        # Otherwise consider removing the element at all possible indices to determine if the remaining list has only increasing elements
        for i in range(len(nums)):
            nums_copy = nums[:]
            nums_copy.pop(i)
            if checkIncreasing(nums_copy): return True 

        # If all removals did not yield an increasing list, return False
        return False

    """
    MINIMUM DISTANCE TO THE TARGET ELEMENT

    Given an integer array nums and two integers target and start, find an index i such that nums[i] == target and abs(i - start) is minimized. Return abs(i - start). 
    """

    def getMinDistance(self, nums, target, start):
        index_list = [i for i in range(len(nums)) if nums[i] == target]
        return min([abs(index - start) for index in index_list])

    """
    MAXIMUM DIFFERENCE BETWEEN INCREASING ELEMENTS

    Given a 0-indexed integer array nums of size n, find the maximum difference between nums[i] and nums[j] such that 0 <= i < j < n and nums[i] < nums[j]. Return the maximum difference. If no such i and j exists, return -1.
    """

    def maximumDifference(self, nums: list):
        min_val, max_diff = float('inf'), 0

        # Process all the elements in the list
        for num in nums: 
            if num < min_val: min_val = num
            else: max_diff = max(max_diff, num - min_val)

        # Return the maximum difference if there is one otherwise -1
        return max_diff if max_diff != 0 else -1

    """
    COUNTING WORDS WITH A GIVEN PREFIX

    You are given an array of strings words and a string pref.

    Return the number of strings in words that contain pref as a prefix.

    A prefix of a string s is any leading contiguous substring of s.
    """

    def prefixCount(self, words: list[str], pref: str):
        return sum([word.startswith(pref) for word in words])

    """
    LENGTH OF LAST WORD

    Given a string s consisting of words and spaces, return the length of the last word in the string.
    """
    def lengthOfLastWord(self, s: str):
        return 0 if not s or s.isspace() else len(s.split()[-1])

    """
    CHECK IF NUMBER HAS EQUAL DIGIT COUNT AND DIGIT VALUE

    You are given a 0-indexed string num of length n consisting of digits.

    Return True if for every index i in the range 0 <= i < n, the digit i occurs num[i] times in num, otherwise return False.
    """

    def digitCount(self, num: str):
        num_counts = Counter(num)
        return all(num_counts[str(i)] == int(num[i]) for i in range(len(num)))

    """
    MAXIMUM NUMBER OF WORDS FOUND IN SENTENCES

    A sentence is a list of words that are separated by a single space with no leading or trailing spaces.

    You are given an array of strings sentences, where each sentences[i] represents a single sentence.

    Return the maximum number of words that appear in a single sentence.
    """

    def mostWordsFound(self, sentences):
        return max(len(sentence.split()) for sentence in sentences)

    """
    SORT EVEN AND ODD INDICES INDEPENDENTLY

    You are given a 0-indexed integer array nums. Rearrange the values of nums according to the following rules: 

    Sort the values at odd indices of nums in non-increasing order. Sort the values at even indices of nums in non-decreasing order. 
    
    Return the array formed after rearranging the values of nums.
    """

    def sortEvenOdd(self, nums):
        odds = sorted([nums[i] for i in range(len(nums)) if i % 2 == 1])
        evens = sorted([nums[i] for i in range(len(nums)) if i % 2 == 0], reverse=True)
        return [evens.pop() if i % 2 == 0 else odds.pop() for i in range(len(nums))]

    """
    FIND ALL NUMBERS DISAPPEARED IN AN ARRAY

    Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers in the range [1, n] that do not appear in nums.
    """

    def findDisappearedNumbers(self, nums):
        counts = Counter(nums)
        return [num for num in range(1, len(nums) + 1) if counts[num] == 0]

    """
    SELF DIVIDING NUMBERS
    
    A self-dividing number is a number that is divisible by every digit it contains.

        - For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.

    A self-dividing number is not allowed to contain the digit zero. Given two integers left and right, return a list of all the self-dividing numbers in the range [left, right].
    """

    def selfDividingNumbers(self, left, right):
        
        def checkselfdividing(num):
            num_string = str(num)
            return all(num_string[i] != '0' and num % int(num_string[i]) == 0 for i in range(len(num_string)))

        # Use list comprehension to generate a list of the self-dividing numbers
        return [num for num in range(left, right + 1) if checkselfdividing(num)]
    