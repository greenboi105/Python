from collections import * 
from itertools import *

class Solution:

    """
    TOEPLITZ MATRIX 

    Given an m x n matrix, return True if the matrix is Toeplitz. Otherwise, return False.

    A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.
    """

    def isToeplitzMatrix(self, matrix):
        
        m, n = len(matrix), len(matrix[0])

        diags = defaultdict(list)

        for i in range(m):
            for j in range(n):
                diag = i - j 
                diags[diag].append(matrix[i][j])

        for key in diags:
            if len(set(diags[key])) > 1: return False 

        return True 

    """
    AVAILABLE CAPTURES FOR ROOK

    On an 8 x 8 chessboard, there is exactly one white rook 'R', and some number of white bishops 'B', black pawns 'p', and empty squares '.'.

    When the rook moves, it chooses one of four cardinal directions, then moves in that direction until it chooses to stop, reaches the edge of the board, captures a black pawn, or is blocked by a white bishop.

    Return the number of available captures for the white rook.
    """

    def numRookCaptures(self, board):

        res = 0

        for i in range(8):
            for j in range(8):
                if board[i][j] == 'R': 
                    rook_row = i
                    rook_col = j
        
        for i in range(rook_row + 1, 8): 
            if board[i][rook_col] == 'B': break 
            elif board[i][rook_col] == 'p': 
                res += 1
                break 

        for i in range(rook_row - 1, -1, -1):
            if board[i][rook_col] == 'B': break 
            elif board[i][rook_col] == 'p': 
                res += 1
                break 

        for j in range(rook_col + 1, 8):
            if board[rook_row][j] == 'B': break 
            elif board[rook_row][j] == 'p':
                res += 1
                break 

        for j in range(rook_col - 1, -1, -1):
            if board[rook_row][j] == 'B': break 
            elif board[rook_row][j] == 'p':
                res += 1
                break

        return res

    """
    CHECK IF ONE STRING SWAP CAN MAKE STRINGS EQUAL

    You are given two strings s1 and s2 of equal length.
    
    A string swap is an operation where you choose two indices in a string and swap the characters at these indices.

    Return Ture if it is possible to make both strings equal by performing at most one string swap on exactly one of the strings.

    Otherwise, return False.
    """

    def areAlmostEqual(self, s1, s2):

        counts1 = Counter(s1)
        counts2 = Counter(s2)

        if s1 == s2: return True 

        differences = 0
        for char1, char2 in zip(s1, s2):
            if char1 != char2: differences += 1

        return True if differences == 2 and counts1 == counts2 else False

    """
    SUM OF EVEN NUMBERS AFTER QUERIES 

    You are given an integer array nums and an array queries where queries[i] = [val_i, index_i].

    For each query i, first, apply nums[index_i] = nums[index_i] + val_i, then print the sum of the even values of nums.

    Return an integer array answer where answer[i] is the answer to the ith query.
    """

    def sumEvenAfterQueries(self, nums, queries):

        res, cur = [], sum(num for num in nums if num % 2 == 0)

        for value, index in queries:
            if nums[index] % 2 == 0: cur -= nums[index]
            nums[index] += value 
            if nums[index] % 2 == 0: cur += nums[index]
            res.append(cur)

        return res

    """
    REPLACE ELEMENTS IN AN ARRAY

    You are given a 0-indexed array nums that consists of n distinct positive integers. 

    Apply m operations to this array, where in the ith operation you replace the number operations[i][0] with operations[i][1].

    It is guaranteed that in the ith operation:

        - operations[i][0] exists in nums.

        - operations[i][1] does not exist in nums.

    Return the array obtained after applying all the operations.
    """
    
    def arrayChange(self, nums, operations):

        mapping = {num: i for i, num in enumerate(nums)}
        
        for num1, num2 in operations:

            index = mapping[num1]

            nums[index] = num2 

            mapping[num2] = index 

        return nums 

    """
    REVEAL CARDS IN INCREASING ORDER 

    You are given an integer array deck. There is a deck of cards where every card has a unique integer. 

    The integer on the ith card is deck[i]. You can order the deck in any order you want. Initially, all the cards start face down (unrevealed) in one deck.
    """

    def deckRevealedIncreasing(self, deck):

        sorted_deck = deque(sorted(deck))
        res = deque([])

        for i in range(len(deck)):
            
            res.appendleft(sorted_deck.pop())

            if i == len(deck) - 1: return list(res)

            for i in range(len(res) - 1, 0, -1):
                res[i], res[i - 1] = res[i - 1], res[i]

    """
    DISTRIBUTE CANDIES TO PEOPLE

    Return an array that represents the final distribution of candies.
    """

    def distributeCandies(self, candies, num_people):

        res = [0 for _ in range(num_people)]
        current = 1
        index = 0

        while True:

            if candies >= current: 
                res[index] += current 
                candies -= current
                if candies == 0: break 
            else:
                res[index] += candies 
                break 

            if index < len(res) - 1: index += 1
            else: index = 0

            current += 1

        return res 

    """
    AVERAGE WAITING TIME

    Return the average waiting time of all customers. Solutions within 10^-5 from the actual answer are considered accepted.
    """

    def averageWaitingTime(self, customers):

        res = 0
        current_time = customers[0][0]

        for arrival, time in customers:

            finish_time = max(current_time + time, arrival + time)

            current_time = finish_time

            waiting_time = current_time - arrival 

            res += waiting_time

        return res / len(customers)

    """
    TIME NEEDED TO BUY TICKETS

    Return the time taken for the person at position k to finish buying tickets.
    """
    
    def timeRequiredToBuy(self, tickets, k):

        res = 0

        for _ in range(tickets[k]):

            for i in range(len(tickets)):
                if tickets[i] != 0: 
                    tickets[i] -= 1
                    res += 1
                if i == k and tickets[i] == 0:
                    break

        return res

    """
    FINDING THE USERS ACTIVE MINUTES

    You are to calculate a 1-indexed array answer of size k such that, for each j (1 <= j <= k), answer[j] is the number of users whose UAM equals j.
    """

    def findingUsersActiveMinutes(self, logs, k):

        res = [0 for _ in range(k)]
        user_times = defaultdict(list)

        for id, time in logs:
            if time not in user_times[id]: user_times[id].append(time)

        for user in user_times:
            res[len(user_times[user]) - 1] += 1

        return res
    
    """
    MAXIMUM PRODUCT OF WORD LENGTHS

    Given a string array words, return the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters.

    If no such two words exist, return 0.
    """

    def maxProduct(self, words):

        word_mapping = defaultdict(set)
        res = 0

        for word in words:
            word_mapping[word].update(set(word))

            for other_word in word_mapping:
                if any(char in word_mapping[word] for char in word_mapping[other_word]): continue 
                else: res = max(res, len(word) * len(other_word))

        return res
        
    """
    MINIMUM INCREMENT TO MAKE ARRAY UNIQUE

    You are given an integer array nums. In one move, you can pick an index i where 0 <= i < nums.length and increments nums[i] by 1.

    Return the minimum number of moves needed to make every value in nums unique.
    """

    def minIncrementForUnique(self, nums):

        sorted_nums = sorted(nums)
        res = 0
        previous = sorted_nums[0]

        for i in range(1, len(sorted_nums)):

            if previous < sorted_nums[i]: 
                previous = sorted_nums[i]
            else: 
                update = previous + 1
                res += update - sorted_nums[i]
                sorted_nums[i] = update 
                previous = update

        return res 
        
    """
    MINIMIZE MAXIMUM PAIR SUM IN ARRAY

    The pair sum of a pair (a, b) is equal to a + b. The maximum pair sum is the largest pair sum in a list of pairs.

    Return the minimized maximum pair sum after optimally pairing up the elements.
    """

    def minPairSum(self, nums):

        new_nums = deque(sorted(nums))
        res = float('-inf')

        for _ in range(len(nums) // 2):
            res = max(res, new_nums.pop() + new_nums.popleft())

        return res 

    """
    MINIMIZE PRODUCT SUM OF TWO ARRAYS

    Given two arrays nums1 and nums2 of length n, return the minimum product sum if you are allowed to rearrange the order of the elements in nums1.
    """

    def minProductSum(self, nums1, nums2):

        new_nums1 = sorted(nums1)
        new_nums2 = sorted(nums2, reverse=True)
        res = 0

        for _ in range(len(nums1)):
            res += new_nums1.pop() * new_nums2.pop()

        return res 

    """
    OPTIMAL PARTITION OF STRING

    Given a string s, partition the string into one or more substrings such that the characters in each substring are unique.

    That is, no letter appears in a single substring more than once.
    """

    def partitionString(self, s):

        seen = set()
        res = 0

        for i in range(len(s)):
            if s[i] not in seen: seen.add(s[i])
            else: 
                seen.clear()
                seen.add(s[i])
                res += 1

            if i == len(s) - 1: res += 1
    
        return res

    """
    REMOVING MINIMUM AND MAXIMUM FROM ARRAY

    You are given a 0-indexed array of distinct integers nums.

    Return the minimum number of deletions it would take to remove both the minimum and maximum element from the array.
    """
    def minimumDeletions(self, nums):

        if len(nums) == 1: return 1

        min_val, max_val = float('inf'), float('-inf')
        min_idx, max_idx = None, None 
        N = len(nums)
        res = 0 

        for index, val in enumerate(nums):
            if val > max_val: 
                max_val = val 
                max_idx = index 
            if val < min_val:
                min_val = val
                min_idx = index 

        min_removal = min(abs(0 - min_idx), abs(N - 1 - min_idx)) + 1
        max_removal = min(abs(0 - max_idx), abs(N - 1 - max_idx)) + 1

        if min_removal <= max_removal:
            res += min_removal
            if max_idx > min_idx: max_idx -= min_removal
            N -= min_removal
            res += min(abs(0 - max_idx), abs(N - 1 - max_idx)) + 1
        else: 
            res += max_removal 
            if min_idx > max_idx: min_idx -= max_removal
            N -= max_removal
            res += min(abs(0 - min_idx), abs(N - 1 - min_idx)) + 1

        return res 

    """
    JAFAR'S PAWNS

    Jafar is playing checkers with his friend Aladdin.

    Jafar has just one pawn left, and he is going to take a final turn, beating as many of Aladdin's pawns as possible. Pawns in checkers move diagonally. The pawn always moves one step the up-right or up-left direction. 

    If Jafar's pawn moves and its target field is occupied by one of Aladdin's pawns, Aladdin's pawn can be beaten:

        - Jafar's pawn leaps over Aladdin's pawn, taking two steps in the chosen direction and removing Aladdin's pawn from the board.

    Jafar can beat Aladdin's pawn in this way only when the field beyond Aladdin's pawn is empty. 

    After beating Aladdin's pawn, Jafar can continue his turn and make another move, but only if he will again beat another one of Aladdin's pawns. 

    Of course, after this additional move, Jafar can continue his turn again by beating another of Aladdin's pawns, and so on for as long as there are further pawns to beat.

    When it is no longer possible to beat one of Aladdin's pawns, Jafar's turn ends. What is the maximum number of pawns owned by Aladdin that Jafar can beat in his turn?

    Write a function: def solution(B) which, given a square board of N x N size describing Aladdin's and Jafar's pawns, returns the maximum number of pawns Jafar can beat in one turn. If none of Aladdin's pawns can be beaten, the function should return 0.
    """

    def jafarsPawns(self, board):

        def find_jumps(row, column, jumps):
            
            # Determine if we can jump either left or right
            captureLeft = row > 1 and column > 1 and board[row - 1][column - 1] == 'X' and board[row - 2][column - 2] != 'X'
            captureRight = row > 1 and column < len(board) - 2 and board[row - 1][column + 1] == 'X' and board[row - 2][column + 2] != 'X'

            # Recursive base case, this path has reached a dead end - this can occur either after a series of jumps or in the immediate position
            if not captureRight and not captureLeft: return jumps 
            
            # Determine the number of captures for this recursive call
            num_captures = float("-inf")

            # Explore right 
            if captureRight:
                jump_val = find_jumps(row - 2, column + 2, jumps + 1)
                num_captures = max(num_captures, jump_val)
            
            # Explore left
            if captureLeft:
                jump_val = find_jumps(row - 2, column - 2, jumps + 1)
                num_captures = max(num_captures, jump_val)

            # Return the number of pawns that can be beat for the current recursive call
            return num_captures

        # Determine the dimensions of the board
        N = len(board)

        # Row and column index for the player's piece (to be used in the helper call)
        playerRow, playerColumn = 0, 0

        # Iterate through the board and determine where the 'O' character is 
        for r in range(N):
            for c in range(N):
                if board[r][c] == 'O':
                    playerRow = r
                    playerColumn = c

        # Return the maximum number of potential captures with the given board and starting position
        return find_jumps(playerRow, playerColumn, 0)

    """
    ARRAY NESTING

    You are given an integer array nums of length n where nums is a permutation of the numbers in the range [0, n - 1].

    You should build a set s[k] = {nums[k], nums[nums[k]], nums[nums[nums[k]]], ...} subjected to the following rule:

        - The first element in s[k] starts with the selection of the element nums[k] of index = k.

        - The next element in s[k] should be nums[nums[k]], and then nums[nums[nums[k]]], and so on.

        - We stop adding right before a duplicate element occurs in s[k].

    Return the longest length of a set s[k].
    """

    def arrayNesting(self, nums):

        # A list of seen elements and the length of a set with the given exploration rules
        seen, res = [0 for _ in range(len(nums))], 0

        # Process all the numbers in the list
        for num in nums:
            
            # The current length of such a set with the given exploration
            cnt = 0

            # So long as there is no duplicate continue to explore
            while not seen[num]:
                seen[num] = 1
                cnt += 1
                num = nums[num]
            
            # Determine the length of the current set and whether it exceeds the previous result
            res = max(res, cnt)

        # Return the length of such a set
        return res
    
    """
    MOST STONES REMOVED WITH SAME ROW OR COLUMN

    On a 2D plane, we place n stones at some integer coordinate points. Each coordinate point may have at most one stone.

    A stone can be removed if it shares either same row or the same column as another stone that has not been removed.

    Given an array stones of length n where stones[i] = [x_i, y_i] represents the location of the ith stone, return the largest possible number of stones that can be removed.
    """

    def removeStones(self, stones):

        def dfs(i, j):
            """
            Helper function to discard stones recursively from a given column or row.
            """

            points.discard((i, j))
            for y in rows[i]:
                if (i, y) in points: dfs(i, y)
            for x in cols[j]:
                if (x, j) in points: dfs(x, y)

        # A set of the unique points in the list of stones
        points = {(x, y) for x, y in stones}

        # The number of remaining stones
        count = 0

        # A mapping for the rows and a mapping for the columns
        rows = defaultdict(list)
        cols = defaultdict(list)

        # Process all the stones in the list to construct the graph
        for x, y in stones:
            rows[x].append(y)
            cols[y].append(x)

        # Process the stones and determine if the stone is present in the list of points
        for x, y in stones:
            if (x, y) in points:
                dfs(x, y)
                count += 1

        # Return the number of stones that can be removed
        return len(stones) - count

    """
    NUMBER OF ENCLAVES 

    You are given an m x n binary matrix grid, where 0 represents a sea cell and 1 represents a land cell.

    A move consists of walking from one land cell to another adjacent (4-directionally) land cell or walking off the boundary of the grid.

    Return the number of land cells in grid for which we cannot walk off the boundary of the grid in any number of moves.
    """

    def numEnclavesBFS(self, grid):

        def explore(r, c):

            if (r, c) in seen: return 
            queue = deque([(r, c)])
            seen.add((r, c))

            while queue: 
                
                r, c = queue.popleft()

                for nr, nc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if 0 <= r + nr < m and 0 <= c + nc < n and (r + nr, c + nc) not in seen and grid[r + nr][c + nc] == 1:
                        queue.append((r + nr, c + nc))
                        seen.add((r + nr, c + nc))

        res = 0
        m, n = len(grid), len(grid[0])
        seen = set()

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1 and (r in [0, m - 1] or c in [0, n - 1]): explore(r, c)
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1 and (r, c) not in seen: res += 1

        return res

    def numEnclavesDFS(self, grid):

        def explore(r, c):

            seen.add((r, c))

            for nr, nc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:

                if (r + nr, c + nc) not in seen and 0 <= r + nr < m and 0 <= c + nc < n and grid[r + nr][c + nc] == 1:
                    explore(r + nr, c + nc)

        res = 0
        m, n = len(grid), len(grid[0])
        seen = set()

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1 and (r in [0, m - 1] or c in [0, n - 1]): explore(r, c)

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1 and (r, c) not in seen: res += 1

        return res
    
    """
    SHORTEST DISTANCE TO A CHARACTER

    Given a string s and a character c that occurs in s, return an array of integers answer where answer.length == s.length and answer[i] is the distance from index i to the closest occurrence of character c in s.

    The distance between two indices i and j is abs(i - j), where abs i the absolute value function.
    """

    def shortestToChar(self, s: str, c: str):

        res = []
        occurrences = defaultdict(list)

        for index, char in enumerate(s):
            occurrences[char].append(index)

        for index, char in enumerate(s):
            min_distance = float('inf')
            for occurrence_index in occurrences[c]:
                min_distance = min(min_distance, abs(index - occurrence_index))
            res.append(min_distance)

        return res

    """
    DI STRING MATCH

    A permutation perm of n + 1 integers of all the integers in the range [0, n] can be represented as a string s of length n where:

        - s[i] == 'I' if perm[i] < perm[i + 1], and

        - s[i] == 'D' if perm[i] > perm[i + 1].

    Given a string s, reconstruct the permutation perm and return it. If there are multiple valid permutations perm, return any of them. 
    """

    def diStringMatch(self, s):

        numbers = deque([num for num in range(len(s) + 1)])
        res = []

        for inst in s:
            if inst == 'I': res.append(numbers.popleft())
            elif inst == 'D': res.append(numbers.pop())

        res.append(numbers.pop())
        return res 

    """
    GROUP THE PEOPLE GIVEN THE GROUP SIZE THEY BELONG TO

    There are n people that are split into some unknown number of groups. Each person is labeled with a unique ID from 0 to n - 1.

    You are given an integer array groupSizes, where groupSizes[i] is the size of the group that person i is in.

    Return a list of groups such that each person i is in a group of size groupSizes[i].
    """

    def groupThePeople(self, groupSizes):

        mapping, res = defaultdict(list), []

        for person, size in enumerate(groupSizes): mapping[size].append(person)

        for key in mapping:

            group_list, current_group = mapping[key], []

            if len(group_list) > key: 

                for num in group_list:
                    
                    current_group.append(num)

                    if len(current_group) < key: continue 
                    elif len(current_group) == key: 
                        res.append(current_group[:])
                        current_group = []

            else: res.append(mapping[key][:])

        return res

    """
    MAKE ARRAY ZERO BY SUBTRACTING EQUAL AMOUNTS

    You are given a non-negative integer array nums. In one operation, you must:

        - Choose a positive integer x such that x is less than or equal to the smallest non-zero element in nums.

        - Subtract x from every positive element in nums.

    Return the minimum number of operations to make every element in nums equal to 0.
    """

    def minimumOperations(self, nums):

        current_nums = nums[:]
        res = 0

        while sum(current_nums) != 0:

            res += 1

            smallest_element = min([val if val != 0 else float('inf') for val in current_nums])

            for i in range(len(current_nums)): current_nums[i] -= smallest_element if current_nums[i] > 0 else current_nums[i]

        return res 
    
    """
    BUILD ARRAY FROM PERMUTATION

    Given a zero-based permutation nums build an array ans of the same length and return it.
    """
    
    def buildArray(self, nums):
        return [nums[nums[i]] for i in range(len(nums))]

    """
    FIND COMMON CHARACTERS

    Given a string array words, return an array of all characters that show up in all string within the words.
    """

    def commonChars(self, words):

        res = []

        current_counts = Counter(words[0])

        for word in words[1:]:
            
            word_counts = Counter(word)

            for char in list(current_counts):
                if char not in word: del current_counts[char]
                else: current_counts[char] = min(current_counts[char], word_counts[char])

        for key, value in current_counts.items():
            for _ in range(value): res.append(key)

        return res

    """
    ISLAND PERIMETER

    You are given a row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water. 

    Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island. 

    The island has no "lakes", meaning that the water inside isn't connected to the water around the island. 

    One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island. 
    """

    def islandPerimeter(self, grid):
        
        m, n, island_perimeter = len(grid), len(grid[0]), 0

        # Iterate through all the cells of the grid to detect land tiles (tiles with a value of 1)
        for r in range(m):
            for c in range(n):
            
                # Determine if there is a land tile and the relative perimeter for this tile considering whether the adjacent tiles are land or water, if the tiles are land we cannot factor them into the perimeter
                if grid[r][c] == 1:
                    
                    if r == 0: up = 0
                    else: up = grid[r - 1][c]
                    if r == m - 1: down = 0
                    else: down = grid[r + 1][c]
                    if c == 0: left = 0
                    else: left = grid[r][c - 1]
                    if c == n - 1: right = 0
                    else: right = grid[r][c + 1]

                    # Increment the total perimeter based on the tile constraints (whether the adjacent tiles are land or water)
                    island_perimeter += 4 - (up + left + right + down)

        # Return the perimeter of the island after considering all tiles and the relative contributions to the island perimeter
        return island_perimeter

    """
    LUCKY NUMBERS IN A MATRIX

    Given an m x n matrix of distinct numbers, return all lucky numbers in the matrix in any order.

    A lucky number is an element of the matrix such that it is the minimum element in its row and maximum in its column.
    """

    def luckyNumbers(self, matrix):
      
        def checkLucky(i, j):
            return True if matrix[i][j] == min(matrix[i]) and matrix[i][j] == max(matrix[i][j] for i in range(len(matrix))) else False

        # Utilize the helper along with conditional list comprehension to generate a list of the lucky elements
        return [matrix[i][j] for i, j in product(range(len(matrix)), range(len(matrix[0]))) if checkLucky(i, j)]

    """
    SET MATRIX ZEROES

    Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's. Perform this operation in-place.
    """

    def setZeroes(self, matrix):

        def modify_zero(r, c):
            for i in range(len(matrix)): matrix[i][c] = 0
            for j in range(len(matrix[0])): matrix[r][j] = 0

        # Determine the dimensions of the grid
        m, n = len(matrix), len(matrix[0])

        # In order to perform the in-place operation, we first need to enqueue all the tiles in the grid that are zeroes
        cells = [(i, j) for i, j in product(range(m), range(n)) if matrix[i][j] == 0]
        
        # Iterate through the coordinates of all the cells and call the helper function on the cells with 0
        for i, j in cells: modify_zero(i, j)

    """
    CHECK IF EVERY ROW AND COLUMN CONTAINS ALL NUMBERS

    An n x n matrix is valid if every row and every column contains all the integers from 1 to n (inclusive).

    Given an n x n integer matrix matrix, return True if the matrix is valid. Otherwise, return False.
    """
    
    def checkValid(self, matrix):
        cols = [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]
        rows = [[val for val in row] for row in matrix]
        return all(set(col) == set(range(1, len(matrix) + 1)) for col in cols) and all(set(row) == set(range(1, len(matrix) + 1)) for row in rows)

    """
    FIND SMALLEST COMMON ELEMENT IN ALL ROWS

    Given an m x n matrix mat where every row is sorted in strictly increasing order, return the smallest common element in all rows. If there is no common element, return -1.
    """

    def smallestCommonElement(self, mat):
        m, n = len(mat), len(mat[0])
        counts = defaultdict(int)
        for row in range(m): 
            for col in range(n):
                counts[mat[row][col]] += 1
                if counts[mat[row][col]] == len(mat): return mat[row][col]
        return -1 

    """
    STRING MATCHING IN AN ARRAY

    Given an array of string words, return all strings in words that is a substring of another word.

    You can return the answer in any order. A substring is a contiguous sequence of characters within a string.
    """

    def stringMatching(self, words):

        res = []

        for word1 in words:
            words_copy = words[:]
            words_copy.remove(word1)
            for word2 in words_copy:
                if word1 in word2 and word1 not in res: res.append(word1)

        return res

    """
    SHUFFLE STRING

    You are given a string s and an integer array indices of the same length.

    The string s will be shuffled such that the character at the ith position moves to indices[i] in the shuffled string.

    Return the shuffled string.
    """

    def restoreString(self, s, indices):
        res = [None for _ in range(len(s))]
        for i in range(len(s)): res[indices[i]] = s[i]
        return "".join(res)
    
    