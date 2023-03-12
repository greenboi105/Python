from collections import * 
from itertools import * 
from heapq import *
from copy import * 
import heapq

class Solution:
    
    """
    FIND ORIGINAL ARRAY FROM DOUBLED ARRAY

    Given an array changed, return original if changed is a doubled array. 

    If changed is not a doubled array, return an empty array. The elements in original may be retuend in any order.
    """

    def findOriginalArray(self, changed):

        if len(changed) <= 1 or len(changed) % 2 == 1: return []

        changed.sort()
        occurrences = Counter(changed)
        res = []

        for key in sorted(list(occurrences)):

            if key not in occurrences: continue

            if key == 0: 
                number_removals = occurrences[key] // 2
                res += [key for _ in range(number_removals)]
                continue 

            double_key = 2 * key

            if double_key in occurrences: 

                number_removals = min(occurrences[key], occurrences[double_key])

                res += [key for _ in range(number_removals)]

                occurrences[double_key] -= number_removals
                occurrences[key] -= number_removals

                if occurrences[double_key] == 0: del occurrences[double_key]
                if occurrences[key] == 0: del occurrences[key]

        return res if len(res) == len(changed) // 2 else []

    """
    INCREMENTAL MEMORY LEAK

    You are given two integers memory1 and memory2 representing the available memory in bits on two memory sticks.

    There is currently a faulty program running that consumes an increasing amount of memory each second.
    """

    def memLeak(self, memory1, memory2):

        current_time = 1
        current_memory = 1

        while True:

            if current_memory > max(memory1, memory2): break

            if memory1 >= memory2: memory1 -= current_memory
            else: memory2 -= current_memory

            current_memory += 1
            current_time += 1

        return [current_time, memory1, memory2]

    """
    SORT FEATURES BY POPULARITY

    Return the features in sorted order.
    """

    def sortFeatures(self, features, responses):

        word_counts = defaultdict(int)

        for response in responses: 

            response_words = set(response.split())

            for feature in features: 
                if feature in response_words: word_counts[feature] += 1

        original_indices = {feature: feature_index for feature_index, feature in enumerate(features)}

        res = features[:]

        return sorted(res, key = lambda x: (-word_counts[x], original_indices[x]))

    """
    REDUCTION OPERATIONS TO MAKE THE ARRAY ELEMENTS EQUAL

    Return the number of operations to make all elements in nums equal.
    """

    def reductionOperations(self, nums):

        sorted_nums = sorted(nums)

        min_element = sorted_nums[0]
        current_required = 0
        current_num = min_element
        res = 0

        for num in sorted_nums[1:]:

            if num != current_num: 
                current_num = num 
                current_required += 1
            
            if num != min_element: 
                res += current_required

        return res 

    """
    SORT INTEGERS BY THE POWER VALUE

    Return the kth integer in the range [lo, hi] sorted by the power value.
    """

    def getKth(self, lo: int, hi: int, k: int) -> int: 

        def num_steps(number):
            res = 0

            while number != 1:
                if number % 2 == 0: 
                    number = number / 2
                    res += 1
                else: 
                    number = 3 * number + 1
                    res += 1

            return res 

        numbers = sorted([num for num in range(lo, hi + 1)], key = lambda x: (num_steps(x), x))

        return numbers[k - 1]

    """
    THOUSAND SEPARATOR

    Given an integer n, add a dot (".") as the thousands separator and return it in string format.
    """

    def thousandSeparator(self, n: int) -> str:

        if n < 1000: return str(n)

        counter = 3
        res = []

        number_string = str(n)[::-1]

        for char in number_string:

            counter -= 1

            if counter == 0: 
                res.append(char)
                res.append(".") 
                counter = 3
            else: res.append(char)

        if res[-1] == '.': res.pop()

        return "".join(res[::-1])

    """
    DECREASE ELEMENTS TO MAKE ARRAY ZIGZAG

    Given an array nums of integers, a move consists of choosing any element and decreasing it by 1.

    Return the minimum number of moves to transform the given array nums into a zigzag array.
    """

    def movesToMakeZigzag(self, nums):

        if len(nums) == 1: return 0
        
        even_removals = 0
        odd_removals = 0

        for i in range(0, len(nums), 2):
            
            if i == 0: 
                if nums[i] < nums[i + 1]: continue 
                else: even_removals += (nums[i] - nums[i + 1]) + 1
            elif i == len(nums) - 1:
                if nums[i] < nums[i - 1]: continue 
                else: even_removals += (nums[i] - nums[i - 1]) + 1
            else:
                if nums[i] < nums[i + 1] and nums[i] < nums[i - 1]: continue
                else: even_removals += nums[i] - min(nums[i + 1], nums[i - 1]) + 1

        for i in range(1, len(nums), 2):

            if i == 0: 
                if nums[i] < nums[i + 1]: continue 
                else: odd_removals += (nums[i] - nums[i + 1]) + 1
            elif i == len(nums) - 1:
                if nums[i] < nums[i - 1]: continue 
                else: odd_removals += (nums[i] - nums[i - 1]) + 1
            else:
                if nums[i] < nums[i + 1] and nums[i] < nums[i - 1]: continue
                else: odd_removals += nums[i] - min(nums[i + 1], nums[i - 1]) + 1

        return min(odd_removals, even_removals)

    """
    RANK TEAMS BY VOTES

    In a speical ranking system, each voter gives a rank from highest to lowest to all teams participating in the competition.

    Return a string of all teams sorted by the ranking system.
    """

    def rankTeams(self, votes):

        rankings = {}

        for vote in votes:

            for i, char in enumerate(vote):
                if char not in rankings: rankings[char] = [0] * len(vote)
                rankings[char][i] += 1

        voted_names = sorted(rankings.keys())

        return "".join(sorted(voted_names, key = lambda x: rankings[x], reverse=True))

    """
    MAXIMUM NUMBER OF NON-OVERLAPPING SUBARRAYS WITH SUM EQUALS TARGET

    Given an array nums and an integer target, return the maximum number of non-empty non-overlapping subarrays such that the sum of values in each subarray is equal to target.
    """

    def maxNonOverlapping(self, nums, target):

        seen = set([0])
        res = curr = 0

        for i, num in enumerate(nums):
            curr += num 
            prev = curr - target 

            if prev in seen:
                res += 1
                seen = set()

            seen.add(curr)

        return res 

    """
    MINIMUM INSERTIONS TO BALANCE A PARENTHESES STRING

    Return the minimum number of insertions needed to make s balanced.
    """

    def minInsertions(self, s):

        res = right = 0

        for char in s:

            if char == '(':
                if right % 2: 
                    right -= 1
                    res += 1
                right += 2

            if char == ')':
                right -= 1
                if right < 0:
                    right += 2
                    res += 1

        return right + res 

    """
    NUMBER OF SUBARRAYS OF SIZE K AND AVERAGE GREATER THAN OR EQUAL TO THRESHOLD

    Given an array of integers arr and two integers k and threshold, return the number of subarrays of size k and average greater than or equal to threshold.
    """

    def numOfSubarrays(self, arr, k, threshold):

        res = 0
        window_sum = 0

        for right in range(len(arr)):

            window_sum += arr[right]

            if right >= k:
                window_sum -= arr[right - k]

            if right >= k - 1:
                window_average = window_sum // k 

                if window_average >= threshold: res += 1

        return res 
    
    """
    MAXIMUM ELEMENT AFTER DECREASING AND REARRANGING

    Return the maximum possible value of an element in arr after performing the operations to satisfy the conditions.
    """

    def maximumElementAfterDecrementingAndRearranging(self, arr):

        sorted_nums = sorted(arr)

        if sorted_nums[0] != 1: sorted_nums[0] = 1

        for i in range(1, len(arr)):
            if sorted_nums[i] - sorted_nums[i - 1] > 1: sorted_nums[i] = sorted_nums[i - 1] + 1

        return sorted_nums[-1]

    """
    REMOVE ALL OCCURRENCES OF A SUBSTRING

    Return s after removing all occurrences of part.
    """

    def removeOccurrences(self, s, part):

        to_remove = ""
        right = 0

        while right < len(s):

            to_remove += s[right]

            if part in to_remove:
                to_remove = to_remove.replace(part, "")

            right += 1

        return to_remove

    """
    MINIMUM PATH COST IN A GRID

    Return the minimum cost of a path that starts from any cell in the first row and ends at any cell in the last row.
    """

    def minPathCost(self, grid, moveCost):

        m, n = len(grid), len(grid[0])

        grid_update = deepcopy(grid)

        for row in range(1, m):
            for col in range(n):
                grid_update[row][col] += min([grid_update[row - 1][k] + moveCost[grid[row - 1][k]][col] for k in range(n)])
        
        return min(grid_update[-1])

    """
    ARRAY OF DOUBLED PAIRS

    Given an integer array of even length arr, return True if it is possible to reorder arr such that arr[2 * i + 1] = 2 * arr[2 * i] for every 0 <= i < len(arr) / 2, or False otherwise.
    """

    def canReorderDoubled(self, arr):

        arr.sort(key = lambda x: abs(x))
        counts = Counter(arr)

        for i in range(len(arr)):

            if 2 * arr[i] in counts and arr[i] in counts:

                if arr[i] == 0: 
                    counts[arr[i]] -= 2
                    if counts[arr[i]] == 0: del counts[arr[i]]
                else:
                    counts[2 * arr[i]] -= 1
                    counts[arr[i]] -= 1

                    if counts[2 * arr[i]] == 0: del counts[2 * arr[i]]
                    if counts[arr[i]] == 0: del counts[arr[i]]

        return len(counts) == 0

    """
    MAP OF HIGHEST PEAK

    Return an integer matrix height of size m x n where height[i][j] is cell (i, j)'s height. 

    If there are multiple solutions, return any of them.
    """

    def highestPeak(self, isWater):

        m, n = len(isWater), len(isWater[0])
        seen = set((r, c) for r, c in product(range(m), range(n)) if isWater[r][c] == 1)
        queue = deque([(0, r, c) for r, c in product(range(m), range(n)) if isWater[r][c] == 1])
        res = [[None for _ in range(n)] for _ in range(m)]

        while queue:

            dist, r, c = queue.popleft()

            res[r][c] = dist

            for nr, nc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:

                if 0 <= r + nr < m and 0 <= c + nc < n and (r + nr, c + nc) not in seen:

                    queue.append((dist + 1, r + nr, c + nc))
                    seen.add((r + nr, c + nc))

        return res 

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
    