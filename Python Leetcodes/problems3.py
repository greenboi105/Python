from collections import * 

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
    FIND ALL LONELY NUMBERS IN THE ARRAY 

    You are given an integer array nums. A number x is lonely when it appears only once, and no adjacent numbers appear in the array.

    Return all lonely numbers in nums. You may return the answer in any order.
    """

    def findLonely(self, nums):
        counts = Counter(nums)
        return [num for num in nums if counts[num - 1] == 0 and counts[num + 1] == 0 and counts[num] < 2]

    """
    BRIGHTEST POSITION ON STREET

    A perfectly straight street is represented by a number line. 

    The street has street lamp(s) on it and is represented by a 2D integer array lights. 

    Each lights[i] = [position_i, range_i] indicates that there is a street lamp at position position_i that lights up the area from [position_i - range_i, position_i + range_i] (inclusive).

    Given lights, return the brightest position on the street. If there are multiple positions, return the smallest one.
    """

    def brightestPosition(self, lights):

        # A lookup mapping 
        mapping = defaultdict(int)

        # Determine the point where the brightness increases for a given light and the point where that lamp no longer illuminates the position (we want to determine overlapping portions, inclusive)
        for pos, dis in lights: 
            mapping[pos - dis] += 1
            mapping[pos + dis + 1] -= 1

        # The variables we want to update
        current_brightness, max_brightness, max_position = 0, -1, -1

        # Process the positions in sorted order
        for idx, val in sorted(mapping.items()):

            # Increment the current running brightness depending on whether a lamp region is beginning or ending
            current_brightness += val 

            # If the current brightness exceeds the former maximum brightness, set a new result
            if current_brightness > max_brightness:
                max_brightness, max_position = current_brightness, idx 
        
        # Return the position of the maximum brightness
        return max_position

    """
    FIND K-LENGTH SUBSTRINGS WITH NO REPEATED CHARACTERS

    Given a string s and an integer k, return the number of substrings in s of length k with no repeated characters.
    """

    def numKLenSubstrNoRepeats(self, s, k):

        N = len(s)
        if k > N: return 0

        res = 0 
        window_counts = defaultdict(int)

        for right in range(N):

            window_counts[s[right]] += 1

            if right > k - 1: 

                if window_counts[s[right - k]] > 1: window_counts[s[right - k]] -= 1
                else: del window_counts[s[right - k]]

            if len(window_counts.keys()) == k: res += 1

        return res 

    """
    WATERING PLANTS

    You want to water n plants in your garden with a watering can. The plants are arranged in a row and are labeled from 0 to n - 1 from left to right where the ith plant is located at x = i.

    Return the number of steps needed to water all the plants.
    """

    def wateringPlants(self, plants, capacity):

        current_capacity = capacity
        res = 0

        for i in range(len(plants)):

            if current_capacity >= plants[i]:
                res += 1 
                current_capacity -= plants[i]
                plants[i] = 0
            else: 
                go_back = i
                walk_foward = i + 1
                res += (go_back + walk_foward)
                current_capacity = capacity - plants[i]

        return res

    """
    QUEENS THAT CAN ATTACK THE KING

    On a 0-indexed 8 x 8 chessboard, there can be multiple black queens and one white king.

    Return the coordinates of the black queens that can directly attack the king. You may return the answer in any order.
    """

    def queensAttacktheKing(self, queens, king):

        def explore_board(row_direction, col_direction):

            king_row, king_col = king[0], king[1]

            for _ in range(1, 8): 
                king_row += row_direction
                king_col += col_direction

                if (king_row, king_col) in queen_positions: return [king_row, king_col]

            return []
        
        queen_positions = {(queen_row, queen_col) for queen_row, queen_col in queens}
        res = []

        for r, c in [(-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
            queen_pos = explore_board(r, c)
            if len(queen_pos) == 2: res.append(queen_pos)

        return res 

    """
    MAXIMUM SUM OF AN HOURGLASS

    You are given an m x n integer matrix grid.

    Return the maximum sum of the elements of an hourglass.
    """

    def maxSum(self, grid):

        def findHourglass(r, c):

            valid_hourglass = r + 2 <= m - 1 and c + 2 <= n - 1

            if valid_hourglass:

                hourglass_sum = 0

                for i in range(c, c + 3):
                    hourglass_sum += grid[r][i]

                hourglass_sum += grid[r + 1][c + 1]

                for j in range(c, c + 3):
                    hourglass_sum += grid[r + 2][j]

                return hourglass_sum
            
            else: return 

        m, n = len(grid), len(grid[0])
        res = 0

        for r in range(m):
            for c in range(n):
                hourglass = findHourglass(r, c)
                if hourglass: 
                    res = max(res, hourglass)
                    
        return res 
    
    """
    MAXIMUM MATCHING OF PLAYERS WITH TRAINERS 

    Return the maximum number of matching between players and trainers that satisfy the conditions.
    """

    def matchPlayersAndTrainers(self, players, trainers):

        res = 0

        sorted_players = sorted(players)
        sorted_trainers = sorted(trainers)

        while sorted_players and sorted_trainers:

            if sorted_players[-1] <= sorted_trainers[-1]:
                sorted_players.pop()
                sorted_trainers.pop()
                res += 1
            else: 
                sorted_players.pop()

        return res 

    """
    QUERIES ON NUMBER OF POINTS INSIDE A CIRCLE 

    You are given an array points where points[i] = [x_i, y_i] is the coordinates of the ith point on a 2D plane. 

    You are also given an array queries where queries[j] = [x_j, y_j, r_j] describes a circle centered at (x_j, y_j) with a radius of r_j.

    Return an array answer, where answer[j] is the answer to the jth query.
    """

    def countPoints(self, points, queries):

        res = []

        for x, y, r in queries:
            
            current_number = 0

            for point_x, point_y in points:
                current_distance = (abs(point_x - x) ** 2 + abs(point_y - y) ** 2) ** 0.5
                if current_distance <= r: current_number += 1
            
            res.append(current_number)

        return res 

    """
    NUMBER OF WAYS TO BUY PENS AND PENCILS

    You are given an integer total indicating the amount of money you have.

    You are also given two integers cost1 and cost2 indicating the price of a pen and pencil respectively. 

    Return the number of distinct ways you can buy some number of pens and pencils.
    """

    def waysToBuyPensPencils(self, total, cost1, cost2):

        if total < cost1 and total < cost2: return 1

        res = 0

        possible_pens = total // cost1 + 1 

        for num_pens in range(possible_pens):

            res += 1

            current_total = total - (cost1 * num_pens)

            num_pencils = current_total // cost2

            res += num_pencils
        
        return res
    
    """
    NUMBER OF WAYS TO SELECT BUILDINGS

    You are given a 0-indexed binary string s which represents the types of buildings along a street where:

        - s[i] = '0' denotes that the ith building is an office and 

        - s[i] = '1' denotes that the building is a restaurant.

    As a city official, you would like to select 3 buildings for random inspection. However, to ensure variety, no two consecutive buildings out of the selected buildings can be of the same type.

    Return the number of valid ways to select 3 buildings.
    """

    def numberOfWays(self, s: str) -> int:
        """
        Problem is effectively a counting problem using dynamic programming to track previous outcomes.
        """

        dp = defaultdict(int)

        for i in range(len(s)):

            if s[i] == "0":
                dp["0"] += 1
                dp["10"] += dp["1"]
                dp["010"] += dp["01"]

            if s[i] == "1":
                dp["1"] += 1
                dp["01"] += dp["0"]
                dp["101"] += dp["10"]

        return dp["010"] + dp["101"]
    
    """
    FRUIT INTO BASKETS

    You are visiting a farm that has a single row of fruit trees arranged from left to right. 

    The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.

    Given the integer array fruits, return the maximum number of fruits you can pick.
    """

    def totalFruit(self, fruits):

        res = 0
        left = 0
        window_occurrences = defaultdict(int)

        for right in range(len(fruits)):

            window_occurrences[fruits[right]] += 1

            while len(window_occurrences) > 2: 

                if window_occurrences[fruits[left]] > 1: window_occurrences[fruits[left]] -= 1
                else: del window_occurrences[fruits[left]]

                left += 1

            res = max(res, right - left + 1)

        return res 

    """
    BAG OF TOKENS

    You have an initial power of power, an initial score of 0, and a bag of tokens where tokens[i] is the value of the ith token.

    Return the largest possible score you can achieve after playing any number of tokens.
    """

    def bagOfTokensScore(self, tokens, power):

        removals = deque(sorted(tokens))
        score = 0

        while True: 

            if not removals or (power < removals[0] and score < 1) or (power < removals[0] and len(removals) == 1): break 

            if power >= removals[0]: 
                score += 1
                power -= removals.popleft()
            elif score >= 1:
                power += removals.pop()
                score -= 1

        return score

    """
    MAXIMUM NUMBER OF VOWELS IN A SUBSTRING OF GIVEN LENGTH

    Given a string s and an integer k, return the maximum number of vowel letters in any substring of s with length k.

    Vowel letters in English are 'a', 'e', 'i', 'o', and 'u'.
    """

    def maxVowels(self, s, k):

        window_occurrences = deque([])
        window_vowels = 0
        res = 0 

        for right in range(len(s)):

            window_occurrences.append(s[right])

            if right >= k: 

                removed_char = window_occurrences.popleft()

                if removed_char in ['a', 'e', 'i', 'o', 'u']: window_vowels -= 1

            if s[right] in ['a', 'e', 'i', 'o', 'u']: window_vowels += 1

            res = max(res, window_vowels)

        return res

    """
    PEOPLE WHOSE LIST OF FAVORITE COMPANIES IS NOT A SUBSET OF ANOTHER LIST

    Return the indices of people whose list of favorite companies is not a subset of any other list of favorites companies.

    You must return the indices in increasing order.
    """

    def peopleIndexes(self, favoriteCompanies):

        companies = defaultdict(list)
        res = []

        def determine_overlap(person):

            remaining_companies = favoriteCompanies[:]
            remaining_companies.pop(person)

            for company in remaining_companies:
                if set(companies[person]).issubset(set(company)): return False 

            return True 

        for i in range(len(favoriteCompanies)):
            companies[i] = favoriteCompanies[i]

        for person in sorted(companies.keys(), key = lambda x: len(companies[x]), reverse=True): 
            
            if determine_overlap(person): res.append(person)

        return sorted(res)

    """
    DIVIDE ARRAY IN SETS OF K CONSECUTIVE NUMBERS

    Given an array of integers nums and a positive integer k, check whether it is possible to divide this array into sets of k consecutive numbers.

    Return True if it is possible. Otherwise, return False.
    """

    def isPossibleDivide(self, nums, k):

        if len(nums) % k != 0: return False 

        counts = Counter(nums)

        for num in sorted(nums):
            if counts[num] > 0:
                for inc in range(k - 1, -1, -1):
                    counts[num + inc] -= counts[num]
                    if counts[num + inc] < 0: return False 

        return True 
    
    """
    DISPLAY TABLE OF FOOD ORDERS IN A RESTAURANT

    Return the restaurant's "display table". The "display table" is a table whose row entries denote how many of each food item each table ordered.
    """

    def displayTable(self, orders): 

        items = set()
        table = defaultdict(lambda: defaultdict(int))

        for order in orders: 
            items.add(order[2])
            table[order[1]][order[2]] += 1

        entry_order = sorted(list(items))

        res = []
        res.append(["Table"] + [entry for entry in entry_order])

        for table_number in sorted(table.keys(), key = lambda x: int(x)):
            table_row = []
            table_row.append(table_number)
            for entry in entry_order:
                num_item = str(table[table_number][entry])
                table_row.append(num_item)
            res.append(table_row)
        return res

    """
    REMOVE COVERED INTERVALS

    Given an array intervals where intervals[i] = [l_i, r_i] represent the interval [l_i, r_i), remove all intervals that are covered by another interval in the list.

    The interval [a, b) is covered by the interval [c, d) if and only if c <= a and b <= d.

    Return the number of remaining intervals.
    """

    def removeCoveredIntervals(self, intervals):

        intervals.sort(key = lambda x: (x[0], -x[1]))

        current_end = intervals[0][1]
        res = 1

        for _, interval_end in intervals: 

            if interval_end <= current_end: continue 
            else: 
                current_end = interval_end
                res += 1
    
        return res

    """
    FILTER RESTAURANTS BY VEGAN-FRIENDLY, PRICE AND DISTANCE

    Given the array restaurants where restaurants[i] = [id_i, rating_i, veganFriendly_i, price_i, distance_i].

    You have to filter the restaurants using three filters.
    """

    def filterRestaurants(self, restaurants, veganFriendly, maxPrice, maxDistance):

        res = []

        rating_mapping = {restaurant[0]: restaurant[1] for restaurant in restaurants}

        for restaurant in restaurants:

            if veganFriendly == 1 and restaurant[2] == 0: continue 

            if restaurant[3] > maxPrice: continue 

            if restaurant[4] > maxDistance: continue 

            res.append(restaurant[0])

        return sorted(res, key = lambda x: (-rating_mapping[x], -x))

    """
    NUMBER OF GOOD WAYS TO SPLIT A STRING   

    You are given a string s.

    Return the number of good splits you can make in s.
    """
    
    def numSplits(self, s):

        current_chars = set()
        overall_chars = defaultdict(set)

        overall_chars[0].add(s[0])
        for i in range(1, len(s)):
            overall_chars[i].add(s[i])
            overall_chars[i] |= overall_chars[i - 1]

        res = 0

        for i in range(len(s) - 1, 0, -1):

            current_chars.add(s[i])
            split_chars = overall_chars[i - 1]

            if len(current_chars) == len(split_chars): res +=1

        return res

    """
    NUMBER OF SUBSTRINGS WITH ONLY 1S

    Given a binary string s, return the number of substrings with all characters 1's. Since the answer may be too large, return it modulo 10 ** 9 + 7.
    """

    def numSub(self, s):

        running_ones = 0
        res = 0

        for i in range(len(s)):

            if s[i] == '1': running_ones += 1
            else: 
                res += (running_ones * (running_ones + 1)) // 2
                running_ones = 0

        res += (running_ones * (running_ones + 1)) // 2

        return res % (10 ** 9 + 7)

    """
    RESTORE THE ARRAY FROM ADJACENT PAIRS

    There is an integer array nums that consists of n unique elements, but you have forgetten it. 

    However, you do remember every pair of adjacent elements in nums. 

    Return the original array nums. If there are multiple solutions, return any of them.
    """

    def restoreArray(self, adjacentPairs):

        if len(adjacentPairs) == 1: return adjacentPairs[0]

        neighbors = defaultdict(list)
        seen = set()
        res = []

        for val1, val2 in adjacentPairs:
            neighbors[val1].append(val2)
            neighbors[val2].append(val1)

        head = sorted(neighbors.keys(), key = lambda x: len(neighbors[x]))[0]

        while True:

            if len(res) == len(adjacentPairs) + 1: return res 

            res.append(head)
            seen.add(head)

            for neighbor in neighbors[head]:
                if neighbor in seen: continue 
                else: head = neighbor 

    """
    CHECK IF NUMBER IS A SUM OF POWERS OF THREE

    Given an integer n, return True if it is possible to represent n as the sum of distinct powers of three.

    Otherwise, return False. An integer y is a power of three if there exists an integer x such that y == 3 ** x.
    """

    def checkPowersOfThree(self, n):

        def largestThree(num):
            res = 1
            power = 0
            while res <= num: 
                power += 1
                res *= 3 
            return power if res == num else power - 1

        current_num = n 
        seen = set()

        while True:
            three_removal = 3 ** largestThree(current_num)
            if three_removal in seen: return False
            seen.add(three_removal)
            current_num -= three_removal
            if current_num == 0: return True
