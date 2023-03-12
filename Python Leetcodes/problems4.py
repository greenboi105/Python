from collections import *
import heapq 
import math 

class Solution:

    """
    CAN MAKE ARITHMETIC PROGRESSION FROM SEQUENCE 

    A sequence of numbers is called an arithmetic progression if the difference between any two consecutive elements is the same.

    Given an array of numbers arr, return True if the array can be rearranged to form an arithmetic progression. Otherwise, return False.
    """

    def canMakeArithmeticProgression(self, arr):
        arr.sort()
        return len(set([arr[i] - arr[i - 1] for i in range(1, len(arr))])) == 1

    """
    FIND THE KTH LARGEST INTEGER IN THE ARRAY

    You are given an array of strings nums and an integer k. 

    Each string in nums represents an integer without leading zeros.

    Return the string that represents the kth largest integer in nums.
    """

    def kthLargestNumber(self, nums, k):
        return sorted(nums, key=lambda x: int(x), reverse=True)[k - 1]
    
    """
    REMOVING MINIMUM NUMBER OF MAGIC BEANS

    You are given an array of positive integers beans, where each integer represents the number of magic beans in a particular magic bag.

    Remove any number of beans from each bag such that the number of beans in each remaining non-empty bag is equal. Once a bean has been removed from a bag, you are not allowed to return it to any of the bags.

    Return the minimum number of magic beans that you have to remove.
    """

    def minimumRemoval(self, beans):
        optimal_remaining, n = 0, len(beans)
        for num_remaining, value in enumerate(sorted(beans)): optimal_remaining = max((n - num_remaining) * value, optimal_remaining)
        return sum(beans) - optimal_remaining

    """
    MERGE INTERVALS 

    Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
    """

    def merge(self, intervals):
        res = []
        for start, end in sorted(intervals, key = lambda x: x[0]):
            if not res or res[-1][1] < start: res.append([start, end])
            else: res[-1][1] = max(res[-1][1], end)
        return res

    """
    INTERVAL LIST INTERSECTIONS

    You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [starti, endi] and secondList[j] = [startj, endj].

    Each list of intervals is pairwise disjoint and and in sorted order. Return the intersection of these two interval lists. A closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.

    The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval.
    """
    
    def intervalIntersection(self, firstList, secondList):
        overlaps, i, j = [], 0, 0

        # For each iteration we calculate a possible overlapping portion given the start and end of each interval, if there is a valid intersecting portion we append that portion
        while i < len(firstList) and j < len(secondList):
            
            # Determine the possible overlapping portion for the current two intervals
            left, right = max(firstList[i][0], secondList[j][0]), min(firstList[i][1], secondList[j][1])
            
            # If there is a possible intersection, append this intersection to the list of overlaps as another sublist
            if left <= right: overlaps.append([left, right])

            # Determine which pointer to move up consider the ends of the two current intervals
            if firstList[i][1] <= secondList[j][1]: i += 1
            elif firstList[i][1] > secondList[j][1]: j += 1

        # Return the container of overlapping intervals after all the merges
        return overlaps
    
    """
    CAR POOLING

    There is a car with capacity empty seats. The vehicle only drives east.

    You are given the integer capacity and an array trips where trips[i] = [numPassengers, from, to] indicates that the ith trip has numPassengers passengers and the locations to pick them up and drop them off are from_i and to_i respectively.

    The locations are given as the number of kilometers due east from the car's initial location.

    Return True if it is possible to pick up and drop off all passengers for all the given trips, or False otherwise.
    """

    def carPooling(self, trips, capacity):
        """
        Process the trips to determine at which points in time passengers are onboarded and offboarded. 
        We then sort the timestamps in increasing order to determine if there is ever a point in time where the number of passengers in the car exceeds the capacity of the car.
        Two pass solution - a first iteration through the trips themselves to add tuple pairs and a second iteration through the list containing the tuple pairs using tuple unpacking to determine if there is ever a point that the number of passengers exceeds the capacity.
        We want to utilize a tuple pair of the form (time, passenger change).
        """

        # List to represent the changing of passengers utilizing the times for when passengers are onboarding and offboarding (entries are stored in the form of a tuple) and a variable to represent the running number of people in the car
        times, num_people = [], 0

        # Iterate through the trips which are lists with three elements (passengers, from, to)
        for trip in trips:
            times.append((trip[1], trip[0]))
            times.append((trip[2], -trip[0]))

        # Sort the timestamps in increasing order to determine the relative passenger capacity at a given time - this is important to determine the relative number of people on the bus at any given point in time
        times.sort(key = lambda x: x[0])

        # iterating through the times in the timestamps container after being sorted in increasing order using tuple unpacking to determine the offboarding number of people 
        for location, passenger_change in times: 
            num_people += passenger_change
            if num_people > capacity: return False 

        # Otherwise if we have iterated through all the timestamps and the capacity was never exceeded, return True
        return True

    """
    MOST PROFIT ASSIGNING WORK

    You are given three arrays: difficulty, profit, and worker where:

        - difficulty[i] and profit[i] are the difficulty and the profit of the ith job, and

        - worker[j] is the ability of jth worker (i.e., the jth worker can only complete a job with difficulty at most worker[j]).

    Every worker can be assigned at most one job, but one job can be completed multiple time.

    Return the maximum profit we can achieve after assigning the workers to the jobs.
    """

    def maxProfitAssignment(self, difficulty, profit, worker):
        """
        Optimization problem requires some more clever construction of lists to process the jobs with corresponding difficulties and profit.
        We want to order the jobs by the difficulty to consider the maximum profits we can make with the workers and their given skill.
        Problem essentially asks us to optimize the amount of profit to be made given associated difficulties and profits along with worker potential.
        Problem requires many structures in order to properly determine the optimal amount of profit we can make given the three iterables.
        We must acknowledge the key aspects that harder jobs do not necessarily give us more profit and that multiple workers can take the same job.
        """

        # Sort the jobs according to increasing difficulty and the workers according to increasing skill
        jobs, workers = sorted(list(zip(difficulty, profit)), key=lambda x: x[0]), sorted(worker)

        # Declare a variable for the optimal profit to be made, the current job index we are considering and the optimal profit for the current worker
        optimal_profit, job_difficulty, worker_best = 0, 0, 0
        
        # Consider each worker and their associated skill in increasing order
        for worker_skill in workers:
            
            # So long as the index for the current job is valid and the skill of the current worker is greater than the difficulty of the current job
            while job_difficulty < len(jobs) and worker_skill >= jobs[job_difficulty][0]: 

                # Determine the optimal profit this worker can make given the possible jobs they can take
                worker_best = max(worker_best, jobs[job_difficulty][1])

                # Process the next job of greater difficulty
                job_difficulty += 1

            # Once we have determined what the best profit the worker can make is given their skill level, increment the amount of profit by this best profit
            optimal_profit += worker_best 

        # Return the max profit we can make given the jobs, associated worker skills and the fact that multiple workers can take the same job if the profit is optimal
        return optimal_profit
    
    """
    MAXIMUM AREA OF A PIECE OF CAKE AFTER HORIZONTAL AND VERTICAL CUTS

    You are given a rectangular cake of size h x w and two arrays of integers horizontalCuts and verticalCuts where:
    
        - horizontalCuts[i] is the distance from the top of the rectangular cake to the ith horizontal cut and similarly, and 
        
        - verticalCuts[j] is the distance from the left of the rectangular cake to the jth vertical cut.

    Return the max area of a piece of cake after you cut at each horizontal and vertical position provided in the arrays horizontalCuts and verticalCuts. 

    Since the answer can be a large number, return this modulo 10 ** 9 + 7.
    """

    def maxArea(self, h, w, horizontalCuts, verticalCuts):
        """
        Determine the maximum area of a piece of cake after the cuts. Problem essentially becomes a math problem that requires ordering to determine the dimensions of the largest piece of cake.
        Sort both the vertical and horizontal cuts then process them in sorted order to determine the largest gap between adjacent cuts.
        Using the maximum possible adjacent cuts we can calculate the largest area of a piece of cake after the cuts.
        """

        # Sort the list of horizontal and vertical cuts to process adjacent cuts
        horizontalCuts.sort()
        verticalCuts.sort()

        possible_horizontal_distances = [horizontalCuts[0], h - horizontalCuts[-1]] + [horizontalCuts[i] - horizontalCuts[i - 1] for i in range(1, len(horizontalCuts))]
        max_height = max(possible_horizontal_distances)

        possible_vertical_distances = [verticalCuts[0], w - verticalCuts[-1]] + [verticalCuts[i] - verticalCuts[i - 1] for i in range(1, len(verticalCuts))]
        max_width = max(possible_vertical_distances)

        # Return the area after determining the largest possible height and width for a piece of cake given the cuts
        return (max_height * max_width) % (10 ** 9 + 7)

    """
    MAXIMUM BAGS WITH FULL CAPACITY OF ROCKS

    Return the maximum number of bags that could have full capacity after placing the additional rocks in some bags.

    The number of additional rocks is guaranteed to be at least 1.
    """

    def maximumBags(self, capacity, rocks, additionalRocks):
        """
        Ordering with monotonic stack for removal.
        """
        
        # Highly Pythonic method for generating a list of the rocks required to fill a given bag using the parameters for the bag capacity and the number of rocks present in the bag
        not_filled = sorted([amount - rock for amount, rock in zip(capacity, rocks)], reverse=True)

        # Iterate so long as there are bags to be filled and we have additional rocks to fill the bags and the number of rocks needed to fill a given bag does not exceed the number of rocks we have remaining
        while not_filled and additionalRocks and not_filled[-1] <= additionalRocks: additionalRocks -= not_filled.pop()

        # After the modifications determine the number of bags we have filled
        num_filled = len(rocks) - len(not_filled)

        # Return the number of bags that can have full capacity after placing the additional rocks in some bags
        return num_filled

    """
    DETERMINE IF TWO EVENTS HAVE CONFLICT 

    You are given two arrays of strings that represent two inclusive events that happened on the same day, event1 and event2, where:

        - event1 = [startTime_1, endTime_1] and 

        - event2 = [startTime_2, endTime_2].

    Event times are valid 24 hours format in the form of HH:MM. Return True if there is a conflict between two events. Otherwise, return False.
    """

    def haveConflict(self, event1, event2):

        events = sorted([event1, event2])

        def determineOverlap(end, start):

            if end == start: return True

            for end_char, start_char in zip(end, start):

                if not end_char.isdigit() and not start_char.isdigit(): continue
                elif int(end_char) < int(start_char): return False 
                elif int(end_char) == int(start_char): continue
                else: return True 

        return determineOverlap(events[0][1], events[1][0])

    """
    MINIMUM NUMBER OF ARROWS TO BURST BALLOONS

    There are some spherical balloons taped onto a flat wall that represents the XY-plane. 

    The balloons are represented as a 2D integer array points where points[i] = [x_start, x_end] denotes a balloon whose horizontal diameter stretches between x_start and x_end.

    Arrows can be shot up directly vertically from different points along the x-axis. A balloon with x_start and x_end is burst by an arrow shot at x if x_start <= x <= x_end.

    There is no limit to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

    Given the array points, return the minimum number of arrows that must be shot to burst all balloons.
    """

    def findMinArrowShots(self, points):
        """
        Greedy algorithm that tracks the ending point to determine the number of arrows that must be shot.
        Greedy rule is similar to the overlapping intervals problem where we need to sort according to the ends of the intervals.
        """

        # If there are no points and thus no balloons, return 0 for the number of arrows that must be shot
        if not points: return 0 

        # Sort according to the ending point
        points.sort(key = lambda x: x[1])

        # We can determine at this point that at least a single arrow must be fired to pop all the balloons, the initial ending point is set to the end of the first balloon
        arrows = 1
        first_end = points[0][1]

        # Process all the starting and ending point in the interval
        for x_start, x_end in points:
            
            # If the start of a future balloon is greater than our current end, we need an additional arrow and move up the end
            if first_end < x_start:
                arrows += 1
                first_end = x_end 

        # Return the number of arrows that must be fired
        return arrows 

    """
    NON-OVERLAPPING INTERVALS

    Given an array of intervals where intervals[i] = [start_i, end_i], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
    """

    def eraseOverlapIntervals(intervals):
        """
        This is effectively a type of greedy problem that requires us to order the intervals by their ending value to determine if a given interval must be removed to make the remaining intervals non-overlapping.
        The greedy rule in this case is to sort the intervals by their ending time and increment the number of intervals to be removed whenever there is an overlapping pair.
        """

        # Determine the current value for an ending interval for comparison and a count of the number of intervals that need to be removed to make the remainder of the intervals non-overlapping.
        running_end, number_removed, sorted_intervals = float('-inf'), 0, sorted(intervals, key=lambda x: x[1])

        # Process all the intervals in the list of sorted intervals using tuple unpacking to determine the beginning and end of a given interval
        for begin, ending in sorted_intervals:

            # There are two possibilities for the current interval - either the beginning of the current interval has no overlapping portion with the previous, meaning we modify the value of the running end or we need to increment the count of overlapping intervals
            if begin >= running_end: running_end = ending 
            else: number_removed += 1

        # Return the number of non-overlapping intervals
        return number_removed

    """
    HOW MANY NUMBERS ARE SMALLER THAN THE CURRENT NUMBER

    Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it.

    That is, for each nums[i] you have to count the number of valid j's such that j != i and nums[j] < nums[i].
    """

    def smallerNumbersThanCurrent(self, nums):

        # Mapping to store the number of values smaller than the current value
        mapping = {}

        # Iterate through the index-value pairs in the sorted container, if the value is not already present in the mapping, we know that the number of elements smaller than the current value is equal to the index
        for index, value in enumerate(sorted(nums)): 
            if value not in mapping: mapping[value] = index 
        
        # Return a list using list comprehension - the indices indicate how many numbers are smaller than the current number
        return [mapping[num] for num in nums]

    """
    THE NUMBER OF WEAK CHARACTERS IN THE GAME

    You are playing a game that contains multiple characters, and each of the characters has two main properties: attack and defense.

    You are given a 2D integer array properties where properties[i] = [attack_i, defense_i] represents the properties of the ith character in the game.

    A character is said to be weak if any other character has both attack and defense levels strictly greater than this character's attack and defense levels.

    More formally, a character i is said to be weak if there exists another character j where attack_j > attack_i and defense_j > defense_i.

    Return the number of weak characters.
    """

    def numberOfWeakCharacters(self, properties):
        """
        Sorting according to increasing order of attack and decreasing order of defense.
        In essence, we need to sort and utilize a stack to determine the characters that have both quantities less than other characters.
        """

        # A stack and a variable to store the number of weak characters in the list 
        stack, ans = [], 0

        # Process all the characters in the properties list with their attack and defense
        for attack, defense in sorted(properties, key = lambda x: (x[0], -x[1])):
            
            # If the character at the end of the stack has less defense than the current character, we know that both its attack and defense are less since the characters were arranged in increasing order of attack and decreasing order of defense
            while stack and stack[-1] < defense:
                stack.pop()
                ans += 1

            # Add the defense value of the current character
            stack.append(defense)

        # Return the number of weak characters
        return ans
    
    """
    ASSIGN COOKIES

    Assume you are an awesome parent and want to give your children some cookies. But, you should given each child at most one cookie.

    Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content.

    You goal is to maximize the number of your content children and output the maximum number.
    """

    def findContentChildren(self, g, s):
        """
        We have two lists as parameters, a list of relative child greeds and a list of cookie numbers.
        We ideally want to match the largest cookie amount to the largest greed at each given iteration, as such we order the containers to match and determine the number of content children given the two parameters.
        Continue to attempt to match cookies with greed so long as both lists have at least a single element.
        There are two possibilities for each iteration, either the cookie is sufficient to make the child content and we remove from both the greed and cookie lists, or it is impossible to make the child content with the remaining amounts.
        """

        # Problem variables - we need a variable for the number of content children and two sorted containers for the relative greed and cookies
        greed, cookies, num_content = sorted(g), sorted(s), 0

        # For each iteration there are two possibilities, we can either make a child content and remove from both the cookies and greed or establish that the child will not be content
        while greed and cookies:

            # Determine if the greatest number of cookes at the end of the stack is greater than or equal to the greatest greed at the end of the list of greeds
            if cookies[-1] >= greed[-1]: 
                cookies.pop()
                greed.pop()
                num_content += 1
            else: greed.pop()

        # Return the number of content children after the loop ends
        return num_content
    
    """
    CAR FLEET

    There are n cars going to the same destination along a one-lane road. The destination is target miles away.

    You are given two integer arrays position and speed, both of length n, where position[i] is the position of the ith car and speed[i] is the speed of the ith car (in miles per hour).

    A car can never pass another car ahead of it, but it can catch up to it and drive bumper to bumper at the same speed.

    A car fleet is some non-empty set of cars driving at the same position and same speed. Note that a single car is also a car fleet.

    If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.

    Return the number of car fleets that will arrive at the destination.
    """

    def carFleet(self, target, position, speed):
        """
        Utilize a stack for comparing cars to an existing car fleet.
        """

        # Bundle the cars with their associated position and speed, arrange the cars according to decreasing position and increasing speed
        cars = sorted(list(zip(position, speed)), key = lambda x: (-x[0], x[1]))

        # Utilize a stack to compare the current time needed to form a car fleet with the current car
        stack = []

        for position, speed in cars:
            
            distance = target - position
            if not stack or distance / speed > stack[-1]: stack.append(distance/speed)
            else: continue

        # The number of elements in the stack represents the number of car fleets
        return len(stack)

    """
    SORT INTEGERS BY THE NUMBER OF 1 BITS

    You are given an integer array arr. Sort the integers in the array in ascending order by the number of 1's in their binary representation and in case of two or more integers having the same number of 1's you have to sort them in ascending order.

    Return the array after sorting it.
    """

    def sortByBits(self, arr):

        # A helper to determine the number of 1 bits in the binary representation of a number
        def binarySum(num):
            return sum(int(char) if char.isdigit() else 0 for char in bin(num))

        # Sort according to the number of 1 bits and digit value
        arr.sort(key = lambda x: (binarySum(x), x))

        # Return the sorted list
        return arr 
    
    """
    MINIMUM OPERATIONS TO MAKE A UNI-VALUE GRID

    You are given a 2D integer grid of size m x n and an integer x.

    In one operation, you can add x to or subtract x from any element in the grid. A uni-value grid is a grid where all the elements of it are equal. 
    
    Return the minimum number of operations to make the grid uni-value. If it is not possible, return -1.
    """

    def minOperations(self, grid, x):

        # Generate a single list from the values in the grid
        list_vals = [val for row in grid for val in row]

        # Determine if all the values in the grid can be reduced to the same value given the possible operatins, meaning that the value must be divisible by the modification value x
        if len(set([val % x for val in list_vals])) > 1: return -1 

        # Determine a median value (NOTE: value, not index) from the sorted single matrix list using indexing
        median_val = sorted(list_vals)[len(list_vals) // 2]

        # Determine the minimum number of operations based on the absolute difference divided by the given number of modifications for all values in the single list, this is done with list comprehension
        return sum([abs(val - median_val) // x for val in list_vals])
    
    """
    MAXIMUM ICE CREAM BARS 

    At the store, there are n ice cream bars. You are given an array costs of length n, where costs[i] is the price of the ith ice cream bar in coins.

    The boy initially has coins coins to spend, and he wants to buy as many ice cream bars as possible. 

    Return the maximum number of ice cream bars the boy can buy with coins coins.
    """
    
    def maxIceCream(self, costs, coins):

        costs.sort()
        res = 0 

        for cost in costs:
            if cost <= coins: 
                res += 1
                coins -= cost 
            else: break 

        return res 
    
    """
    LARGEST PERIMETER TRIANGLE 

    Given an integer array nums, return the largest perimeter of a triangle with a non-zero area, formed from three of these lengths.

    If it is impossible to form any triangle of a non-zero area, return 0.
    """

    def largestPerimeter(self, nums):

        nums.sort()
        res = 0

        for i in range(2, len(nums)):
            if nums[i - 2] + nums[i - 1] > nums[i]: res = max(res, nums[i] + nums[i - 1] + nums[i - 2])

        return res 

    """
    LEAST NUMBER OF UNIQUE INTEGERS AFTER K REMOVALS 

    Given an array of integers arr and an integer k. Find the least number of unique integers after removing exactly k elements.
    """

    def findLeastNumOfUniqueInts(self, arr, k):

        counts = Counter(arr)

        arr.sort(key = lambda x: (counts[x], x), reverse=True)

        res = len(set(arr))

        for key in sorted(counts.keys(), key = lambda x: counts[x]):

            if counts[key] <= k: 
                res -= 1
                k -= counts[key]
                del counts[key]
            else: break 

        return res

    """
    WIDEST VERTICAL AREA BETWEEN TWO POINTS CONTAINING NO POINTS

    Given n points on a 2D plane where points[i] = [x_i, y_i], return the widest vertical area between two points such that no points are inside the area.
    """
    
    def maxWidthOfVerticalArea(self, points):

        points.sort()
        return max(points[i][0] - points[i - 1][0] for i in range(len(points)))

    """
    FIND ALL GROUPS OF FARMLAND

    You are given a 0-indexed m x n binary matrix land where a 0 represents a hectare of forested land and 1 represents a hectare of farmland.

    Return a 2D array containing the 4-length arrays described above for each group of farmland in land. If there are not groups of farmland, return an empty array.

    You may return the answer in any order.
    """

    def findFarmland(self, land):

        def farmlandSearch(row, col):
            
            if (row, col) in seen: return 

            queue = deque([(row, col)])
            current_coords = [row, col, None, None]

            while queue: 

                r, c = queue.popleft()

                bottom_end = r == m - 1 or land[r + 1][c] == 0
                right_end = c == n - 1 or land[r][c + 1] == 0

                if right_end and bottom_end: 
                    current_coords[2] = r 
                    current_coords[3] = c 
                    res.append(current_coords)
                    return 

                for nr, nc in [(1, 0), (0, 1)]:
                    if r + nr <= m - 1 and c + nc <= n - 1 and (r + nr, c + nc) not in seen and land[r + nr][c + nc] == 1: 
                        queue.append((r + nr, c + nc))
                        seen.add((r + nr, c + nc))

        m, n = len(land), len(land[0])
        seen = set()
        res = []

        for r in range(m):
            for c in range(n):
                if land[r][c] == 1: farmlandSearch(r, c)

        return res 
    
    """
    REMOVE STONES TO MINIMIZE THE TOTAL

    You are given a 0-indexed integer array piles, where piles[i] represents the number of stones in the ith pile, and an integer k.

    You should apply the following operation exactly k times:

        - Choose any piles[i] and remove floor(piles[i] / 2) stones from it.

    Notice that you can apply the operation on the same pile more than once.

    Return the minimum possible total number of stones remaining after applying the k operations.
    """

    def minStoneSum(self, piles: list, k):
        
        # Generate a max heap of the possible piles
        heap = [-pile for pile in piles]
        heapq.heapify(heap)

        # Determine the total number of stones from which we remove at each given iteration
        total_stones = sum(piles)
        
        # While there are remaining operations and remaining stones
        while k != 0 and heap:

            # Retrieve the largest element currently in the heap
            largest_element = -heapq.heappop(heap)

            # Determine the number of stones we can remove with the given operation
            removal_number = math.floor(largest_element / 2)
            
            # Decrement this number from the number of total stones
            total_stones -= removal_number

            # Modify the previous largest element
            largest_element -= removal_number

            # If this element can be further modified, push it back onto the heap
            if largest_element > 1: heapq.heappush(heap, -largest_element)

            # Decrement the number of possible operations
            k -= 1

        # Return the number of remaining stones after the operations
        return total_stones

    """
    KTH LARGEST ELEMENT IN AN ARRAY

    Given an integer array nums and an integer k, return the kth largest element in the array.

    Note that it is the kth largest element in the sorted order, not the kth distinct element.

    You must solve it in O(n) time complexity.
    """

    def findKthLargest(self, nums, k):
        return heapq.nlargest(k, nums)[-1]

    """
    MAX SUM OF A PAIR WITH EQUAL SUM OF DIGITS

    You are given a 0-indexed array nums consisting of positive integers. You can choose two indices i and j, such that i != j, and the sum of the digits of the nums[i] is equal to that of nums[j].

    Return the maximum value of nums[i] + nums[j] that you can obtain over all possible indices i and j that satisfy the conditions.
    """

    def maximumSum(self, nums):

        def digit_sum(num):
            return sum(int(char) for char in str(num))

        # Mapping to store the digits with an equal sum and a result variable to store the maximum sum pair
        mapping, max_sum = defaultdict(list), -1

        # Process all the numbers and store the results in the mapping
        for num in nums: mapping[digit_sum(num)].append(num)

        # If there are two or more numbers with the same sum of digits for a given key, potentially update the max sum
        for key in mapping:
            if len(mapping[key]) > 1: max_sum = max(max_sum, sum(heapq.nlargest(2, mapping[key])))

        # Return the max sum
        return max_sum

    """
    REDUCE ARRAY SIZE TO THE HALF

    You are given an integer array arr. You can choose a set of integers are remove all the occurrences of these integers in the array.

    Return the minimum size of the set so that at least half of the integers of the array are removed.
    """

    def minSetSize(self, arr):
        """
        Removal-count style heap problem.
        """

        # Generate a mapping with the occurrences of the elements in the list, a threshold to be met, the length of the original list and a result variable for the minimum size of the set
        counts, threshold, length, res = Counter(arr), len(arr) // 2 + 1, len(arr), 0

        # Generate a max heap with the occurrences
        heap = [-counts[num] for num in set(arr)]
        heapq.heapify(heap)

        while heap:
            top_occurrence = -heapq.heappop(heap)
            length -= top_occurrence
            res += 1
            if length < threshold: break 

        # Return the size of the removal set
        return res

    """
    MINIMUM DELETIONS TO MAKE CHARACTER FREQUENCIES UNIQUE

    A string s is called good if there are no two different characters in s that have the same frequency.

    Given a string s, return the minimum number of characters you need to delete to make s good.

    The frequency of a character in a string is the number of times it appears in the string.
    """

    def minDeletions(self, s):
        """
        For this problem we need to determine the unique occurrences and shave them until no two are equal.
        For each iteration we need to modify an occurrence we increment the number of modifications needed.
        Utilize a Counter to determine the character-occurrence pairs and list comprehension to add all the occurrences into the max heap.
        """
        
        # Mapping with the occurrences of characters and the number of deletions needed to make the string have characters with unique frequency
        occurrences, delete_count = Counter(s), 0

        # Max Heap with the occurrences of different characters in s - first generate a list of the inverted occurrences to utilize the heap as a max heap
        heap = [-occurrences[char] for char in occurrences]
        heapq.heapify(heap)
        
        # Continue to iterate until the heap has only one unique occurrence remaining
        while len(heap) > 1:
            current_top = -heapq.heappop(heap)
            if current_top == -heap[0]:
                if current_top > 1:
                    delete_count += 1
                    current_top -= 1
                    heapq.heappush(heap, -current_top)
                elif current_top == 1: delete_count += 1  
            else: continue

        # Return the minimum number of characters to delete before s is good after processing the elements in the max heap for occurrences
        return delete_count

    """
    FURTHEST BUILDING YOU CAN REACH

    You are given an integer array heights representing the heights of buildings, some bricks and some ladders.

    You start your journey from building 0 and move to the next building by possibly using bricks or ladders.

    While moving from building i to building i + 1 (0-indexed),

        - If the current building's height is greater than or equal to the next building's height, you do not need a ladder or bricks.

        - If the current building's height is less than the next building's height, you either use one ladder or (h[i + 1] - h[i]) bricks.

    Return the furthest building index (0-indexed) you can reach if you use the given ladders and bricks optimally.
    """

    def furthestBuilding(self, heights, bricks, ladders):
        """
        Simple underlying logic. Utilize the ladders for the larger climbs and the bricks for the smaller climbs.
        Greedy approach with heap to always store the largest possible climbs, if a climb is calculated to be negative we can simply continue to the next climb.
        """

        # A list that will effectively function as a heap using the heap methods - we always store the largest possible climbs using this heap
        heap = []

        # Iterate through the indices of the heights container from the first index up to the second to last index to compare the heights of adjacent buildings (possible height differences between two adjacent buildings)
        for i in range(1, len(heights)):
            
            # First determine the climb between adjacent buildings
            climb = heights[i] - heights[i - 1]
            if climb <= 0: continue 

            # Push this climb onto the heap and determine if the number of elements in the heap is less than or equal to the number of ladders
            heapq.heappush(heap, climb)
            if len(heap) <= ladders: continue 

            # If there are more climbs than ladders, we remove the smallest climb from the collection of climbs and utilize bricks for that climb - if this climb requires more bricks than we have return i 
            bricks -= heapq.heappop(heap)
            if bricks < 0: return i 

        # If all climbs can be covered with the ladders and bricks, return the final index
        return len(heights) - 1

    """
    MINIMUM FALLING PATH SUM II

    Given an n x n integer matrix grid, return the minimum sum of a falling path with non-zero shifts. 

    A falling path with non-zero shifts is a choice of exactly one element from each row of grid such that no two elements chosen in adjacent rows are in the same column.
    """
    
    def minFallingPathSumII(self, A):
        m, n = len(A), len(A[0])
        for i in range(1, m):
            smallest = heapq.nsmallest(2, A[i - 1])
            for j in range(n): A[i][j] += smallest[1] if A[i - 1][j] == smallest[0] else smallest[0]
        return min(A[-1])
    
    """
    KTH SMALLEST ELEMENT IN A SORTED MATRIX

    Given an n x n matrix where each of the rows and columns is sorted in ascending order, return the kth smallest element in the matrix.

    Note that it is the kth smallest element in the sorted order, not the kth distinct element. You must find a solution with a memory complexity better than O(N^2).
    """

    def kthSmallest(self, matrix, k):

        # The dimension of the matrix
        N = len(matrix)

        # Default cases - if the number k is equal to the dimensions of the matrix return the largest element, otherwise if k is 1 return the smallest element
        if N * N == k: return matrix[-1][-1]
        if k == 1: return matrix[0][0]

        # Generate a heap for the possible k rows, depending on whether k is greater or the dimensions of the matrix
        heap = []
        for row in range(min(k, len(matrix))): heapq.heappush(heap, (matrix[row][0], row, 0))

        # Iterate for the k specified times to determine the kth smallest element by continually extracting the smallest element from the heap
        for _ in range(k):
            element, row, col = heapq.heappop(heap)
            if col < len(matrix) - 1: heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))

        # Return the value of the last element popped off the heap
        return element
    