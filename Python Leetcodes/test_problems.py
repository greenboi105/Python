import problems1 as design
import unittest  

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

class TestProblems(unittest.TestCase):

    def testCircularQueue(self):

        CircularQueue = design.MyCircularQueue(3)
        self.assertEqual(CircularQueue.enQueue(1), True)
        self.assertEqual(CircularQueue.enQueue(2), True)
        self.assertEqual(CircularQueue.enQueue(3), True)

        self.assertEqual(CircularQueue.enQueue(4), False)
        self.assertEqual(CircularQueue.Rear(), 3)
    
    def testHitCounter(self):

        HitCounter = design.HitCounter()
        HitCounter.hit(1)
        HitCounter.hit(2)
        HitCounter.hit(3)
        self.assertEqual(HitCounter.getHits(4), 3)
        HitCounter.hit(300)
        self.assertEqual(HitCounter.getHits(300), 4)
        self.assertEqual(HitCounter.getHits(301), 3)
    
    def testStockSpanner(self):

        StockSpanner = design.StockSpanner()
        self.assertEqual(StockSpanner.next(100), 1)
        self.assertEqual(StockSpanner.next(80), 1)
        self.assertEqual(StockSpanner.next(60), 1)
        self.assertEqual(StockSpanner.next(70), 2)
        self.assertEqual(StockSpanner.next(60), 1)
        self.assertEqual(StockSpanner.next(75), 4)
        self.assertEqual(StockSpanner.next(85), 6)
        self.assertEqual(StockSpanner.next(99), 7)
        self.assertEqual(StockSpanner.next(90), 1)
        self.assertEqual(StockSpanner.next(101), 10)

    def testFileSystem(self):

        FileSystem = design.FileSystem()
        self.assertEqual(FileSystem.createPath("/leet", 1), True)
        self.assertEqual(FileSystem.createPath("/leet/code", 2), True)
        self.assertEqual(FileSystem.createPath("/c/d", 1), False)
        self.assertEqual(FileSystem.get("/c"), -1)
        self.assertEqual(FileSystem.get("/leet/code"), 2)

    def testLeaderboard(self):

        leaderboard = design.Leaderboard()
        leaderboard.addScore(1, 73)
        leaderboard.addScore(2, 56)
        leaderboard.addScore(3, 39)
        leaderboard.addScore(4, 51)
        leaderboard.addScore(5, 4)
        self.assertEqual(leaderboard.top(1), 73)
        leaderboard.reset(1)
        leaderboard.reset(2)
        leaderboard.addScore(2, 51)
        self.assertEqual(leaderboard.top(3), 141)

if __name__ == '__main__':
    unittest.main()
