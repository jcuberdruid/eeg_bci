import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self): #XXX todo add test 
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()
