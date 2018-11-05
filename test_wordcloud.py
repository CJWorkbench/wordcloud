import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from wordcloud import text_to_tokens, most_common_tokens, render


class WordcloudTest(unittest.TestCase):
    def test_text_to_tokens(self):
        self.assertEqual(list(text_to_tokens("@adamhooper hi, I'm Bob!")),
                         ['@adamhooper', 'hi', "i'm", 'bob'])

    def test_allow_embedded_punctuation(self):
        self.assertEqual(list(text_to_tokens("@adamhooper I'm")),
                         ['@adamhooper', "i'm"])

    def test_allow_numbers(self):
        self.assertEqual(list(text_to_tokens(' 3 1,234.56 ')),
                         ['3', '1,234.56'])

    def test_nix_stopwords(self):
        self.assertEqual(list(text_to_tokens('i, the ketchup')),
                         ['ketchup'])

    def test_most_common_tokens(self):
        tokens = ['@adamhooper', 'hi', 'hi', 'bob', '@adamhooper', 'hi']
        expected = [
            ('hi', 3),
            ('@adamhooper', 2),
            ('bob', 1),
        ]
        self.assertEqual(most_common_tokens(tokens), expected)

    def test_most_common_tokens_limit(self):
        tokens = ['@adamhooper', 'hi', 'hi', 'bob', '@adamhooper', 'hi']
        expected = [
            ('hi', 3),
            ('@adamhooper', 2),
        ]
        self.assertEqual(most_common_tokens(tokens, 2), expected)

    def test_integration(self):
        table = pd.DataFrame({
            'A': ['@adamhooper hi', 'hi bob', '@adamhooper hi']
        })
        result = render(table, {'column': 'A'})
        self.assertEqual(result[1], '')  # error
        assert_frame_equal(result[0], table)  # dataframe
        self.assertEqual(result[2]['data'][0]['values'], [
            {'text': 'hi', 'n': 3},
            {'text': '@adamhooper', 'n': 2},
            {'text': 'bob', 'n': 1},
        ])  # json

    def test_no_column_error(self):
        table = pd.DataFrame({
            'A': ['@adamhooper hi', 'hi bob', '@adamhooper hi']
        })
        result = render(table, {'column': ''})
        assert_frame_equal(result[0], table)  # dataframe
        self.assertEqual(result[1], '')  # error
        self.assertEqual(result[2]['error'],
                         'Please select a text column')  # json

    def test_no_text(self):
        table = pd.DataFrame({
            'A': ['.', '.', '.']
        })
        result = render(table, {'column': 'A'})
        assert_frame_equal(result[0], table)  # dataframe
        self.assertEqual(result[1], '')  # error
        self.assertEqual(result[2]['error'],
                         'Column contains no words')  # json
