from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from nltk.tokenize.casual import casual_tokenize
from nltk.stem.snowball import EnglishStemmer
import pandas as pd


class GentleValueError(ValueError):
    """
    A ValueError that should not display in red to the user.

    On first load, we don't want to display an error, even though the user
    hasn't selected what to plot. So we'll display the error in the iframe:
    we'll be gentle with the user.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def series_to_str(series: pd.Series) -> str:
    """
    Build a str with text contents of `series`, newline-separated.

    We'll pass this to the tokenizer so we only need to call it once. The
    newlines guarantee the last token in row 1 won't merge with the first token
    in row 2.
    """
    return series.dropna().astype(str).str.cat(sep='\n')


def str_to_tokens(s: str) -> List[str]:
    """
    Tokenize the input string.
    """
    return casual_tokenize(s, preserve_case=False)


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Stem tokens.
    """
    stemmer = EnglishStemmer(ignore_stopwords=True)
    stemmed = [stemmer.stem(token) for token in tokens]
    nonempty = [s for s in stemmed if s]
    return nonempty


def most_common_tokens(tokens: List[str],
                       max_n_tokens=100) -> List[Tuple[str, int]]:
    """
    Build `(token, count)` pairs for the most frequent tokens in `tokens`.
    """
    counter = Counter(tokens)
    return counter.most_common(max_n_tokens)


class Form:
    """
    User input.
    """
    def __init__(self, column):
        self.column = column

    def to_chart(self, table):
        series = table[self.column]
        text = series_to_str(series)
        tokens = str_to_tokens(text)
        stemmed_tokens = stem_tokens(tokens)
        common_tokens = most_common_tokens(stemmed_tokens)
        return Chart(common_tokens)

    @staticmethod
    def parse(*, column: Optional[str]=None) -> 'Form':
        if not column:
            raise GentleValueError('Please select a text column')

        return Form(column)


class Chart:
    """
    Data we can structure into a Vega chart.
    """
    def __init__(self, tokens: List[Tuple[str, int]]):
        self.tokens = tokens

    def to_vega_data_values(self) -> List[Dict[str, Any]]:
        """
        Build `{ text, n }` dicts.
        """
        return [{'text': text, 'n': n} for text, n in self.tokens]

    def to_vega(self) -> Dict[str, Any]:
        """
        Build a Vega wordcloud.
        """
        return {
            '$schema': 'https://vega.github.io/schema/vega/v4.json',
            'data': [
                {
                    'name': 'tokens',
                    'values': self.to_vega_data_values(),
                    'transform': [
                        {
                            'type': 'wordcloud',
                            'text': {'field': 'text'},
                            'fontSize': {'field': 'n'},
                            'font': 'Helvetica Neue, Helvetica, Arial',
                            'fontSizeRange': [10, 56],
                            'rotate': 0
                        },
                    ],
                },
            ],
            'marks': [
                {
                    'type': 'text',
                    'from': {'data': 'tokens'},
                    'encode': {
                        'enter': {
                            'text': {'field': 'text'},
                            'align': {'value': 'center'},
                            'baseline': {'value': 'alphabetic'},
                            'fill': {'value': '#333333'},
                        },
                        'update': {
                            'x': {'field': 'x'},
                            'y': {'field': 'y'},
                            'fontSize': {'field': 'fontSize'},
                            'fillOpacity': {'value': 1},
                        },
                        'hover': {
                            'fillOpacity': {'value': 0.75},
                        },
                    },
                }
            ]
        }


def render(table, params):
    try:
        form = Form.parse(**params)
        chart = form.to_chart(table)
    except GentleValueError as err:
        return (table, '', {'error': str(err)})

    json_dict = chart.to_vega()
    return (table, '', json_dict)
