from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
import nltk.corpus
import pandas as pd
import re2

# TOKEN is copy/pasted from token_regex.txt
TOKEN = re2.compile('''(?:(?:https?:\\/\\/)?(?:(?:(?:[^!\"\#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~ \t\n\v\f\r\u0000-\u001F\u007F\uFFFE\ufeff\uFFFF\u202a-\u202e\t-\r \u0085\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000](?:[_-]|[^!\"\#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~ \t\n\v\f\r\u0000-\u001F\u007F\uFFFE\ufeff\uFFFF\u202a-\u202e\t-\r \u0085\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000])*)?[^!\"\#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~ \t\n\v\f\r\u0000-\u001F\u007F\uFFFE\ufeff\uFFFF\u202a-\u202e\t-\r \u0085\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]\\.)+(?:(?:[^!\"\#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~ \t\n\v\f\r\u0000-\u001F\u007F\uFFFE\ufeff\uFFFF\u202a-\u202e\t-\r \u0085\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000](?:[-]|[^!\"\#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~ \t\n\v\f\r\u0000-\u001F\u007F\uFFFE\ufeff\uFFFF\u202a-\u202e\t-\r \u0085\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000])*)?[^!\"\#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~ \t\n\v\f\r\u0000-\u001F\u007F\uFFFE\ufeff\uFFFF\u202a-\u202e\t-\r \u0085\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000])(?:(?:\\.xn--[0-9a-z]+))?)(?::([0-9]+))?(?:\\/(?:(?:[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]*(?:(?:[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]+|(?:[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]*\\([a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]+\\)[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]*))[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]*)*[a-zA-Z\\p{Cyrillic}0-9=_#\\/\\+\\-\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]|(?:(?:[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]+|(?:[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]*\\([a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]+\\)[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]*))))|(?:[a-zA-Z\\p{Cyrillic}0-9!\\*';:=\\+\\,\\.\\$\\/%#\\[\\]\\-_~&\\|@\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0259\u025b\u0263\u0268\u026f\u0272\u0289\u028b\u02bb\u0300-\u036f\u1e00-\u1eff]+\\/))*)?(?:\\?[a-zA-Z0-9!?\\*'\\(\\);:&=\\+\\$\\/%#\\[\\]\\-_\\.,~|@]*[a-zA-Z0-9_&=#\\/\\-])?)|(?:[<>]?[:;=8][\\-o\\*\\']?[\\)\\]\\(\\[dDpPS\\/\\:\\}\\{@\\|\\\\]|[\\)\\]\\(\\[dDpPS\\/\\:\\}\\{@\\|\\\\][\\-o\\*\\']?[:;=8][<>]?|<3)|(?:-+>|<-+)|(?:[@\uff20][a-zA-Z0-9_]+(?:\\/[a-zA-Z][a-zA-Z0-9_\\-]*)?)|(?:[#\uff03][\\p{L}\\p{M}\\p{Nd}_\u200c\u200d\ua67e\u05be\u05f3\u05f4\uff5e\u301c\u309b\u309c\u30a0\u30fb\u3003\u0f0b\u0f0c\u00b7]+)|(?:(?:[\\pL]+['\\-_]+)+[\\pL]+)|(?:[+\\-]?[\\pN]+([,\\/.:\\-][\\pN]+)*[+\\-]?)|(?:[\\pL\\pN_]+)|(?:[^\t-\r \u0085\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\\pL\\pN]+)''')

stopwords = set(word.encode('utf-8')
                for word in nltk.corpus.stopwords.words('english'))
USEFUL_CHAR = re2.compile('\\pL|\\pN')


class GentleValueError(ValueError):
    """
    A ValueError that should not display in red to the user.

    On first load, we don't want to display an error, even though the user
    hasn't selected what to plot. So we'll display the error in the iframe:
    we'll be gentle with the user.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def series_to_text(series: pd.Series) -> str:
    """
    Build a str with text contents of `series`, newline-separated.

    We'll pass this to the tokenizer so we only need to call it once. The
    newlines guarantee the last token in row 1 won't merge with the first token
    in row 2.
    """
    return series.dropna().astype(str).str.cat(sep='\n')


def text_to_tokens(text: str) -> Iterable[bytes]:
    """
    Tokenize the input string, returning non-stopword tokens.

    Output bytes, not str, because re2 is best with bytes and we needn't
    convert back to str until after counting tokens.
    """
    textb = text.lower().encode('utf-8')

    pos = 0
    while True:
        match = TOKEN.search(textb, pos)
        if match is None:
            break

        token = match.group(0)
        pos = match.end()

        if token not in stopwords and USEFUL_CHAR.search(token):
            yield token


def most_common_tokens(tokens: Iterable[bytes],
                       max_n_tokens=100) -> List[Tuple[str, int]]:
    """
    Build `(token, count)` pairs for the most frequent tokens in `tokens`.

    Converts tokens back to utf-8.
    """
    counter = Counter(tokens)
    most_common = counter.most_common(max_n_tokens)
    return [(token.decode('utf-8'), n) for token, n in most_common]


class Form:
    """
    User input.
    """
    def __init__(self, column):
        self.column = column

    def to_chart(self, table):
        series = table[self.column]
        text = series_to_text(series)
        tokens = text_to_tokens(text)
        common_tokens = most_common_tokens(tokens)

        if not common_tokens:
            raise GentleValueError('Column contains no words')

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
                            'font': (
                                'Nunito Sans, Helvetica Neue, Helvetica, Arial'
                            ),
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
