wordcloud
---------

Workbench module that shows the most-common words in a text column.

Features
--------

* Tokenizes with `NLTK https://www.nltk.org/>`_ "casual" algorithm, converting
  to lowercase.
* Stems with ``nltk.stem.snowball.EnglishStemmer``.
* Ignores English stopwords while stemming.

Developing
----------

First, get up and running:

#. ``python3 ./setup.py test`` # to test

To add a feature:

#. Write a test in ``test_wordcloud.py``
#. Run ``python3 ./setup.py test`` to prove it breaks
#. Edit ``wordcloud.py`` to make the test pass
#. Run ``python3 ./setup.py test`` to prove it works
#. Commit and submit a pull request

To develop continuously on Workbench:

#. Check this code out in a sibling directory to your checked-out Workbench code
#. Start Workbench with ``CACHE_MODULES=false bin/dev start``
#. In a separate tab in the Workbench directory, run ``bin/dev develop-module wordcloud``
#. Edit this code; the module will be reloaded in Workbench immediately
