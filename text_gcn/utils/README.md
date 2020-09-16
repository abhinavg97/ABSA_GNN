# Description

The files in this folder contain standalone functions which can be used according to your own needs.

## text_processing.py

This file contains a class for text cleaning.
The text is cleaned by passing it through the following pipeline.

1. Remove extra whitespaces
2. decode to unicode
3. replace acronyms
4. lower case the string
5. expand contractions of english words like ain't
6. correct spelling mistakes
7. replace NE in the text

Import the class and use the .process_text method on the object with your text as an arguement to clean it.

## Other files

The function details of the other files can be easily understood by looking at their corresponding docstrings.