# Description

The files in this folder contain standalone functions which can be used according to your own needs.

## text_processing.py

This file contains a class for text cleaning.
The text is cleaned by passing it through the following pipeline.

1. Decode the text into unicode
2. Remove extra repeated special characters
3. Put space around the special characters
4. Remove trailing whitespaces
5. Replace acronyms
6. Expand contractions of english words like ain't
7. Correct spelling mistakes
8. Replace NE in the text
9. Lower case the string

Import the class and use the .process_text method on the object with your text as an arguement to clean it.

NOTE:

For 2. 2. Remove extra repeated special characters, the special characters are as mentioned below:

* !
* @
* #
* $
* %
* ^
* &
* *
* ,
* .
* /
* ?
* '
* "
* ;
* :
* \

For 3. Put space around the special characters, the special characters are as mentioned below:

* $
* ?
* %
* @
* !
* #
* ^
* *
* &
* "
* :
* ;
* /
* \
* ,
* +
* (
* )
* [
* ]
* {
* }
* <
* >

The cleaned datasets can be downloaded from the data folder of the root directory of this repository.


## Other files

The function details of the other files can be easily understood by looking at their corresponding docstrings.