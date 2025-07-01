**Understanding Unicode**
a) The Unicode character 0 is the terminate-string character. It is not a regular character.

b) Its repr is '\\x00' while the str is '\x00'.

c) For me the character does not display anything when used outside a print statement. In a print it shows as '\x00'. Online discussion, show that some terminals might skip the following output.


**Unicode encodings**
a) It takes up less space.

b) It works only for single-byte chars as the function are splitting per byte. It will fail for 'Å' for example.

c) 0xa5 followed by any other byte. It is not a valid start byte, and can only be a valid end-byte in two or more-byte characters.

