from collections import Counter
files = open("sample.txt",'r')
f = files.read()
print(f)
words = f.split(" ")
print(words)
word_count = Counter(words)
print(word_count)
cap_words=[word.upper() for word in words]
print(cap_words)