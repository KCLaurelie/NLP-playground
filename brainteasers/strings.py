from itertools import groupby
import re
from collections import Counter

#region 7. String segmentation
"""
You are given a dictionary of words and a large input string. 
You have to find out whether the input string can be completely segmented into the words of a given dictionary. 
e.g.
Given a dictionary of words: apple apple pear pie
Input string of “applepie” can be segmented into dictionary words. apple pie

Runtime Complexity: Exponential, O(2^n)
Memory Complexity: Polynomial, O(n^2)
"""

class Solution7(object):
    def wordBreak(self,dict, str, out=""):
        # if the end of the string is reached,
        # print the output string
        if not str:
            print(out)
            return

        for i in range(1, len(str) + 1):
            # consider all prefixes of the current string
            prefix = str[:i]

            # if the prefix is present in the dictionary, add it to the
            # output string and recur for the remaining string
            if prefix in dict:
                self.wordBreak(dict, str[i:], out + " " + prefix)

dict = ["self", "th", "is", "famous", "Word", "break", "b", "r", "e", "a", "k", "br","bre", "brea", "ak", "problem"]
Solution7().wordBreak(dict, "Wordbreakproblem")

#endregion

#region 8. Reverse words in a sentence

class Solution8(object):
    def rev_sentence(self, sentence):
        # first split the string into words
        words = sentence.split(' ')

        # then reverse the split string list and join using space
        reverse_sentence = ' '.join(reversed(words))

        # finally return the joined string
        return reverse_sentence

Solution8().rev_sentence('hello world')
#endregion

#region Count anagrams in string
def get_all_substrings(input_string):
    length = len(input_string)
    return [input_string[i:j + 1] for i in range(length) for j in range(i, length)]

def get_unordered_anagram_count(string):
    allsubs = get_all_substrings(string)
    subsorted = [sorted(x) for x in allsubs]
    subsorted.sort()
    print(subsorted)
    count = 0;
    curr_count = 0;
    for j in range(len(subsorted) - 1):
        if subsorted[j] == subsorted[j + 1]:
            curr_count += 1
            count += curr_count
        else:
            curr_count = 0

    return count

get_unordered_anagram_count('mom')
#endregion

#region sherlock anagrams
def sherlockAndAnagrams(s):
    all_subs = []
    # get all possible substrings of s
    # for anagrams, order doesn't matter so we will sort all substrings
    # if 2 substrings are anagrams they will now be the same
    for i in range(1,len(s)):
        for j in range(0,len(s)-i+1):
            sub = s[j:j+i]
            all_subs.append(''.join(sorted(sub)))  #to sort substring alphabetically

    # now we count how many times appears each substring. if >1 then it has an anagram
    count = Counter(all_subs)
    count_ana = {k:v for k,v in count.items() if v>1}
    # if a substring appears v times, then we can make 1+2+...+v-1 anagrams out of it
    # 1+2+...+v-1 = sum(range(v))
    return sum(sum(range(v)) for v in count.values())

s = 'abba'
sherlockAndAnagrams(s)
#endregion

#region Common element in 2 strings?
# METHOD 1 This is too slow
def twoStrings(s1, s2):
    # Write your code here
    l1 = len(s1)
    l2 = len(s2)

    for i in range(l1):
        for j in range(i + 1, l1 + 1):
            stem = s1[i:j]
            if stem in s2:
                return "YES"
    return "NO"
# METHOD 2 using set
def twoStrings(s1, s2):
    set1 = set(s1) #converting string to set
    set2 = set(s2)
    if set.intersection(set1,set2):
        return "YES"
    else:
        return "NO"
#endregion

#region how many elements t delete to make anagrams?
def makeAnagram(a, b):
    cnt_a = Counter(a)
    cnt_b = Counter(b)
    delete_from_a = cnt_a - cnt_b
    delete_from_b = cnt_b - cnt_a
    deletions = delete_from_a + delete_from_b
    return len(list(deletions.elements()))
#endregion

#region longest common substring
def longestSubstring(s1, s2):
    # Write your code here
    answer = ""
    len1, len2 = len(s1), len(s2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and s1[i + j] == s2[j]):
                match += s2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return len(answer)
#endregion

#region common child between strings (longest common substring if deleting elems allowed
# NOT THE FASTEST
def commonChild(s1, s2):
    # Write your code here
    l1, l2 = len(s1), len(s2)
    lengths = [[0 for j in range(l2 + 1)] for i in range(l1 + 1)]
    for i1, el1 in enumerate(s1):
        for i2, el2 in enumerate(s2):
            if el1 == el2:
                lengths[i1 + 1][i2 + 1] = lengths[i1][i2] + 1
            else:
                lengths[i1 + 1][i2 + 1] = \
                    max(lengths[i1 + 1][i2], lengths[i1][i2 + 1])

    return lengths[-1][-1]

#endregion

#region Count deletions needed to remove matching adjacent characters in string
def alternatingCharacters(s):
    # Write your code here
    i=0
    num_del=0
    while i < len(s)-1:
        if s[i]==s[i+1]: # while consecutive elements are equal
            s = s[0 : i : ] + s[i + 1 : :] # remove second element
            num_del+=1 #increment number of deletions
        else: # otherwise move to next element
            i+=1
    return num_del
#endregion

#region Sherlock valid string
#https://www.martinkysel.com/hackerrank-sherlock-and-valid-string-solution/
def isValid(s):
    char_map = Counter(s)
    char_occurence_map = Counter(char_map.values())

    if len(char_occurence_map) == 1:
        return "YES"

    if len(char_occurence_map) == 2:
        k1, k2 = char_occurence_map.keys()
        v1, v2 = char_occurence_map.values()

        # there is exactly 1 extra symbol and it can be deleted
        if (k1 == 1 and v1 == 1) or (k2 == 1 and v2 == 1):
            return "YES"

        # the is exactly 1 symbol that occurs an extra 1 time
        if (k1 == k2 + 1 and v1 == 1) or (k2 == k1 + 1 and v2 == 1):
            return "YES"
    return "NO"
#endregion

#region All substrings of string
def subString(s):
    n = len(s)
    res = []
    for i in range(n):
        for _len in range(i + 1, n + 1):
            res.append(s[i: _len])
            print(s[i: _len])
s = "abcd"
subString(s)
#endregion

#region All Special substrings of string
"""
Special string defined by
All of the characters are the same, e.g. aaa. (case A)
All characters except the middle one are the same, e.g. aadaa (case B)
"""
# METHOD1
def substrCount(s):
    count = n = len(s)
    for i, char in enumerate(s):
        diff_char_idx = None
        for j in range(i + 1, n):
            if char == s[j]:
                if diff_char_idx is None:
                    count += 1
                elif j - diff_char_idx == diff_char_idx - i:
                    count += 1
                    break
            else:
                if diff_char_idx is None:
                    diff_char_idx = j
                else:
                    break
    return count
#METHOD2 using groupby

def substrCount(s):
    n = len(s)
    cpt_a = cpt_b = 0

    # count the number of cases A
    a_cases = [len(list(g)) for k, g in groupby(s)]
    cpt_a = sum([i*(i+1)//2 for i in a_cases])

    # count the number of cases B
    for i in range(1, n-1):
        skip = 1
        if s[i-skip] == s[i] or s[i+skip] == s[i]:
            continue # already counted in case A, move back to top of lfor oop
        match = s[i-skip]
        while i-skip > -1 and i+skip < n and s[i-skip] == s[i+skip] == match:
            cpt_b += 1
            skip += 1
    return cpt_a+cpt_b

substrCount('mnonopoo')
#endregion

#region Abbreviations
a = "AbcDE"
b = "ABDE"
b="AFDE"

def abbreviation(a,b):
    if set(b.lower())-set(a.lower()) == set():
        print("YES")
    else:
        print("NO")

q = int(input().strip())
for q_itr in range(q):
    a = input()
    b = input()
    result = abbreviation(a, b)
#endregion

#region Simple Negex
"""
find mentions of suicide in text and check if they are negated or not"""
target = 'suicid'
context_words = 5
context_chars=20
neg_terms = ['no', "n't", 'deni', 'deny']
string = 'he patient denies having suicidal thoughts. This was not an intentional overdose. ' \
         'She has been suicidal in the past. Suicidal ideation was not intentional. ' \
         'blabla random sentence???? ' \
         'another random stuff'


class SolutionNegex(object):
    # WORKING ON WORDS
    def negex_words(self, string, target, neg_terms=['no', "n't"], context_words=5):
        sentences = [snt.lower().split() for snt in re.split('[.,!?;]', string)]
        neg=[]
        for word_list in sentences:
            idx_lst = [i for i, word in enumerate(word_list) if target in word] # find if suicide words used
            if idx_lst == []:
                neg.append('no_word')
            else:
                for idx in idx_lst:
                    context_list = word_list[max(0,idx-context_words):min(len(word_list), idx+context_words)]
                    context_str = ''.join(context_list)
                    neg.append(max([neg in context_str for neg in neg_terms]))
                    print(context_str, neg)
        return [sentences, neg]

    #WORKING ON STRINGS
    def negex_chars(self, string, target, neg_terms=['no', "n't"], context_chars=20):
        idx_lst = [i for i in range(len(string)) if string.lower().startswith(target, i)]
        print(idx_lst)
        res=[]
        for ix in idx_lst:
            context_str = string[max(0,ix-context_chars):min(len(string),ix+context_chars+len(target))]
            neg = max([neg in context_str for neg in neg_terms])
            print(context_str, neg)
            res.append([context_str, neg])
        return 0

SolutionNegex().negex_words(string, target, neg_terms = ['no', "n't", 'deni', 'deny'])
SolutionNegex().negex_chars(string, target, neg_terms = ['no', "n't", 'deni', 'deny'])

#endregion
