import collections
import math
from random import randint
import re
from collections import Counter
from collections import defaultdict
from itertools import groupby

"""
range(0,3) == range(3) → [0,1,2]
range(3,0,-1) → [3,2,1]

lst=[1,2,3,4,5,6]
lst[1::2] → [2,4,6], lst[0::2] → [1,3,5]
lst[:3] == lst[0:3] → [1, 2, 3] BUT lst[3]=4
lst[:-1] → [1, 2, 3, 4, 5]
lst[:2]+lst[3:] → [1,3,4,5,6] #removes 2nd element

Counter(‘abbcd’) → ({'b': 2, 'a': 1, 'c': 1, 'd': 1})
dict(list(enumerate('abbcd'))) → {0: 'a', 1: 'b', 2: 'b', 3: 'c', 4: 'd'}
[k for k, g in groupby('AAAABBBCCDAABBB')] → A B C D A B
[list(g) for k, g in groupby('AAAABBBCCD')] → AAAA BBB CC D

ddic = defaultdict(int)
ddic[‘first’]=1, ddic[‘sec’]=2 → defaultdict(None, {‘first’: 1, ‘sec’: 2})
ddic.get(‘first’) → 1

contests = [[5, 1], [2, 1], [1, 0]]
[row[1] for row in contests] → [1, 1, 0] #get all elements in column 1
zeros = sum([elem[0] for elem in contests]) → 8 #sums all elements for column 0
contests[2:][:] → [[2, 1], [1, 0]]

"""
flightDuration=90
movieDuration=[1,10,25,35,60]



def foo(flightDuration, movieDuration):
    best=0
    result = -1,-1
    for xi,x in enumerate(movieDuration):
        for yi,y in enumerate(movieDuration[xi:],xi):
            if xi != yi:
                total = x+y
                if total<=flightDuration-30:
                    if total > best:
                        best = total
                        result = xi,yi
                    if total == best and max(x,y)>max(*result):
                        result = xi, yi
    return result


s = '|**|*|*'
starti = [1,1]
endi=[5,6]
res=[]
for start, end in zip(starti, endi):
    s_to_use = s[start-1:end]
    s_comp = s_to_use.strip('*')
    print(start, end, s_to_use, s_comp)
    res.append(s_comp.count('*'))
