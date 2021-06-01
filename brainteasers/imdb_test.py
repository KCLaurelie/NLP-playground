"""
QUESTION 1

during a flight, passengers can watch movies up to 30 minutes before the flight lands.
Given the list of movie durations available in the flight, pick 2 distinct movies that will last exactly the entire flight
If several pairs are possible, pick the pair that contains the longest lasting movie
return the indices of the optimal pair
"""
flightDuration=90
movieDuration=[1,10,25,35,60]
movieDuration=[1, 10, 25, 35, 60, 40, 20, 10, 15, 60]

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

# does not work if several movies have the same length
def foo2(flightDuration, movieDuration):
    movies_dic = {v: k for k, v in enumerate(movieDuration)}
    for _len in sorted(movies_dic, reverse=True):
        remaining_len = flightDuration-30-_len
        if remaining_len in movies_dic:
            return movies_dic[remaining_len], movies_dic[_len]
    return -1,-1
#foo2(90, [1,10,25,35,60])

import timeit
print(timeit.timeit('foo(90, [1, 10, 25, 35, 60, 40, 20, 10, 15, 60])', globals=globals()))

"""
QUESTION 2

amazon has a list of items '*' contained in containers ('||') or not
e.g. |**| means 2 items in a container, |*|***| means 1 and 3 items in 2 containers
given a string of items and containers, count the number of items that are inside containers
e.g. |*|***|** --> 4
start/end indicate which portion of the string to look at
"""
s = '|**|*|*'
starti = [1,1]
endi=[5,6]
res=[]
for start, end in zip(starti, endi):
    s_to_use = s[start-1:end]
    s_comp = s_to_use.strip('*')
    print(start, end, s_to_use, s_comp)
    res.append(s_comp.count('*'))
