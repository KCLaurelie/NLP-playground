"""
QUESTION 1
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

def foo2(flightDuration, movieDuration):
    movies_dic = {v: k for k, v in list(enumerate(movieDuration))}
    for _len in sorted(movies_dic, reverse=True):
        print(_len)
        remaining_len = flightDuration-30-_len
        if remaining_len in movies_dic:
            return movies_dic[remaining_len], movies_dic[_len]
    return -1,-1
foo2(90, [1,10,25,35,60])

"""
QUESTION 2
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
