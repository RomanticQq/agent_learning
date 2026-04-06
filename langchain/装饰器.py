import time

def fun1(f):
    def fun2(s1:str, s2:str) -> str:
        print(time.time())
        res = f(s1, s2)
        print(time.time())
        return res
    return fun2

@fun1
def s_concat(s1:str, s2:str)->str:
    s = s1 + s2
    return s

print(s_concat('a','b'))