from collections import defaultdict

d = defaultdict(lambda: 0)          # Default 값을 0으로 설정
# lambda: 0는 return 0로 이해한다. 즉, 어떤 파라미터가 들어오더라도 0을 반환한다는 것이다.
print(d["first"])       # lambda 함수는 d["first"] 함수를 대변한다. 람다 함수가 호출하면 0을 반환하는 것으로 선언하였다.
print(d[3])

print(type(d), len(d))


d = defaultdict()
print(type(d), len(d))
print(d[3])
