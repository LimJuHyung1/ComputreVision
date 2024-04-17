from collections import Counter
# Counter는 열거형 자료의 원소의 갯수를 사전형 자료로 반환해 주는 함수이다.

# -------------------------------------------------------------------------------
# 실습 1: 문자열의 글자수 세기
# 리스트나 문자열과 같은 시퀀스 자료형 안의 요소 중 값이 같은 것이 몇 개 있는지 확인한다.
# -------------------------------------------------------------------------------
#"""
a = 'gallahad'
a = 'Seokyeong university'

text = list(a)
print(text)

dict = Counter(text)
print(dict)

print(dict['e'])

exit(0)
#"""



"""
# -------------------------------------------------------------------------------
# 실습 2: 1) dict 자료 혹은 2) keyword=수량 지정을 이용하여 필요 수량 만큼의 리스트 원소 생성하기
# -------------------------------------------------------------------------------

dict = {'eagle': 3, 'tiger': 1, 'horse': 2}
#dict = {5: 3, 9: 1, 7: 2}

print(type(dict))
# <class 'dict'>

c = Counter(dict)                       # 1)  dict 자료 지정
# Counter({'eagle': 3, 'horse': 2, 'tiger': 1})

print(c)
# ['eagle', 'eagle', 'eagle', 'tiger', 'horse', 'horse']

a = list(c.elements())
print(a)

c = Counter(cats=2, dogs=3)             # 2) keyword=수량 지정
print(c)
# Counter({'dogs': 3, 'cats': 2})

a = list(c.elements())
print(a)
# ['cats', 'cats', 'dogs', 'dogs', 'dogs']

exit(0)
"""



# -------------------------------------------------------------------------------
# 실습 3: 카운터 모듈의 사칙 연산
# -------------------------------------------------------------------------------

a = Counter(a=4, b=3, c=5, d=-5)
b = Counter(a=-1, b=2, c=0, d=4)

print(a.subtract(b))
print('a=', a)
# a= Counter({'a': 5, 'c': 5, 'b': 1, 'd': -9})


print('a+b=', a+b)
# a+b= Counter({'c': 5, 'a': 4, 'b': 3})
# 'd' 원소가 없어졌다?


print('a & b=', a & b)
# a & b= Counter({'b': 1})






