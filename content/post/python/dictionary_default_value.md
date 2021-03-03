---
title: "[Python] dictionary에 default value 사용하기"
date: 2020-12-16 09:11:00 +0800
categories: [Python]
tags: [python, Python Tricks]
toc: true
---

> Python Tricks 책의 7.1 Dictionary Default Values 정리 내용입니다.

아래와 같은 dictionary가 있을때
```python
arsenal_player_number = {
    14: "auba",
    10: "ozil",
    35: "gabi",
}
```

각 선수이름에 해당하는 등번호를 출력하고 싶다.
```python
def print_number(id):
    return f"안녕 {arsenal_player_number[id]}"
```

만약 선수이름이 dictionary에 존재하지 않을 경우 exception이 발생한다.
```
>>> print_number(14)
'안녕 auba'

>>> print_number(9)
KeyError: 9
```

key가 없을 경우 어떻게 처리를 할까?
먼저 dictionary에 key가 존재하는지 확인하는 방법이 있다.
```python
def print_number(id):
    if id in arsenal_player_number:
        return f"안녕 {arsenal_player_number[id]}"
    else:
        return "안녕 친구"
```
```
>>> print_number(14)
'안녕 auba'

>>> print_number(9)
'안녕 친구'
```

하지만 이 방식에는 개선할 부분이 있다.
* dictionary에 2번 접근
* 이 예시의 경우, string이 반복적으로 사용된다.
* pythonic하지 않다. 공식적으로 python은 EAFP 스타일을 추천한다.
  * EAFP : easier to ask for forgiveness than permission
  * python은 dictionary에 key가 유효하다는 가정하에 사용하고, key가 없으면 exception 처리를 한다.
  
EAFP를 적용하여, KeyError를 예외처리해보자.
```python
def print_number(id):
    try:
        return f"{name}: {arsenal_player_number[name]}"
    except KeyError:
        return "안녕 친구"
```

dictionary에 2번 접근하는 것은 해결했다.
좀 더 깔끔한 방법으로 해보자.
```python
def print_number(id):
    return f"안녕 {arsenal_player_number.get(id, '친구')}"
```

get()에 key값과 key가 없을 경우 사용할 default 값을 함께 넘긴다.
```
>>> print_number(10)
'안녕 ozil'

>>> print_number(9)
'안녕 친구'
```

아예 처음부터 default 값을 가지고 있는 defaultdict를 활용할 수도 있다.
```python
import collections
arsenal_player_number = collections.defaultdict(lambda: "친구")
arsenal_player_number[14] = "auba"
arsenal_player_number[10] = "ozil"
arsenal_player_number[35] = "gabi"
```
```python
def print_number(id):
    return f"안녕 {arsenal_player_number[id]}"
```
```
>>> print_number(10)
'안녕 ozil'

>>> print_number(9)
'안녕 친구'
```

아래 처럼 default 값을 가지도록 만들어서, null check를 하지 않고 사용할 수 있다.
```python
import collections
default_list_dict = collections.defaultdict(list)
default_str_dict = collections.defaultdict(str)
default_int_dict = collections.defaultdict(int)
```

```
>>> default_list_dict[0]
[]

>>> default_str_dict[0]
''

>>> default_int_dict[0]
0
```