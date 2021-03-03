---
title: "[Python]Decorator"
date: 2020-06-06 13:11:00 +0800
categories: [Python]
tags: [python, decorator, pep-318]
toc: true
---

# Decorator란?
디자인 패턴 중 하나인 decorator pattern과 비슷한 개념으로, 반복적인 것을 공통화하고, 메소드나 클래스의 기능을 확장할 수 있는 방법이다.  
자세한 내용은 예제를 통해 알아보자.

# Decorator 예제
* 먼저 decorator로 사용할 function을 정의한다.
* function 상단에 @ 심볼과 decorator function명을 입력한다.  
```python
def decorator_func(original_function):
    def wrapper():
        print("decorator 호출")
        return original_function()
    return wrapper

@decorator_func
def func():
    print('func 호출')

func()

# decorator 호출
# func 호출
```

* function arguments도 추가할 수 있다.
```python
def decorator_func(original_function):
    def wrapper(*args, **kwargs):
        print("decorator 호출")
        return original_function()
    return wrapper

@decorator_func
def func(name):
    print('func({}) 호출'.format(name))

func("func")

# decorator 호출
# func(alex) 호출
```
* 여러개의 decorator를 붙일 수 있다.
```python
def time_logging_func(func):
    def wrapper(*args, **kwargs):
        print(datetime.datetime.now())
        return func(*args, **kwargs)
    return wrapper


def decorator_func(func):
    def wrapper(*args, **kwargs):
        print("decorator 호출")
        return func(*args, **kwargs)
    return wrapper

@time_logging_func
@decorator_func
def func(name):
    print('func({}) 호출'.format(name))

func("alex")

# 2020-06-06 23:49:27.586175
# decorator 호출
# func(alex) 호출
```