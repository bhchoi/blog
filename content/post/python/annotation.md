---
title: "[Python] Annotation"
date: 2020-05-02 23:11:00 +0800
categories: [Python]
tags: [python, docstring]
toc: true
---

# Annotation이란?
* annotation은 function의 parameter와 return value에 대한 hint를 주는 것이다.
* 강제하는 것이 아니라 단지 알려주는 것이다.
* 파이썬은 동적으로 type을 결정하기 때문에 잘못된 type을 사용할 수 있다.

# 문법
## Parameters
* parameter 뒤에 annotation을 작성한다.
* type을 지정할 수도 있고, defaule value를 지정할 수도 있다.
```python
def foo(a: expression, b: expression=5):
    ...
def foo(a: str, b: float=1.5):
    ...
```

* *args와 **kwargs에도 적용된다.
```python
def foo(*args: expression, **kwargs: expression):
    ...
```

* 중첩해서 사용할 수도 있다.
```python
def foo((x1, y1: expression),
        (x2: expression, y2: expression)=(None, None)):
    ...
```

## Return Values
* "->" 뒤에 expression을 작성한다.
```python
def sum() -> expression:
    ...
def sum() -> int:
    ...
```

## Lambda
* lambda 문법에는 annotation이 적용되지 않는다.

# Annotation attribute
* annotation은 \_\_annotations__ attribute로 접근이 가능하다.  
![annotation_attribute](/images/python/annotation_attribute.png)

> 참조 : <a href="https://www.python.org/dev/peps/pep-3107/#grammar" target="_blank">PEP-3107</a>
