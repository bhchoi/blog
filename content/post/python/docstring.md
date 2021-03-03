---
title: "[Python] Docstring"
date: 2020-05-02 23:11:00 +0800
categories: [Python]
tags: [python, docstring]
toc: true
---

# Docstring이란?
* docstring은 module, function, class, method에 대한 문서화 작업을 말한다.
* docstring은 \_\_doc__ special attribute에 저장된다.

# Docstring 작성 방법
## One-line Docstrings
docstring을 한 줄에 작성하는 방법이다.
```python
def get_name():
    """name을 리턴한다."""
    ...
```
* 메소드의 summary를 작성한다.
* 쌍따옴표 3개를 사용하여 시작하고 종료한다.
* docstring 앞 뒤로 blank line이 없다.

## Multi-line Docstrings
```python
def set_profile(name=None, age=None):
    """프로필(name, age)를 업데이트한다.

    Keyword arguments:
    name -- 이름 (default None)
    age  -- 나이 (default None)
    """
    ...
```
* 첫번째 줄은 summary를 작성한다.
* blank line을 추가한 후 나머지를 작성한다.
* arguments, return values, side effects, exceptions, restrictions 등을 작성한다.

## \_\_doc__ attribute
![doc attribute](/images/python/doc_attribute.png)
> 참조 : <a href="https://www.python.org/dev/peps/pep-0257" target="_blank">PEP-257</a>
