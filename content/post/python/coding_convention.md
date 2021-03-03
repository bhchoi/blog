---
title: "[Python] coding convention 및 tool"
date: 2020-05-10 13:11:00 +0800
categories: [Python]
tags: [python, style checker, formatter, convention]
toc: true
---

# Coding Convention이란?
* coding convention은 code를 작성하는 구체적인 가이드라인이다.
* 협업을 위해서는 읽기 쉽고 일관적인 코드를 작성하는 것이 중요하다.
* 하지만 매우 많은 기준을 모두 지키면서 code를 작성하는 것이 매우 힘들다.
* 자동화 툴을 적용해보자

# pylint
* pylint는 PEP-8을 준수했는지 체크하는 정적 코드 분석 tool
* cli 및 ide에서 사용 가능하다.

### vscode pylint 적용방법
* Command Palette를 누른다.
* Python:Select Linter 입력하여 선택한다.
* pylint 선택  
![pylint](/images/python/vscode_pylint.gif)
* 적용결과  
![pylint_code](/images/python/pylint_code.png)  
![pylint_result](/images/python/pylint_result.png)

# Black
* Black은 PEP-8을 기반으로 자동으로 code를 수정해주는 formatter이다.

### vscode Black 적용방법
* black 설치
```shell
pip install black
```
* settings.json에 옵션 추가
```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length",
        "100"
    ],
    "editor.formatOnSave": false
}
```
* code를 작성하고 save를 하면 자동으로 format을 변경해준다.
![black](/images/python/vscode_black.gif)