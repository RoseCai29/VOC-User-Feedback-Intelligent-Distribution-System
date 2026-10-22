"""
emoji 预处理 —— 只剥离、只做弱信号统计，不武断推断情绪极性。

为什么不做极性判断: 同一个 emoji 在不同语境下含义相反，「太好用了😭😭😭」里的
😭 表达的是感动/夸张，而不是难过；「😅」「[微笑]」在中文互联网上还常带反讽。
因此这里只统计事实（出现了多少、属于哪一类），极性交给文本语义或后续模型判断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Unicode emoji（含变体选择符与零宽连接符）
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u2764]+"
)
# 中文论坛常见的方括号表情
_BRACKET_RE = re.compile(
    r"\[(微笑|大笑|偷笑|流泪|哭|发怒|生气|鼓掌|赞|抱拳|捂脸|旺柴|无语|裂开)\]"
)

# 语义相对明确的强负面表情，仅用于“文本也负面”时的强度加权
_STRONG_NEGATIVE = set("😡😠🤬💢😤👎🖕")
_POSITIVE = set("😀😃😄😁😆😊😇🙂😉😌😍🥰😘👍❤💯🎉✨")
# 语境依赖、不能单独定性的表情
_AMBIGUOUS = set("😭🥺😢😔😞😟😕🙁☹😣😖😫😩😂🤣😅🙃😑😐😶🙏")
_BRACKET_POSITIVE = {"大笑", "偷笑", "鼓掌", "赞"}
_BRACKET_NEGATIVE = {"流泪", "哭", "发怒", "生气"}
_BRACKET_AMBIGUOUS = {"微笑", "抱拳", "捂脸", "旺柴", "无语", "裂开"}


@dataclass(frozen=True)
class EmojiProfile:
    text: str                    # 剥离 emoji 后的文本
    count: int = 0
    strong_negative: int = 0
    positive: int = 0
    ambiguous: int = 0
    only_emoji: bool = False     # 原文除 emoji 外没有有效内容


def _add(profile: dict[str, int], char: str) -> None:
    if char in _STRONG_NEGATIVE:
        profile["strong_negative"] += 1
    elif char in _POSITIVE:
        profile["positive"] += 1
    elif char in _AMBIGUOUS:
        profile["ambiguous"] += 1


def analyze_emoji(text: str) -> EmojiProfile:
    """剥离 emoji 并统计各类数量；对无法定性的表情归入 ambiguous。"""
    counts = {"count": 0, "strong_negative": 0, "positive": 0, "ambiguous": 0}

    for match in _EMOJI_RE.finditer(text):
        for char in match.group():
            if char in ("\uFE0F", "\u200D"):
                continue
            counts["count"] += 1
            _add(counts, char)

    def _replace_bracket(match: re.Match) -> str:
        name = match.group(1)
        counts["count"] += 1
        if name in _BRACKET_POSITIVE:
            counts["positive"] += 1
        elif name in _BRACKET_NEGATIVE:
            counts["strong_negative"] += 1
        else:
            counts["ambiguous"] += 1
        return ""

    clean = _BRACKET_RE.sub(_replace_bracket, text)
    clean = _EMOJI_RE.sub("", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return EmojiProfile(
        text=clean,
        only_emoji=bool(text.strip()) and not clean,
        **counts,
    )


def strip_emoji(text: str) -> str:
    """分类/去重使用的文本口径；纯 emoji 时回退原文，避免输入变空。"""
    return analyze_emoji(text).text or text
