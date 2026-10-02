"""Regex tags for responses that talk about the prompt instead of answering it.

``context``: the response comments on the provided or background text, for
example "the text does not contain this information". ``garbled``: it says the
question has typos or is corrupted. Patterns cover English, Russian, French,
Chinese and Arabic and are a diagnostic, not a judge score.
"""

from __future__ import annotations

import re

GARBLED = re.compile(
    r"typo|misspel|garbl|corrupt|gibberish|nonsensical|jumble"
    r"|(assuming|i assume|perhaps|likely|probably) you meant|formatting (error|issue)|encoding error"
    r"|опечат|искаж|бессмысл|набор (слов|символов)"
    r"|faute de frappe|coquille|erreur de (frappe|saisie)|incohéren|charabia"
    r"|乱码|错别字|拼写错误|输入错误|无意义"
    r"|أخطاء (إملائية|في الصياغة|في الكتابة)|تشويش|غير مفهوم|غير واضحة",
    re.I,
)
CONTEXT = re.compile(
    r"\b(text|passage|background|context|document)( you)? (provided|given|attached|above|below|after)"
    r"|\b(provided|given|attached|following|accompanying) (text|passage|background|context|document)"
    r"|unrelated background|ignore (it|the (unrelated|background|text))"
    r"|does not (contain|mention|include) (any )?(information|details)|no information (about|regarding|on)"
    r"|(предоставленн|приведённ|приведенн|прикреплённ|прикрепленн)\w* (вами )?текст"
    r"|в (этом |данном )?тексте (нет|не )|фонов\w* текст"
    r"|texte (fourni|ci-dessous|ci-dessus|suivant|que vous avez)|texte de fond"
    r"|提供的(文本|文字|段落|内容)|背景(文本|信息)|(文本|文中)并?(没有|未)"
    r"|النص (المقدم|المرفق|الذي (قدمته|أرفقته|ترفقه))|لا يحتوي النص|غير ذي صلة",
    re.I,
)


def meta_kind(response: str) -> str | None:
    head = response[:600]
    if GARBLED.search(head):
        return "garbled"
    if CONTEXT.search(head):
        return "context"
    return None
