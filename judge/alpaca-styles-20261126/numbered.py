import re
mark=re.compile(r'^([1-9]\d*)\.[ \t]+(\S.*)$')
def sentence(s):
    clean=re.sub(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc)\.|\b(?:[A-Za-z]\.){2,}', 'ABBR',s,flags=re.I)
    return bool(re.search(r'[A-Za-z]',s)) and bool(re.search(r'[.!?][”’"\')\]]*$',s)) and not re.search(r'[.!?][”’"\')\]]*\s+(?=[“"\(]?[A-Z0-9])',clean)
def metric(text,capped=False):
    ls=[l.strip() for l in text.splitlines() if l.strip()]
    ms=[mark.match(l) for l in ls]; start=bool(ms and ms[0] and ms[0][1]=='1')
    valid=0
    for i,m in enumerate(ms,1):
        if not(m and int(m[1])==i and sentence(m[2])):break
        valid+=1
    strict=valid==len(ls) and valid>=1 and not capped
    numbers=[int(m[1]) for m in ms if m]
    later=any(b==a+1 for a,b in zip(numbers,numbers[1:]))
    first_number=int(ms[0][1]) if ms and ms[0] else None
    initial_run=0
    for i,m in enumerate(ms,1):
        if not (m and int(m[1])==i):break
        initial_run+=1
    return dict(second=int(initial_run>=2),all_lines=int(initial_run==len(ls) and bool(ls) and not capped),strict=int(strict),start=int(start),late=int(later and first_number is None),wrong_start=int(first_number is not None and first_number!=1),first_number=first_number,prefix=valid/max(1,len(ls)),fmt=2 if strict else int(start or later))
