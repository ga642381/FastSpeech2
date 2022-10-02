from lexicon.initial_finals import initials, vowels


print(initials)
print(vowels)
tone = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
init, vow = [], []
for ls in initials:
    init.extend(ls)
for ls in vowels:
    vow.extend(ls)

lexicon = {}
for st in init:
    for ed in vow:
        for t in tone:
            lexicon[f"{st}{ed}{t}"] = [st, ed + t]
for ed in vow:
    for t in tone:
        lexicon[f"{ed}{t}"] = [ed + t]


with open("lexicon/taiwanese.txt", 'w', encoding='utf-8') as f:
    for k, v in lexicon.items():
        f.write(f"{k}\t{' '.join(v)}\n")
