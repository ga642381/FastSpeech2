"""
    Reference
    https://tailo.moe.edu.tw/ium5.html
"""

# 聲母
lip = ['p', 'ph', 'm', 'b']
tongue_top = ['t', 'th', 'n', 'l']
tougue_tail = ['k', 'kh', 'g', 'ng']
throat = ['h']
tooth = ['ts', 'tsh', 's', 'j']
initials = [lip, tongue_top, tougue_tail, throat, tooth]

# 韻母
unit = ['a', 'i', 'u', 'e', 'o', 'oo']
complex_unit = ['ai', 'au', 'ia', 'iu', 'io', 'ua', 'ui', 'ue', 'iau', 'uai']
noise_unit = ['ann', 'inn', 'enn', 'onn', 'ainn', 'aunn', 'iann', 'iunn', 'uinn', 'iaunn', 'uann', 'uainn']
noise_vowel_0 = ['am', 'im', 'om', 'iam']
noise_vowel_1 = ['an', 'in', 'un', 'ian', 'uan']
noise_vowel_2 = ['ang', 'ing', 'ong', 'iang', 'iong', 'uang']
checked_tone_vowel_0 = ['ah', 'ih', 'uh', 'eh', 'oh', 'auh', 'iah', 'iuh', 'ioh', 'uah', 'uih', 'ueh', 'iauh']
checked_tone_vowel_1 = ['ap', 'ip', 'op', 'iap']
checked_tone_vowel_2 = ['at', 'it', 'ut', 'iat', 'uat']
checked_tone_vowel_3 = ['ak', 'ik', 'ok', 'iak', 'iok']
sound_vowel = ['m', 'ng']
other = ['er', 'erh', 'ir', 'irh', 'ere', 'ereh']
vowels = [other, unit, complex_unit, noise_unit, noise_vowel_0, noise_vowel_1, noise_vowel_2, checked_tone_vowel_0, checked_tone_vowel_1, checked_tone_vowel_2, checked_tone_vowel_3, sound_vowel]

# 聲母 + 韻母的所有 group 合起來
phn_list = initials + vowels

