from g2pM import G2pM
model = G2pM()
for s in ['零','一','幺','二','三','四','五','六','七','八','九','一二三','幺二三']:
    print('TEXT', s)
    print('tone=True,char_split=False ->', model(s, tone=True, char_split=False))
    print('tone=False,char_split=False ->', model(s, tone=False, char_split=False))
    print('tone=True,char_split=True ->', model(s, tone=True, char_split=True))
    print()