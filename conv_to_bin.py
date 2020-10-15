num = input()
def pad(x):
    x = str(x)
    while len(x)<4:
        x = '0' + x
    return x

ans = ''    
for digit in str(num):
    y = bin(int(str(digit)))[2:]
    y = pad(y)
    ans += y

print(ans)
