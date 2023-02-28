from random import randint

def f_bruteforce(buildings):
    maximum = 0
    i_max, l_max, r_max = 0, 0, 0
    for i in range(len(buildings)):
        l = i
        r = i

        while l - 1 >= 0 and buildings[l-1] >= buildings[l]:
            l -= 1
        while r + 1 < len(buildings) and buildings[r] <= buildings[r+1]:
            r += 1

        if r - l + 1 > maximum:
            maximum = max(maximum, r - l + 1)
            i_max = i
            l_max = l
            r_max = r

    #print(f"maximum at {i_max} [{l_max},{r_max}]")
    return maximum

def f(buildings):
    prev = 0
    maximum = 0
    lmax = 0
    rmax = 0
    imax = 0
    i = 0
    while i <= len(buildings):
        if i + 1 < len(buildings) and buildings[i] <= buildings[i+1]:
            # stop going from left
            # print(f"minima at {i}, prev = {prev} | ", end="")
            j = i
            while j + 1 < len(buildings) and buildings[j] <= buildings[j+1]:
                j += 1

            #print(f"i={i}, l={prev}, r={j}")
            if j - prev + 1 > maximum:
                maximum = max(maximum,j - prev + 1)
                imax = i
                lmax = prev
                rmax = j

            # now go to right
            prev = i +1
            i = j
        else:
            i += 1
    #print(f"Prog 2 : {imax} [{lmax}, {rmax}]")
    return maximum
def eval(val):
    print(f_bruteforce(val), f(val))

eval([2, 6, 8, 5])
eval([1,5,5,2,6])
eval([10, 10, 10, 3, 10, 10, 10,10,10,10,10])
eval([11, 19, 12, 3, 132, 11, 10,10,10,10])

k = [randint(0,10) for _ in range(20)]
print(k)

eval(
    k
)