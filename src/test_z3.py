from z3 import *

# x = Int('x')
# y = Int('y')
# solve(x > 2, y < 10, x + 2*y > 7)

# x = Int('x')
# y = Int('y')
# n = x + y >= 3
# print("x.decl().name(): ", x.decl().name())
# print("num args: ", n.num_args())
# print("children: ", n.children())
# print("1st child:", n.arg(0))
# print("2nd child:", n.arg(1))
# print("operator: ", n.decl())
# print("op name:  ", n.decl().name())


# v = Real('v')
# w = Real('w')
# z = Bool('z')
# s = Solver()
# s.add(2*v + 2 > 3 * w + 4)
# print(s.check())
# # print(s.statistics())
# print(s.model())
# s.add(v > 3, z == (1 == 1))
# s.check()
# m = s.model()
# print(m)
# print(("%s" % m[z]) == "True")

# X = [ Int('x%s' % i) for i in range(5) ]
# Y = [ Int('y%s' % i) for i in range(5) ]
# print(X)
# X = IntVector('x', 5)
# Y = RealVector('y', 5)
# P = BoolVector('p', 5)
# print X
# print Y
# print P
# print([ y**2 for y in Y ])
# print(Sum([ y**2 for y in Y ]))

# x = z3.Real('x') 
# dlr = z3.RealVal(0) 
# dlr = z3.If(True, 1, dlr)
# dlx = z3.Real('dlx')
# s = Solver()
# s.add(dlx + dlr == x)
# s.check()
# print(s.model())

# print(1 + z3.Real('x'))
# print(50247917214211602138089591205120157/4000000000000000000000000000000)



# b1 = Bool('b1')
# b2 = Bool('b2')
# b3 = Bool('b3')

# bor : Probe | BoolRef = z3.And(b1 == True)
# s = Solver()
# s.add(bor, And())
# decision = s.check()
# print((decision))
# print(s.model())

# ss = Solver()
# ss.append(b2 == True)
# s.add(ss)
# s.check()
# print(s.model())


def not_pairs(bs):
    return And([(Not(And(b1, b2))) 
            for i, b1 in enumerate(bs)
            for j, b2 in enumerate(bs)
            if i < j])

c1 = Bool('c1')
d1 = Bool('d1')

c2 = Bool('c2')
d2 = Bool('d2')

ite = If(c1, d1, If(c2, d2, False))

imps = And(Implies(c1,d1),  Implies(c2, d2)) 
ors = Or(c1, c2)
nps = not_pairs([c1, c2])

# solve(Not(Implies(And(ite, nps), imps)))
solve(Not(Implies(And(ite, nps), And(ors, imps))))

x = Real('x')
y = Real('y')
# solve(y == x / 0)

outcol_mean = [h + t for h,t in zip([x], [y])]
solve(*[x > 3 for x in outcol_mean])
xs = [0] * 5
xs[0] = 7
xs[1] = 7
xs[2] = 7
xs[3] = 7
print(xs)