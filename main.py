from itertools import repeat
from random import randint
import parser
from random import sample

global equation
global total


def splitArray(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class Equation:
    def __init__(self, code=None, total=0):
        self.code = code
        self.total = total


class Genetic:
    def __init__(self, gen=None, id=None):
        if gen is not None:
            self.gen = gen
        else:
            self.gen = [randint(1, 10) for _ in range(1, len(equation.code.co_names) + 1)]

        if id is not None:
            self.id = gen
        else:
            self.id = randint(0, 2)

    def fitness(self):
        list = self.gen.copy()
        # деофантовая функция
        zs = dict(zip(equation.code.co_names, list))
        xx = eval(equation.code, zs)
        return xx - total

    def crossover(self, f):
        # гены исходной особи
        s = self.gen.copy()
        mid = self.id

        # гены некоторой другой особи
        ss = f.genotype().copy()
        fid = f.show()

        if mid == 0:
            if mid == fid:
                newList = ss[1:]
                newList.insert(0, s[0])
                return Genetic(newList)
            else:
                newList = s[1:]
                newList.insert(0, ss[0])
                return Genetic(newList)
        elif mid == 1:
            if mid == fid:
                return Genetic(list(splitArray(s, 2))[0] + list(splitArray(ss, 2))[1])
            else:
                return Genetic(list(splitArray(ss, 2))[0] + list(splitArray(s, 2))[1])
        else:
            if mid == fid:
                newList = s[:-1]
                newList.append(ss[-1])
                return Genetic(newList)
            else:
                newList = ss[:-1]
                newList.append(s[-1])
                return Genetic(newList)

    def genotype(self):
        return self.gen

    def show(self):
        return self.id


class Population:
    def __init__(self, num=10):
        self.p = list(repeat(Genetic(), num))

    def cross(self):
        l = len(self.p)
        # создаем две перемешанных копии популяции
        # для выполнения кроссовера
        f = sample(self.p, l)
        m = sample(self.p, l)
        z = []
        # сам кроссовер
        for i in range(0, len(f)):
            z.append(f[i].crossover(m[i]))

        def check(a, b):
            if a.fitness() < b.fitness():
                return a
            else:
                return b

        # популяция после отбора
        self.p = list(map(check, self.p, z))

    def mutate(self):
        """
       Генератор мутаций внутри популяции.
       """
        l = len(list(self.p))
        num = int(l * 0.4213) + 1  # количество мутаций

        for i in range(1, num):
            v = randint(0, l - 1)  # элемент популяции нужный для мутации
            p = self.p.pop(v)  # запоминаем этот элемент и удаляем его из популяции
            ex = p.genotype()  # выясняем его генотип
            ind = randint(0, 3)  # выбираем элемент генотипа для мутации
            ex.pop(ind)  # удаляем его
            ex.insert(ind, randint(1, 30))  # вставляем мутацию
            self.p.append(Genetic(ex))  # вставляем мутированную особь

    @property
    def view(self):
        return self.p


# уравнение
eq = "a + 2*b + 3*c + 4*d + 2*k"
# результат
total = 35
code = parser.expr(eq).compile()
equation = Equation(code, total)

# популяция из 150 особей
x = Population(150)

# запускаем развитие в течение 15 поколений.
for i in range(1, 15):
    x.cross()
    x.mutate()

# отсеиваем особей с генотипами, соответствующими решениям
# диофантова уравнения
k = [i for i in x.view if i.fitness() == 0]

if len(k) == 0:
    print(u'Решений не обнаружено !')
else:
    for i in k:
        print(i.genotype())
