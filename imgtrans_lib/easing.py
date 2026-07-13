import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import *

def Linear(time, begin, change, duration):
    return change * time / duration + begin

def inQuad(time, begin, change, duration):
  return change * (time/duration)**2 + begin

def outQuad(time, begin, change, duration):
  return -change * (time/duration) * (time/duration - 2.) + begin

def inOutQuad(time, begin, change, duration):
  if time/duration < 0.5:
    return change / 2 * ((time/duration) * 2)** 2 + begin
  else:
    return -change / 2 * (((time/duration) * 2 - 1) * ((time/duration) * 2 - 3) - 1) + begin

def outInQuad(time, begin, change, duration):
  if time < duration / 2:
    return outQuad (time * 2, begin, change / 2, duration)
  else:
    return inQuad((time * 2) - duration, begin + change / 2, change / 2, duration)

def inCubic(time, begin, change, duration):
  return change * (time/duration)**3 + begin

def outCubic(time, begin, change, duration):
  return change * ((time/duration-1)**3 + 1) + begin

def inOutCubic(time, begin, change, duration):
  t2 = time/duration*2
  if time/duration < 0.5 :
    return change / 2 * t2**3 + begin
  else:
    return change / 2 * ((t2-2)**3 + 2) + begin

def outInCubic(time, begin, change, duration):
  if time/duration < 0.5:
    return outCubic(time * 2, begin, change / 2, duration)
  else:
    return inCubic((time * 2) - duration, begin + change / 2, change / 2, duration)


def inQuart(time, begin, change, duration):
  return change *(time/duration)**4 + begin


def outQuart(time, begin, change, duration):
  return -change * (( time / duration - 1)**4- 1) + begin

def inOutQuart(time, begin, change, duration):
  if time/duration < 0.5:
    return change / 2 * (time / duration * 2)**4 + begin
  else:
    return -change / 2 * ((time / duration*2-2)**4 - 2) + begin


def outInQuart(time, begin, change, duration):
  if time/duration < 0.5:
    return outQuart(time * 2, begin, change / 2, duration)
  else:
    return inQuart((time * 2) - duration, begin + change / 2, change / 2, duration)



from  math import *
def inSine(time, begin, change, duration):
  return -change * cos(time / duration * (pi / 2)) + change + begin

def outSine(time, begin, change, duration):
  return change * sin(time / duration * (pi / 2)) + begin

def inOutSine(time, begin, change, duration):
  return -change / 2 * (cos(pi * time / duration) - 1) + begin

def outInSine(time, begin, change, duration):
  if time/duration < 0.5:
    return outSine(time * 2, begin, change / 2, duration)
  else:
    return inSine((time * 2) -duration, begin + change / 2, change / 2, duration)


def inCirc(time, begin, change, duration):
  return change * (1 - sqrt(1 - time / duration)) + begin


def outCirc(time, begin, change, duration):
  return change * sqrt(time / duration) + begin


def inOutCirc(time, begin, change, duration):
  if time/duration < 0.5:
    return -change / 2 * (sqrt(1 - (time / duration * 2) **2) - 1) + begin
  else:
    return change / 2 * (sqrt(1 - (time / duration * 2 - 2) **2) + 1) + begin


def outInCirc(time, begin, change, duration):
  if time/duration < 0.5:
    return outCirc(time * 2, begin, change / 2, duration)
  else:
    return inCirc((time * 2) - duration, begin + change / 2, change / 2, duration)


def InElastic(time, begin, change, duration):
    return change * (time/duration)**4  * sin(time/duration * pi * 4.5 ) + begin


def OUtElastic(time, begin, change, duration):
    return (1 -  (time/duration - 1)**4 * cos( time/duration * pi * 4.5 )) * change + begin


def InOutElastic(time, begin, change, duration):
    if time/duration < 0.45:
        return change *8 * (time/duration)**4 * sin( time/duration * pi * 9 ) + begin
    elif time/duration < 0.55:
        return change*(0.5 + 0.75 * sin( time/duration * pi * 4 )) + begin
    else:
        return change * (1 - 8 * (time/duration - 1)**4 * sin( time/duration * pi * 9)) + begin


def outBounce(time, begin, change, duration):
  if  time / duration < 1 / 2.75:
    return change * (7.5625 * (time/duration)**2) + begin
  elif  time / duration < 2 / 2.75:
    return change * (7.5625 * (time/duration - (1.5 / 2.75))**2 + 0.75) + begin
  elif time/duration < 2.5 / 2.75:
    return change * (7.5625 *(time/duration - (2.25 / 2.75))**2+ 0.9375) + begin
  else:
    return change * (7.5625 * (time/duration - (2.625 / 2.75))**2+ 0.984375) + begin


def inBounce(time, begin, change, duration):
  return change - outBounce(duration - time, 0, change, duration) + begin

def inOutBounce(time, begin, change, duration):
  if time/duration < 0.5 :
    return inBounce(time * 2, 0, change, duration) * 0.5 + begin
  else:
    return outBounce(time * 2 - duration, 0, change, duration) * 0.5 + change * .5 + begin


def outInBounce(time, begin, change, duration):
  if time/duration < 0.5 :
    return outBounce(time * 2, begin, change / 2, duration)
  else:
    return inBounce((time * 2) - duration, begin + change / 2, change / 2, duration)

def sampleFunc(time, begin, change, duration):
  return -change * (( time / duration - 1)**4- 1) + begin

class Easing_Simulator:
    start = 0.0
    change = 100.0
    duration = 3.0

    def __init__(self):
        self.time = 0.0
        # plot設定
        self.fig = plt.figure()
        self.ax = plt.axes(xlim = (-30,130),  ylim=(-5,5))
        self.circle = plt.Circle((0., 0.), 5)
        self.ax.set_aspect("equal")
        self.ax.add_patch(self.circle)
        self.anim =FuncAnimation(self.fig, self.update, interval=33, frames = 30)


    def update(self, _):
        self.time += 0.1
        self.circle.center = (sampleFunc(self.time, self.start, self.change, self.duration), 0)
        # print(self.time,self.circle.center )
        
    def show(self): 
        plt.show()

    def save(self):
        self.anim.save("test.gif",writer='imagemagick', fps = 30)

class graph_plot:
    start = 0
    change = 10
    duration = 20
        
    def __init__(self):
        self.time = 0

    def plot(self):
        pos = []
        for i in range(100):
            # pos.append(sampleFunc(self.time, self.start, self.change, self.duration))
            # pos.append(outInQuad(self.time, self.start, self.change, self.duration))
            # pos.append(Linear(self.time, self.start, self.change, self.duration))
            # pos.append(inOutQuad(self.time, self.start, self.change, self.duration))
            # pos.append(inCubic(self.time, self.start, self.change, self.duration))
            # pos.append(inOutCubic(self.time, self.start, self.change, self.duration))
            # pos.append(outCubic(self.time, self.start, self.change, self.duration))
            pos.append(inQuart(self.time, self.start, self.change, self.duration))
            
            self.time += 1
        plt.plot(pos)
        plt.show()

# P = graph_plot()
# P.plot()
