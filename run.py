from mdp import gridworld, PolicyIteration, wrapper
import numpy as np



grids = gridworld(8)

grids.draw_grids()

grids.draw_plot(rewards = grids.R)


agent = PolicyIteration(grids, policy_init = None)
agent.iterate()
grids.draw_policy(grids.policy)
agent.iterate(True)

'''
grids.draw_grids(grids.V)

grids.draw_plot(values = grids.V)

grids.policy = np.random.randint(0, 5, size = [400]) 
'''

grids.draw_policy(grids.policy)


env = wrapper(grids)
#env.render()

def haha(func):
    def store_sample(*args, haha=False):
        func(*args)
        print(*args)
        if haha:
            print("haha is True")
        else:
            print("haha is False")
    return store_sample

@haha
def func(hehe):
    print(hehe)

func("hehe", haha = True)
