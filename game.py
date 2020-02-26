
from scipy.special import softmax
import numpy as np
import time
import toricgame
import neat
import matplotlib.pyplot as plt

class ToricCodeGame():
    def __init__(self, board_size, error_rate, max_steps, epsilon):
        self.board_size = board_size
        self.error_rate = error_rate
        self.max_steps = max_steps
        self.epsilon = epsilon

        self.env = toricgame.ToricGameEnv()
        self.env.init(self.board_size, self.error_rate)

    # Return the score of the game
    def play(self, nn, verbose=False):
        fitness = 0
        current_state = self.env.reset()
        if verbose:
            print("Initial", current_state)
            print(self.env.done, self.env.state.syndrome_pos)
            self.draw(current_state, 3)


        # If there is no syndrome in the initial configuration
        # Either we generate a new one containing syndromes
        # Or if there happens to be a logical error, we return a failure
        while self.env.done and self.error_rate>0:
            if self.env.reward == -1:
                return 0

            current_state = self.env.reset()
            if verbose:
                print("Initial", current_state)
                print(self.env.done, self.env.state.syndrome_pos)
                self.draw(current_state, 3)

        for step in range(self.max_steps+1):

            if type(nn) is neat.nn.FeedForwardNetwork:
                # Generate perspectives
                action = self.generate_perspectives(current_state, nn)

            current_state, reward, done, info = self.env.step(action)

            if verbose:
                print(step, current_state, reward, action, info["message"])
                self.draw(current_state, 3)


            # if no syndromes are present anymore
            if done:
                #print(step, (reward+1)/2, info["message"], self.env.initialmoves, self.env.state.action_to_coord(self.env.initialmoves[0]), action)
                #fitness = float(step)/float(self.max_steps)

                # Reward is 1 if there is no logical error
                # -1 if there is a logical error
                fitness = (reward+1)/2
                #print(fitness)
                break

        return fitness

    def generate_perspectives(self, current_state, nn):
        actions=[]
        probs=[]
        for plaq in self.env.state.syndrome_pos:
            perspec = np.roll(current_state, (self.board_size - plaq[0])%(2*self.board_size), axis=0)
            perspec = np.roll(perspec, (self.board_size - plaq[1])%(2*self.board_size), axis=1)

            # Output has 4 neurons
            probs += nn.activate(perspec.flatten())
            actions += [[(plaq[0]+1)%(2*self.board_size), plaq[1]]]
            actions += [[(plaq[0]-1)%(2*self.board_size), plaq[1]]]
            actions += [[plaq[0], (plaq[1]+1)%(2*self.board_size)]]
            actions += [[plaq[0], (plaq[1]-1)%(2*self.board_size)]]

        #print("actions", actions)
        #print("probs", probs)

        # epsilon-greedy search
        if np.random.rand() < self.epsilon:
            action = actions[np.random.randint(len(actions))]
        else:
            action = actions[np.argmax(probs)]

        return action


    def draw(self, array, d = 3):
        fig, ax = plt.subplots(dpi=300)

        scale = 3/d

        artists = []
        for p in self.env.state.plaquet_pos:
            fc = 'white' if array[p[1],p[0]]==0 else 'darkorange'#[1,0,0,0.8]
            plaq = plt.Rectangle( (-0.7 + scale*p[0]*0.25 - scale*0.25, 0.7 - scale*p[1]*0.25 + scale*0.25), scale*0.5, -0.5*scale, fc=fc, ec='black' )
            artists.append( ax.add_patch(plaq) )

        for p in self.env.state.qubit_pos:
            circle = plt.Circle( (-0.7 + scale*0.25*p[0], 0.7 - scale*0.25*p[1]), radius=scale*0.05, ec='k', fc='darkgrey' if array[p[1],p[0]] == 0 else 'darkblue')
            artists.append( ax.add_patch(circle) )

        #for p in g.toriccode.plaquet_pos:
        #    ax.text( -0.72 + 0.25*p[0], 0.68 - 0.25*p[1], "p")

        #for s in g.toriccode.star_pos:
        #    ax.text( -0.7 + 0.25*s[0], 0.7 - 0.25*s[1], "s")

        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

    def close(self):
        self.env.close()
