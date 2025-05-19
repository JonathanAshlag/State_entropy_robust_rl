from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall,Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gym.envs.registration import register
import numpy as np
import gym
import pygame
import matplotlib.pyplot as plt
class EmptyEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps= 200,
        blocked=False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.grid_size = size

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        #declare the action space
        self.action_space = gym.spaces.Discrete(3)
        #declare the observation space
        self.blocked = blocked
        self.observation_space = gym.spaces.Dict({'image': gym.spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype='uint8')})
        self.reset()
        self.occupancy = np.zeros((self.grid_size, self.grid_size))
        self.possible_block_modes =[False,1,2,3,4]

        


    @staticmethod
    def _gen_mission():
        return "reach the goal"
    def seed(self, seed=None):
        # Seed the random number generator
        np.random.seed(seed)
        self.seed = seed
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        
        if self.blocked == 1:
             x_blocked = np.random.randint(2, width-3)
             x_blocked=4
             blocks= [(x_blocked,5),(x_blocked,4),(x_blocked,3),(x_blocked,2),(x_blocked,1)]
             for i in range(len(blocks)):
                self.grid.set(blocks[i][0],blocks[i][1],Wall())

        elif self.blocked == 2:
            y_blocked = np.random.randint(1, height-5)
            blocks =[(5,y_blocked),(6,y_blocked),(7,y_blocked),(5,y_blocked+1),(5,y_blocked+2)]
            for i in range(len(blocks)):
                self.grid.set(blocks[i][0],blocks[i][1],Wall())
            
        elif self.blocked == 3:
            all_points = [[x, y] for x in range(2, width-1) for y in range(2, height-1)]
            #remove the points (2,2),(8,8):
            all_points.remove([2,2])
            all_points.remove([8,8])
            all_points = np.array(all_points)
            
            num_points = 7
            random_indices = np.random.choice(all_points.shape[0], num_points, replace=False)
            random_points = all_points[random_indices]
            for i in range(len(random_points)):
                self.grid.set(random_points[i][0],random_points[i][1],Wall())





        # Place a goal square in the bottom-right corner
        if self.blocked == 4:
            # generate a random position for the goal
            x_goal = np.random.randint(2, width-2)
            y_goal = np.random.randint(2, height-2)
            # set the goal position
            self.put_obj(Goal(), x_goal, y_goal)
            
        else:
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "reach the goal"
        
    #inherit the step function from the parent class
    def step(self, action):
        obs, reward, done,truncated, info = super().step(action)

        if isinstance(self.grid.get(*self.agent_pos), self.PenaltyTile):
            #with probability 0.75 a penalty is given:
            if np.random.rand() < 0.75:
                reward += -0.2
            #if steps limit isnt met, then the agent isnt done
            if self.step_count < self.max_steps:
                done = False
        #mask unwanted actions
        # if action in [3,4,5,6]:
        #     reward += -10

        if done or truncated:
            done = True
        self.occupancy[self.agent_pos[0], self.agent_pos[1]] += 1


        return obs, reward, done, info
    
    def reset(self,seed=None):
        self.occupancy = np.zeros((self.grid_size, self.grid_size))
        obs,_ = super().reset()
        return obs
    
    class PenaltyTile(Lava):
        def __init__(self):
            super().__init__()
        
        def render(self,img):
             super().render(img)
            
            
        def can_overlap(self):
            return True #allows the agent to step on the tile
    


def main():
    env = EmptyEnv(blocked= 4,render_mode='human')
    pygame.init()
    while True:
        env.render()
    # enable manual control for testing
    # manual_control = ManualControl(env, seed=42,)
    
    # manual_control.start()
    # print(env.agent_dir)
    
    
    
if __name__ == "__main__":
    main()
#register the environment:
register(
    id='MiniGrid-EmptyEnv-v0',
    entry_point='utils.custom_env:EmptyEnv'
)
