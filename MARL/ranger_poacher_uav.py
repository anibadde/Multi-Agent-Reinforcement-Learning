import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

#Where/How is observation function called?

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 8
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            if i==0:
                agent.name = 'poacher'
                agent.size = 0.075
                agent.id = 1
                agent.accel = 2
                agent.max_speed = 1
                agent.sight = 0.3
                agent.color = np.array([0.85, 0.35, 0.35])
            if i==1:
                agent.name = 'ranger'
                agent.size = 0.075
                agent.id = 2
                agent.accel = 2.15
                agent.max_speed = 1
                agent.sight = 0.3
                agent.color = np.array([0.35, 0.35, 0.85])
            if i==2:
                agent.name = 'uav' 
                agent.size = 0.05
                agent.id = 3
                agent.accel = 3
                agent.max_speed = 1.3
                agent.sight = 0.5
                agent.color = np.array([0.35, 0.85, 0.35])
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'animal %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.collisionTimes = 0
            landmark.captured = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if agent.name == 'uav':
                agent.sight = 0.5
            else:
                agent.sight = 0.3
        for i, landmark in enumerate(world.landmarks):
            if i==0:
                landmark.state.p_pos = np.array([0.647,0.268])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False
            if i==1:
                landmark.state.p_pos = np.array([0.268,0.647])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False
            if i==2:
                landmark.state.p_pos = np.array([-0.268,0.647])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False   
            if i==3:
                landmark.state.p_pos = np.array([-0.647,0.268])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False
            if i==4:
                landmark.state.p_pos = np.array([-0.647,-0.268])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False
            if i==5:
                landmark.state.p_pos = np.array([-0.268,-0.647])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False
            if i==6:
                landmark.state.p_pos = np.array([0.268,-0.647])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False
            if i==7:
                landmark.state.p_pos = np.array([0.647,-0.268])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.collisionTimes = 0
                landmark.captured = False

    def benchmark_data(self, agent, world):
        return self.reward(agent, world)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def uav_reward(self, agent, world):
        for other in world.agents:
            if other.name == 'ranger':
                agent2 = other
        return self.ranger_reward(agent2, world)

    def ranger_reward(self, agent, world):
        
        for other in world.agents:
                if other.name == 'poacher':
                    agent2 = other
                    if self.is_collision(agent, agent2):
                        #print("s")
                        return 1.0
        
        for animal in world.landmarks:
            if self.is_collision(agent2, animal):
                if animal.collisionTimes >= 20:
                    animal.captured = True
                    #print("S")
                    return -5.0
        return 0

    def poacher_reward(self, agent, world):
        for other in world.agents:
                if other.name == 'ranger':
                    if self.is_collision(agent,other):
                        #print("r")
                        return -1.0

        for animal in world.landmarks:
            if self.is_collision(agent, animal):
                animal.collisionTimes = animal.collisionTimes + 1
                if animal.collisionTimes >= 20:
                    animal.captured = True
                    #print("R")
                    return 1.25
            else:
                animal.collisionTimes = 0
        return 0

    def reward(self, agent, world):
        # 3 conditions
        # Poacher - 2 conditions (ranger collision and animal collision)
        # Ranger - 1 condition (ranger collision)
        rew = 0

        if agent.name == 'poacher':
            rew = self.poacher_reward(agent, world)
        elif agent.name == 'uav':
            rew = self.uav_reward(agent, world)
        elif agent.name == 'ranger':
            rew = self.ranger_reward(agent,world)

        return rew


    def observation(self, agent, world):
        observ = [0.647,0.268,0.268,0.647,-0.268,0.647,-0.647,0.268,-0.647,-0.268,-0.268,-0.647,0.268,-0.647,0.647,-0.268]
        agent2 = None
        if agent.name == 'ranger':
            for other in world.agents:
                if other.name == 'uav':
                    agent2 = other
        if agent.name == 'uav':
            for other in world.agents:
                if other.name == 'ranger':
                    agent2 = other
        
        if agent2 != None:
            temp = np.concatenate((agent2.state.p_pos, agent2.state.p_vel), axis=None)
            temp1 = np.concatenate((temp, [agent2.id]), axis=None)
            observ = np.concatenate((observ, temp1), axis=None)
    
        if (agent.name == 'poacher'):
            for other in world.agents:
                if other is agent: continue
                delta_pos = agent.state.p_pos - other.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                if dist<= agent.sight:
                    temp = np.concatenate((other.state.p_pos, other.state.p_vel), axis=None)
                    temp1 = np.concatenate((temp, [other.id]), axis=None)
                    observ = np.concatenate((observ, temp1), axis=None)
                else:
                    observ = np.concatenate((observ, [0,0,0,0,0]), axis=None)
        
        else:
            for other in world.agents:
                if other.name == 'poacher':
                    delta_pos1 = agent.state.p_pos - other.state.p_pos
                    dist1 = np.sqrt(np.sum(np.square(delta_pos1)))
                    delta_pos2 = agent2.state.p_pos - other.state.p_pos
                    dist2 = np.sqrt(np.sum(np.square(delta_pos2)))
                    if (dist1 <= agent.sight or dist2 <= agent2.sight):
                        temp = np.concatenate((other.state.p_pos, other.state.p_vel), axis=None)
                        temp1 = np.concatenate((temp, [other.id]), axis=None)
                        observ = np.concatenate((observ, temp1), axis=None)
                    else:
                        observ = np.concatenate((observ, [0,0,0,0,0]), axis=None)

        #return len(observ)
        #print(len(observ))
        return observ


    def done(self, agent, world):
        #determine the conditions for the game to end.
        #return 1 if the game should end and 0 otherwise.

        #check collision between poacher and ranger
        if agent.name == 'poacher':
            for other in world.agents:
                if other.name == 'ranger':
                    if self.is_collision(agent, other):
                        return 1
        
        if agent.name == 'ranger':
            for other in world.agents:
                if other.name == 'poacher':
                    if self.is_collision(agent, other):
                        return 1

        if agent.name == 'uav':
            for other in world.agents:
                if other.name == 'ranger':
                    agent1 = other
                if other.name == 'poacher':
                    agent2 = other
            if self.is_collision(agent1, agent2):
                return 1
        
        #check collision with each animal:
        for animal in world.landmarks:
            if animal.captured == True:
                return 1
        
        return 0

        

