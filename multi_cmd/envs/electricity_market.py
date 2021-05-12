import cvxpy as cp
import numpy as np

np.set_printoptions(suppress=True)

def game_func_six_bus(a, b, p, d):
    ############## variable ##############
    # a: linear cost bidding of generators
    # b: quadratic cost bidding of generators
    # p: capacity bidding of generators
    # d: load at each bus

    # Generation shift matrix.
    G = np.array([[0.4, -0.4, -0.4, 0.2, 0.8, 0], 
        [0.2, 0.15, 0.15, -0.4, -0.4, 0],
        [0.4, 0.25, 0.25, 0.2, -0.4, 0],
        [0.25, 0.4, -0.6, 0.15, 0.5, 0],
        [0.15, 0.2, 0.2, 0.05, 0.3, 0],
        [0.25, 0.4, 0.4, 0.15, 0.5, 0],
        [0.2, 0.15, 0.15, 0.6, -0.4, 0],
        [0.6, 0.4, 0.4, 0.8, 0.2, 0]]) 

    # Line flow limit
    F = np.array([100, 200, 200, 200, 200, 300, 200, 100])

    # Create 6 scalar optimization variables.
    p1 = cp.Variable()
    p2 = cp.Variable()
    p3 = cp.Variable()
    p4 = cp.Variable()
    p5 = cp.Variable()
    p6 = cp.Variable()

    # Create all constraints.
    constraints = [p1 + p2 + p3 + p4 + p5 + p6 == np.sum(d),
                   G[0, 0]*(p1-d[0])+G[0, 1]*(p2-d[1])+G[0, 2]*(p3-d[2])+G[0, 3]*(p4-d[3])+G[0, 4]*(p5-d[5])<=F[0],
                   G[1, 0]*(p1-d[0])+G[1, 1]*(p2-d[1])+G[1, 2]*(p3-d[2])+G[1, 3]*(p4-d[3])+G[1, 4]*(p5-d[5])<=F[1],
                   G[2, 0]*(p1-d[0])+G[2, 1]*(p2-d[1])+G[2, 2]*(p3-d[2])+G[2, 3]*(p4-d[3])+G[2, 4]*(p5-d[5])<=F[2],
                   G[3, 0]*(p1-d[0])+G[3, 1]*(p2-d[1])+G[3, 2]*(p3-d[2])+G[3, 3]*(p4-d[3])+G[3, 4]*(p5-d[5])<=F[3],
                   G[4, 0]*(p1-d[0])+G[4, 1]*(p2-d[1])+G[4, 2]*(p3-d[2])+G[4, 3]*(p4-d[3])+G[4, 4]*(p5-d[5])<=F[4],
                   G[5, 0]*(p1-d[0])+G[5, 1]*(p2-d[1])+G[5, 2]*(p3-d[2])+G[5, 3]*(p4-d[3])+G[5, 4]*(p5-d[5])<=F[5],
                   G[6, 0]*(p1-d[0])+G[6, 1]*(p2-d[1])+G[6, 2]*(p3-d[2])+G[6, 3]*(p4-d[3])+G[6, 4]*(p5-d[5])<=F[6],
                   G[7, 0]*(p1-d[0])+G[7, 1]*(p2-d[1])+G[7, 2]*(p3-d[2])+G[7, 3]*(p4-d[3])+G[7, 4]*(p5-d[5])<=F[7],
                   G[0, 0]*(p1-d[0])+G[0, 1]*(p2-d[1])+G[0, 2]*(p3-d[2])+G[0, 3]*(p4-d[3])+G[0, 4]*(p5-d[5])>=-F[0],
                   G[1, 0]*(p1-d[0])+G[1, 1]*(p2-d[1])+G[1, 2]*(p3-d[2])+G[1, 3]*(p4-d[3])+G[1, 4]*(p5-d[5])>=-F[1],
                   G[2, 0]*(p1-d[0])+G[2, 1]*(p2-d[1])+G[2, 2]*(p3-d[2])+G[2, 3]*(p4-d[3])+G[2, 4]*(p5-d[5])>=-F[2],
                   G[3, 0]*(p1-d[0])+G[3, 1]*(p2-d[1])+G[3, 2]*(p3-d[2])+G[3, 3]*(p4-d[3])+G[3, 4]*(p5-d[5])>=-F[3],
                   G[4, 0]*(p1-d[0])+G[4, 1]*(p2-d[1])+G[4, 2]*(p3-d[2])+G[4, 3]*(p4-d[3])+G[4, 4]*(p5-d[5])>=-F[4],
                   G[5, 0]*(p1-d[0])+G[5, 1]*(p2-d[1])+G[5, 2]*(p3-d[2])+G[5, 3]*(p4-d[3])+G[5, 4]*(p5-d[5])>=-F[5],
                   G[6, 0]*(p1-d[0])+G[6, 1]*(p2-d[1])+G[6, 2]*(p3-d[2])+G[6, 3]*(p4-d[3])+G[6, 4]*(p5-d[5])>=-F[6],
                   G[7, 0]*(p1-d[0])+G[7, 1]*(p2-d[1])+G[7, 2]*(p3-d[2])+G[7, 3]*(p4-d[3])+G[7, 4]*(p5-d[5])>=-F[7],
                   p1>=0,
                   p1<=p[0],
                   p2>=0,
                   p2<=p[1],
                   p3>=0,
                   p3<=p[2],
                   p4>=0,
                   p4<=p[3],
                   p5>=0,
                   p5<=p[4],
                   p6>=0,
                   p6<=p[5]]

    # Form objective.
    obj = cp.Minimize(a[0]*p1 + a[1]*p2 + a[2]*p3 + a[3]*p4 + a[4]*p5 + a[5]*p6)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    if(p1.value==None):
        return np.array([None, None, None, None, None, None], dtype=object), None, None
    
    else:
        #calculate LMP
        lmp_ref = constraints[0].dual_value
        mu = []
        for k in range(16):
            mu.append(constraints[k+1].dual_value)

        # upper dual variable and lower bound variable
        mu_upper = np.array(mu[0:8])
        mu_lower = np.array(mu[8:16])

        lmp = np.zeros(6)
        for i in range(6):
            if(i<5):
                lmp[i] = -lmp_ref-mu_upper.dot(G[:,i])+mu_lower.dot(G[:,i])
            else:
                lmp[i] = -lmp_ref

        gen = np.array([p1.value, p2.value, p3.value, p4.value, p5.value, p6.value])
        # deducting generation cost perhaps
        profit = lmp * gen - a * gen
    
        return gen, lmp, profit



class ElectricityMarketV1:

    def __init__(self, trajectory_length=24):
        print('initializing game')
        # Number of players, sixth agent is a non-renewable agent.
        self.num_players = 5
        self.current_count = 0
        self.trajectory_length = trajectory_length

        # LMP thresholds to update load status.
        self.thresholds = np.array([25., 25., 25., 35., 30., 25.])

        # Base demand (where there are no decreases due to demand).
        self.base_demand = np.array([150., 300., 280., 250., 200., 300.])


        # Bus load state (either 0 or 1 indicating whether a bus is in critical state).
        self.reduce_factor = 0.05
        self.load_status = np.random.randint(2, size=6)

        self.demand = self.base_demand - self.base_demand * self.reduce_factor * self.load_status


    def reset(self):
        # Generate a random load state.
        self.load_status = np.random.randint(2, size=6)

        # Return load status as observation.
        return self.load_status

    def step(self, p_maxs):
        """Pmaxs only contain active agents, we assume sixth agent is a large non-renewable."""

        # Calculate new demand (w.r.t. load_status) and return profit.
        self.demand = self.base_demand - self.base_demand * self.reduce_factor * self.load_status

        # Renewable agents only have 
        gen, lmp, profit = game_func_six_bus(
            np.array([0., 25., 0., 35., 0., 24.]),
            np.array([0., 0., 0., 0., 0., 0.]),
            np.array([p_maxs[0], 1000., p_maxs[1], 1000., p_maxs[2], 1000.]),
            self.demand
        )

        # TODO(jjma): Update load status.
        print("gen:", gen)
        print("lmp:", lmp)
        self.load_status = np.greater(lmp, self.thresholds).astype(float)
        self.current_count += 1

        # End after trajectory length.
        if self.current_count >= self.trajectory_length:
            dones = [True] * self.num_players
        else:
            dones = [False] * self.num_players

        return [self.load_status[:5] for _ in range(self.num_players)], profit, dones, None

    

if __name__ == "__main__":
    print('Hi')

    env = ElectricityMarketV1()

    obs1 = env.reset()

    test_pmaxs = np.array([0., 0., 0.])

    obs, rewards, dones, _ = env.step(test_pmaxs[:5])
    print('obs:', obs)
    print('rewards:', rewards)

    test_pmaxs = np.array([200., 250., 250.])

    obs, rewards, dones, _ = env.step(test_pmaxs[:5])
    print('obs:', obs)

    
