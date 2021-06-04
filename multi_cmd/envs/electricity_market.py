import cvxpy as cp
import numpy as np

np.set_printoptions(suppress=True)

def game_func_six_bus(a, b, p, d):
    ############## variable ##############
    # a: linear cost bidding of generators. Assume that renewable bids come first.
    # b: quadratic cost bidding of generators. Assume that renewable bids come first.
    # p: capacity bidding of generators. Assume that renewable bids come first.
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
    F = np.array([200, 100, 150, 100, 150, 100, 200, 100]) * 0.5
    # F = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    # Create 6 scalar optimization variables.
    p1 = cp.Variable()
    p2 = cp.Variable()
    p3 = cp.Variable()
    p4 = cp.Variable()
    p5 = cp.Variable()
    p6 = cp.Variable()

    renew1 = cp.Variable()
    renew2 = cp.Variable()
    renew3 = cp.Variable()
    renew4 = cp.Variable()
    renew5 = cp.Variable()
    renew6 = cp.Variable()

    bus1_gen = p1 + renew1
    bus2_gen = p2
    bus3_gen = p3 + renew2
    bus4_gen = p4
    bus5_gen = p5 + renew3
    bus6_gen = p5


    # Create all constraints. G[i, 5] not included b/c all zeros.
    constraints = [bus1_gen + bus2_gen + bus3_gen + bus4_gen + bus5_gen + bus6_gen == np.sum(d),
                   G[0, 0]*(bus1_gen-d[0])+G[0, 1]*(bus2_gen-d[1])+G[0, 2]*(bus3_gen-d[2])+G[0, 3]*(bus4_gen-d[3])+G[0, 4]*(bus5_gen-d[4])<=F[0],
                   G[1, 0]*(bus1_gen-d[0])+G[1, 1]*(bus2_gen-d[1])+G[1, 2]*(bus3_gen-d[2])+G[1, 3]*(bus4_gen-d[3])+G[1, 4]*(bus5_gen-d[4])<=F[1],
                   G[2, 0]*(bus1_gen-d[0])+G[2, 1]*(bus2_gen-d[1])+G[2, 2]*(bus3_gen-d[2])+G[2, 3]*(bus4_gen-d[3])+G[2, 4]*(bus5_gen-d[4])<=F[2],
                   G[3, 0]*(bus1_gen-d[0])+G[3, 1]*(bus2_gen-d[1])+G[3, 2]*(bus3_gen-d[2])+G[3, 3]*(bus4_gen-d[3])+G[3, 4]*(bus5_gen-d[4])<=F[3],
                   G[4, 0]*(bus1_gen-d[0])+G[4, 1]*(bus2_gen-d[1])+G[4, 2]*(bus3_gen-d[2])+G[4, 3]*(bus4_gen-d[3])+G[4, 4]*(bus5_gen-d[4])<=F[4],
                   G[5, 0]*(bus1_gen-d[0])+G[5, 1]*(bus2_gen-d[1])+G[5, 2]*(bus3_gen-d[2])+G[5, 3]*(bus4_gen-d[3])+G[5, 4]*(bus5_gen-d[4])<=F[5],
                   G[6, 0]*(bus1_gen-d[0])+G[6, 1]*(bus2_gen-d[1])+G[6, 2]*(bus3_gen-d[2])+G[6, 3]*(bus4_gen-d[3])+G[6, 4]*(bus5_gen-d[4])<=F[6],
                   G[7, 0]*(bus1_gen-d[0])+G[7, 1]*(bus2_gen-d[1])+G[7, 2]*(bus3_gen-d[2])+G[7, 3]*(bus4_gen-d[3])+G[7, 4]*(bus5_gen-d[4])<=F[7],
                   G[0, 0]*(bus1_gen-d[0])+G[0, 1]*(bus2_gen-d[1])+G[0, 2]*(bus3_gen-d[2])+G[0, 3]*(bus4_gen-d[3])+G[0, 4]*(bus5_gen-d[4])>=-F[0],
                   G[1, 0]*(bus1_gen-d[0])+G[1, 1]*(bus2_gen-d[1])+G[1, 2]*(bus3_gen-d[2])+G[1, 3]*(bus4_gen-d[3])+G[1, 4]*(bus5_gen-d[4])>=-F[1],
                   G[2, 0]*(bus1_gen-d[0])+G[2, 1]*(bus2_gen-d[1])+G[2, 2]*(bus3_gen-d[2])+G[2, 3]*(bus4_gen-d[3])+G[2, 4]*(bus5_gen-d[4])>=-F[2],
                   G[3, 0]*(bus1_gen-d[0])+G[3, 1]*(bus2_gen-d[1])+G[3, 2]*(bus3_gen-d[2])+G[3, 3]*(bus4_gen-d[3])+G[3, 4]*(bus5_gen-d[4])>=-F[3],
                   G[4, 0]*(bus1_gen-d[0])+G[4, 1]*(bus2_gen-d[1])+G[4, 2]*(bus3_gen-d[2])+G[4, 3]*(bus4_gen-d[3])+G[4, 4]*(bus5_gen-d[4])>=-F[4],
                   G[5, 0]*(bus1_gen-d[0])+G[5, 1]*(bus2_gen-d[1])+G[5, 2]*(bus3_gen-d[2])+G[5, 3]*(bus4_gen-d[3])+G[5, 4]*(bus5_gen-d[4])>=-F[5],
                   G[6, 0]*(bus1_gen-d[0])+G[6, 1]*(bus2_gen-d[1])+G[6, 2]*(bus3_gen-d[2])+G[6, 3]*(bus4_gen-d[3])+G[6, 4]*(bus5_gen-d[4])>=-F[6],
                   G[7, 0]*(bus1_gen-d[0])+G[7, 1]*(bus2_gen-d[1])+G[7, 2]*(bus3_gen-d[2])+G[7, 3]*(bus4_gen-d[3])+G[7, 4]*(bus5_gen-d[4])>=-F[7],
                   p1>=0,
                   p1<=p[3],
                   p2>=0,
                   p2<=p[4],
                   p3>=0,
                   p3<=p[5],
                   p4>=0,
                   p4<=p[6],
                   p5>=0,
                   p5<=p[7],
                   p6>=0,
                   p6<=p[8],
                   renew1>=0,
                   renew1<=p[0],
                   renew2>=0,
                   renew2<=p[1],
                   renew3>=0,
                   renew3<=p[2],
                   ]

    # Form objective. Renewable assumed to be zero cost.
    obj = cp.Minimize(a[3]*p1 + a[4]*p2 + a[5]*p3 + a[6]*p4 + a[7]*p5 + a[8]*p6)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()
    

    if(p1.value==None):
        return np.array([None, None, None, None, None, None, None, None, None], dtype=object), None, None
    
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
                lmp[i] = -lmp_ref - mu_upper.dot(G[:,i]) + mu_lower.dot(G[:,i])
            else:
                lmp[i] = -lmp_ref

        gen = np.array([renew1.value, renew2.value, renew3.value, p1.value, p2.value, p3.value, p4.value, p5.value, p6.value])
        # deducting generation cost perhaps
        profit = np.zeros(len(gen), dtype=float)

        profit[0] = gen[0] * (lmp[0] - a[0])
        profit[1] = gen[1] * (lmp[2] - a[1])
        profit[2] = gen[2] * (lmp[4] - a[2])

        profit[3] = gen[3] * (lmp[0] - a[3])
        profit[4] = gen[4] * (lmp[1] - a[4])
        profit[5] = gen[5] * (lmp[2] - a[5])
        profit[6] = gen[6] * (lmp[3] - a[6])
        profit[7] = gen[7] * (lmp[4] - a[7])
        profit[8] = gen[8] * (lmp[5] - a[8])
    
        print("gen:", gen)

        return gen, lmp, profit


class ProbabilisticElectricityMarket:

    def __init__(self, game_end_prob=1./24):
        print('initializing game')
        # Number of players, sixth agent is a non-renewable agent.
        self.num_players = 3

        # LMP thresholds to update load status.
        self.thresholds = np.array([25., 25., 25., 35., 30., 25.])

        # Base demand (where there are no decreases due to demand).
        self.base_demand = np.array([150., 300., 280., 250., 200., 300.])

        # Bus load state (either 0 or 1 indicating whether a bus is in critical state).
        self.reduce_factor = 0.5
        self.load_status = np.zeros(6)

        self.demand = self.base_demand - self.base_demand * self.reduce_factor * self.load_status

        self.game_end_prob = game_end_prob

    def reset(self):
        # Generate a random load state.
        self.load_status = np.random.randint(2, size=6)

        # Return load status as observation.
        return  [
            self.load_status[:6] for _ in range(self.num_players)
        ]

    def step(self, p_maxs):
        """Pmaxs only contain active agents."""
        old_pmaxs = p_maxs
        p_maxs = np.squeeze(np.clip(p_maxs, 0., 1000.))

        # Calculate new demand (w.r.t. load_status) and return profit.
        self.demand = self.base_demand - self.base_demand * self.reduce_factor * self.load_status

        # print("    input pmaxs:", old_pmaxs)
        # print("    cleaned pmaxs:", p_maxs)
        # print("    demand:", self.demand)

        # Renewable agents only have 
        gen, lmp, profit = game_func_six_bus(
            np.array([0., 0., 0.] + [35., 35., 35., 35., 35., 35.]),
            np.array([0., 0., 0.] + [0., 0., 0., 0., 0., 0.]),
            np.array(list(p_maxs) + [1000., 1000., 1000., 1000., 1000., 1000.]),
            self.demand
        )

        # TODO(anonymous): Update load status.
        self.load_status = np.greater(lmp, self.thresholds).astype(float)

        # Randomly choose to end the game or not.
        if np.random.rand() <= self.game_end_prob:
            dones = [True] * self.num_players
        else:
            dones = [False] * self.num_players

        
        # scale profit down by scalar for easier learning and inititlization?
        # profit = profit / 50.
        
        # print('    gen:', gen)
        # print('    lmp:', [lmp[0], lmp[2], lmp[4]])
        print('    reward:', profit[:3])

        return (
            [
                self.load_status[:6] for _ in range(self.num_players)
            ], 
            profit[:3],
            dones,
            (gen, lmp)
        )


if __name__ == "__main__":

    # env = ElectricityMarketV1()

    # obs1 = env.reset()

    # test_pmaxs = np.array([0., 0., 0.])

    # obs, rewards, dones, (gen, lmp) = env.step(test_pmaxs[:3])
    # print('obs:', obs)
    # print('rewards:', rewards)
    # print('gen:', gen)
    # print('lmp:', lmp)

    # test_pmaxs = np.array([400., 500., 500.])

    # obs, rewards, dones, (gen, lmp) = env.step(test_pmaxs[:3])
    # print('obs:', obs)
    # print('rewards:', rewards)
    # print('gen:', gen)
    # print('lmp:', lmp)

    p_maxs = [100., 100., 0.]
    gen, lmp, profit = game_func_six_bus(
            np.array([0., 0., 0.] + [40., 40., 40., 40., 40., 40.]),
            np.array([0., 0., 0.] + [0., 0., 0., 0., 0., 0.]),
            np.array(list(p_maxs) + [1000., 1000., 1000., 1000., 1000., 1000.]),
            np.array([150., 300., 280., 250., 200., 300.])
        )

    print("gen:", gen)
    print("lmp:", lmp)
    print("profit:", profit)


    env = ProbabilisticElectricityMarket(game_end_prob=1.)
    print(env.step(p_maxs))

    
