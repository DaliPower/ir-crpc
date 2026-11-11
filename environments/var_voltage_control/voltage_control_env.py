from ..multiagentenv import MultiAgentEnv
import numpy as np
import pandapower as pp
from pandapower import ppException
import pandas as pd
import copy
import os
from collections import namedtuple
from .pf_res_plot import pf_res_plotly
from .voltage_barrier.voltage_barrier_backend import VoltageBarrier
from experiments.risk_aware_protocol import reward_components, squared_voltage_violation



def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class ActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high


class VoltageControl(MultiAgentEnv):
    """this class is for the environment of distributed active voltage control

        it is easy to interact with the environment, e.g.,

        state, global_state = env.reset()
        for t in range(240):
            actions = agents.get_actions(state) # a vector involving all agents' actions
            reward, done, info = env.step(actions)
            next_state = env.get_obs()
            state = next_state
    """
    def __init__(self, kwargs):
        """initialisation
        """
        # unpack args
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        self.last_powerflow_failure = None
        self.powerflow_failures = []
        self._validate_action_mapping()
        self._observation_timing()

        # set the data path
        self.data_path = args.data_path

        # set the random seed
        self.rng = np.random.default_rng(args.seed)
        
        # load the model of power network
        self.base_powergrid = self._load_network()
        
        # load data
        self.pv_data = self._load_pv_data()
        self.active_demand_data = self._load_active_demand_data()
        self.reactive_demand_data = self._load_reactive_demand_data()
        self.time_delta = int((self.pv_data.index[1] - self.pv_data.index[0]).total_seconds() // 60)
        if self.time_delta <= 0 or 60 % self.time_delta:
            raise ValueError("source interval must be a positive divisor of 60 minutes")

        # define episode and rewards
        self.episode_limit = args.episode_limit
        self.voltage_barrier_type = getattr(args, "voltage_barrier_type", "l1")
        self.voltage_weight = getattr(args, "voltage_weight", 1.0)
        self.violation_weight = getattr(args, "violation_weight", 0.0)
        self.q_weight = getattr(args, "q_weight", 0.1)
        self.line_weight = getattr(args, "line_weight", None)
        self.dv_dq_weight = getattr(args, "dq_dv_weight", None)

        # define constraints and uncertainty
        self.v_upper = getattr(args, "v_upper", 1.05)
        self.v_lower = getattr(args, "v_lower", 0.95)
        candidates = getattr(args, "start_day_candidates", None)
        self.start_day_candidates = (
            None if candidates is None else np.asarray(candidates, dtype=int)
        )
        self._fit_input_statistics()
        self._set_reactive_power_boundary()

        # define action space and observation space
        self.action_space = ActionSpace(low=-self.args.action_scale+self.args.action_bias, high=self.args.action_scale+self.args.action_bias)
        self.history = getattr(args, "history", 1)
        self.state_space = getattr(args, "state_space", ["pv", "demand", "reactive", "vm_pu", "va_degree"])
        if self.args.mode == "distributed":
            self.n_actions = 1
            self.n_agents = len(self.base_powergrid.sgen)
        elif self.args.mode == "decentralised":
            self.n_actions = len(self.base_powergrid.sgen)
            self.n_agents = len( set( self.base_powergrid.bus["zone"].to_numpy(copy=True) ) ) - 1 # exclude the main zone
        agents_obs, state = self.reset()

        self.obs_size = agents_obs[0].shape[0]
        self.state_size = state.shape[0]
        self.last_v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        self.last_q = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)

        # initialise voltage barrier function
        self.voltage_barrier = VoltageBarrier(self.voltage_barrier_type)
        self._rendering_initialized = False

    def _reset_attempt_limit(self, deterministic=False):
        limit = getattr(self.args, "max_reset_attempts", 10)
        if isinstance(limit, bool) or int(limit) != limit or limit < 1:
            raise ValueError("max_reset_attempts must be a positive integer")
        return 1 if deterministic else int(limit)

    def _record_powerflow_failure(self, error, phase):
        """Retain attempted physical inputs before any rollback or next reset.

        Only numerical load-flow nonconvergence is a physical failure. Other
        pandapower exceptions indicate a configuration/programming error and
        must propagate rather than becoming a penalized training transition.
        """
        if type(error).__name__ != 'LoadflowNotConverged':
            raise error

        def vector(table, column):
            if column not in table:
                return None
            return [float(v) if np.isfinite(v) else None
                    for v in table[column].to_numpy(dtype=float)]

        snapshot = {
            'inverter_index': self.powergrid.sgen.index.tolist(),
            'inverter_p_mw': vector(self.powergrid.sgen, 'p_mw'),
            'inverter_q_mvar': vector(self.powergrid.sgen, 'q_mvar'),
            'inverter_s_max_mva': np.asarray(self.s_max, dtype=float).tolist(),
            'load_index': self.powergrid.load.index.tolist(),
            'load_p_mw': vector(self.powergrid.load, 'p_mw'),
            'load_q_mvar': vector(self.powergrid.load, 'q_mvar'),
            'load_scaling': vector(self.powergrid.load, 'scaling'),
            'load_in_service': vector(self.powergrid.load, 'in_service'),
            'last_or_failed_bus_vm_pu': vector(self.powergrid.res_bus, 'vm_pu'),
            'action_scale': float(getattr(self.args, 'action_scale', 1.0)),
            'nighttime_reactive_power': getattr(self.args, 'nighttime_reactive_power', 'enabled'),
            'nominal_pv_mw': np.asarray(getattr(self, '_last_nominal_pv_mw', []), dtype=float).tolist(),
        }
        failure = {
            'phase': phase,
            'exception': {'type': type(error).__name__, 'module': type(error).__module__, 'message': str(error)},
            'episode_start_day': int(getattr(self, '_episode_start_day', -1)),
            'episode_start_hour': int(getattr(self, '_episode_start_hour', -1)),
            'loaded_step': int(getattr(self, '_loaded_step', self.steps)),
            'reset_attempt': int(getattr(self, '_last_reset_attempts', 0)),
            'snapshot': snapshot,
            'voltage_results_are_valid': False,
        }
        if hasattr(self, '_episode_start_index'):
            failure['timestamp'] = self.current_timestamp().isoformat()
        self.last_powerflow_failure = failure
        if not hasattr(self, 'powerflow_failures'):
            self.powerflow_failures = []
        self.powerflow_failures.append(failure)

    def reset(self, reset_time=True):
        """reset the env
        """
        # reset the time step, cumulative rewards and obs history
        self.steps = 0
        self.sum_rewards = 0
        self._last_action_diagnostics = None
        self._last_hold_q_projection_ratio = 0.0
        self._last_hold_q_adjustment_mvar = None
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)
        solvable = False
        attempt_limit = self._reset_attempt_limit()
        self._last_reset_attempts = 0
        while not solvable:
            self._last_reset_attempts += 1
            # reset the time stamp
            if reset_time:
                self._episode_start_hour = self._select_start_hour()
                self._episode_start_interval = self._select_start_interval()
                self._episode_start_day = self._select_start_day()
            # get one episode of data
            self._episode_start_index = self._resolve_episode_start_index()
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._loaded_step = 0
            self._set_demand_and_pv()
            # random initialise action
            if self.args.reset_action:
                self.powergrid.sgen["q_mvar"] = self.get_action()
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            else:
                self.powergrid.sgen["q_mvar"] = 0.0
            try:    
                pp.runpp(self.powergrid)
                self._pending_measurements = False
                solvable = True
            except ppException as error:
                self._record_powerflow_failure(error, 'random_reset')
                if self._last_reset_attempts >= attempt_limit:
                    raise
                print ("The power flow for the initialisation of demand and PV cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False

        return self.get_obs(), self.get_state()
    
    def manual_reset(self, day, hour, interval):
        """manual reset the initial date
        """
        # reset the time step, cumulative rewards and obs history
        self.steps = 0
        self.sum_rewards = 0
        self._last_action_diagnostics = None
        self._last_hold_q_projection_ratio = 0.0
        self._last_hold_q_adjustment_mvar = None
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)

        # reset the time stamp
        self._episode_start_hour = hour
        self._episode_start_day = day
        self._episode_start_interval = interval
        solvable = False
        attempt_limit = self._reset_attempt_limit(deterministic=not self.args.reset_action)
        self._last_reset_attempts = 0
        while not solvable:
            self._last_reset_attempts += 1
            # get one episode of data
            self._episode_start_index = self._resolve_episode_start_index()
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._loaded_step = 0
            self._set_demand_and_pv(add_noise=False)
            # random initialise action
            if self.args.reset_action:
                self.powergrid.sgen["q_mvar"] = self.get_action()
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            else:
                self.powergrid.sgen["q_mvar"] = 0.0
            try:    
                pp.runpp(self.powergrid)
                self._pending_measurements = False
                solvable = True
            except ppException as error:
                self._record_powerflow_failure(error, 'manual_reset')
                if self._last_reset_attempts >= attempt_limit:
                    raise
                print ("The power flow for the initialisation of demand and PV cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False

        return self.get_obs(), self.get_state()

    def step(self, actions, add_noise=True):
        """function for the interaction between agent and the env each time step
        """
        last_powergrid = copy.deepcopy(self.powergrid)

        # check whether the power balance is unsolvable
        solvable = self._take_action(actions)
        if solvable:
            # get the reward of current actions
            reward, info = self._calc_reward()
        else:
            q_loss = np.mean( np.abs(self.powergrid.sgen["q_mvar"]) )
            self.powergrid = last_powergrid
            reward, info = self._calc_reward()
            reward -= 200.
            # keep q_loss
            info["destroy"] = 1.
            info["totally_controllable_ratio"] = 0.
            info["q_loss"] = q_loss

        # terminate if episode_limit is reached
        self.steps += 1
        self.sum_rewards += reward
        terminated = self.steps >= self.episode_limit or not solvable
        if not terminated:
            # set the pv and demand for the next time step
            self._set_demand_and_pv(add_noise=add_noise)
            self._loaded_step = self.steps
        if terminated:
            print (f"Episode terminated at time: {self.steps} with return: {self.sum_rewards:2.4f}.")

        return reward, terminated, info

    def _observation_timing(self):
        timing = getattr(getattr(self, "args", None), "observation_timing", "legacy")
        if timing not in {"legacy", "current_pre_control"}:
            raise ValueError("observation_timing must be legacy or current_pre_control")
        return timing

    def _ensure_current_measurements(self):
        """Lazily measure the newly loaded source/load snapshot under held Q.

        step() leaves result tables at the just-controlled snapshot so evaluators
        can read it. The next observation request advances the measurements.
        Pre-control power-flow failures propagate to the caller; no stale values
        are silently substituted, and the pending flag remains set on failure.
        """
        if self._observation_timing() != "current_pre_control" or not getattr(self, "_pending_measurements", False):
            return
        previous = self.powergrid.sgen["q_mvar"].to_numpy(dtype=float, copy=True)
        limit = float(self.args.action_scale) * self.reactive_power_capacity()
        held = np.clip(previous, -limit, limit)
        self._last_hold_q_adjustment_mvar = held - previous
        self._last_hold_q_projection_ratio = float(np.mean(np.abs(held - previous) > 1e-10)) if held.size else 0.0
        self.powergrid.sgen["q_mvar"] = held
        try:
            pp.runpp(self.powergrid)
        except ppException as error:
            self._record_powerflow_failure(error, 'pre_control_measurement')
            raise
        self._pending_measurements = False

    def _bus_loads(self, bus_index):
        """Metered load demand by bus, excluding generators and other elements."""
        load = self.powergrid.load
        demand = load[["bus", "p_mw", "q_mvar"]].copy()
        scale = load["scaling"] if "scaling" in load else 1.0
        if "in_service" in load:
            scale = scale * load["in_service"].astype(float)
        demand[["p_mw", "q_mvar"]] = demand[["p_mw", "q_mvar"]].mul(scale, axis=0)
        return demand.groupby("bus")[["p_mw", "q_mvar"]].sum().reindex(bus_index, fill_value=0.0)

    def get_state(self):
        """return the global state for the power system
           the default state: voltage, active power of generators, bus state, load active power, load reactive power
        """
        self._ensure_current_measurements()
        state = []
        if "demand" in self.state_space:
            demand = (self._bus_loads(self.powergrid.res_bus.sort_index().index)
                      if self._observation_timing() == "current_pre_control"
                      else self.powergrid.res_bus.sort_index())
            state += list(demand["p_mw"].to_numpy(copy=True))
            state += list(demand["q_mvar"].to_numpy(copy=True))
        if "pv" in self.state_space:
            state += list(self.powergrid.sgen["p_mw"].sort_index().to_numpy(copy=True))
        if "reactive" in self.state_space:
            state += list(self.powergrid.sgen["q_mvar"].sort_index().to_numpy(copy=True))
        if "vm_pu" in self.state_space:
            state += list(self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True))
        if "va_degree" in self.state_space:
            state += list(self.powergrid.res_bus["va_degree"].sort_index().to_numpy(copy=True))
        state = np.array(state)
        return state
    
    def get_obs(self):
        """return the obs for each agent in the power system
           the default obs: voltage, active power of generators, bus state, load active power, load reactive power
           each agent can only observe the state within the zone where it belongs
        """
        self._ensure_current_measurements()
        clusters = self._get_clusters_info()
        if self.args.mode == "distributed":
            if self._observation_timing() == "legacy":
                # only build sgen-indexed clusters in distributed mode
                for i in range(len(self.powergrid.sgen)):
                    clusters[f"sgen{i}"][0].loc[clusters[f"sgen{i}"][4], ["p_mw"]] += clusters[f"sgen{i}"][2]
                    clusters[f"sgen{i}"][0].loc[clusters[f"sgen{i}"][4], ["q_mvar"]] += clusters[f"sgen{i}"][3]
                    for j in range(len(self.powergrid.sgen)):
                        if i != j and clusters[f"sgen{j}"][1] == clusters[f"sgen{i}"][1]:
                            clusters[f"sgen{j}"][0].loc[clusters[f"sgen{i}"][4], ["p_mw"]] += clusters[f"sgen{i}"][2]
                            clusters[f"sgen{j}"][0].loc[clusters[f"sgen{i}"][4], ["q_mvar"]] += clusters[f"sgen{i}"][3]
            obs_sgen_dict = dict()
            sgen_list = list()
            obs_len_list = list()
            for i in range(len(self.powergrid.sgen)):
                obs = list()
                zone_buses, zone, pv, q, sgen_bus = clusters[f"sgen{i}"]
                sgen_list.append(f"'{list(clusters.keys())[i]}'")
                if not( f"'{list(clusters.keys())[i]}'" in obs_sgen_dict.keys() ):
                    if "demand" in self.state_space:
                        copy_zone_buses = copy.deepcopy(zone_buses)
                        obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                        obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                    if "pv" in self.state_space:
                        obs.append(pv)
                    if "reactive" in self.state_space:
                        obs.append(q)
                    if "vm_pu" in self.state_space:
                        obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                    if "va_degree" in self.state_space:
                        # transform the voltage phase to radian
                        obs += list(zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180)
                    obs_sgen_dict[f"'{list(clusters.keys())[i]}'"] = np.array(obs)
                obs_len_list.append(obs_sgen_dict[f"'{list(clusters.keys())[i]}'"].shape[0])
            agents_obs = list()
            obs_max_len = max(obs_len_list)
            for sgen in sgen_list:
                obs_sgen = obs_sgen_dict[sgen]
                pad_obs_sgen = np.concatenate( [obs_sgen, np.zeros(obs_max_len - obs_sgen.shape[0])], axis=0 )
                agents_obs.append(pad_obs_sgen)
        elif self.args.mode == "decentralised":
            obs_len_list = list()
            zone_obs_list = list()
            for i in range(self.n_agents):
                zone_buses, pv, q, sgen_buses = clusters[f"zone{i+1}"]
                obs = list()
                if "demand" in self.state_space:
                    copy_zone_buses = copy.deepcopy(zone_buses)
                    copy_zone_buses.loc[sgen_buses]["p_mw"] += pv
                    copy_zone_buses.loc[sgen_buses]["q_mvar"] += q
                    obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                    obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                if "pv" in self.state_space:
                    obs += list(pv.to_numpy(copy=True))
                if "reactive" in self.state_space:
                    obs += list(q.to_numpy(copy=True))
                if "vm_pu" in self.state_space:
                    obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                if "va_degree" in self.state_space:
                    obs += list(zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180)
                obs = np.array(obs)
                zone_obs_list.append(obs)
                obs_len_list.append(obs.shape[0])
            agents_obs = []
            obs_max_len = max(obs_len_list)
            for obs_zone in zone_obs_list:
                pad_obs_zone = np.concatenate( [obs_zone, np.zeros(obs_max_len - obs_zone.shape[0])], axis=0 )
                agents_obs.append(pad_obs_zone)
        else:
            raise RuntimeError(f"Unsupported mode: {self.args.mode}")
        if self.history > 1:
            agents_obs_ = []
            for i, obs in enumerate(agents_obs):
                if len(self.obs_history[i]) >= self.history - 1:
                    obs_ = np.concatenate(self.obs_history[i][-self.history+1:]+[obs], axis=0)
                else:
                    zeros = [np.zeros_like(obs)] * ( self.history - len(self.obs_history[i]) - 1 )
                    obs_ = self.obs_history[i] + [obs]
                    obs_ = zeros + obs_
                    obs_ = np.concatenate(obs_, axis=0)
                agents_obs_.append(copy.deepcopy(obs_))
                self.obs_history[i].append(copy.deepcopy(obs))
            agents_obs = agents_obs_
        return agents_obs

    def get_obs_agent(self, agent_id):
        """return observation for agent_id 
        """
        agents_obs = self.get_obs()
        return agents_obs[agent_id]
    
    def get_obs_size(self):
        """return the observation size
        """
        return self.obs_size

    def get_state_size(self):
        """return the state size
        """
        return self.state_size

    def get_action(self):
        """return the action according to a uniform distribution over [action_lower, action_upper)
        """
        rand_action = self.rng.uniform(low=self.action_space.low, high=self.action_space.high, size=self.powergrid.sgen["q_mvar"].values.shape)
        return rand_action

    def get_total_actions(self):
        """return the total number of actions an agent could ever take 
        """
        return self.n_actions

    def get_avail_actions(self):
        """return available actions for all agents
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """ return the available actions for agent_id 
        """
        if self.args.mode == "distributed":
            return [1]
        elif self.args.mode == "decentralised":
            avail_actions = np.zeros(self.n_actions)
            zone_sgens = self.base_powergrid.sgen.loc[self.base_powergrid.sgen["name"] == f"zone{agent_id+1}"]
            avail_actions[zone_sgens.index] = 1
            return avail_actions

    def get_num_of_agents(self):
        """return the number of agents
        """
        return self.n_agents

    def _get_voltage(self):
        return self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)

    def _create_basenet(self, base_net):
        """initilization of power grid
        set the pandapower net to use
        """
        if base_net is None:
            raise Exception("Please provide a base_net configured as pandapower format.")
        else:
            return base_net

    def _select_start_hour(self):
        value = getattr(self.args, "start_hour", None)
        if value is None:
            return int(self.rng.integers(24))
        if int(value) != value or not 0 <= value < 24:
            raise ValueError("start_hour must be an integer in [0, 23]")
        return int(value)

    def _select_start_interval(self):
        value = getattr(self.args, "start_minute", None)
        if value is None:
            return int(self.rng.integers(60 // self.time_delta))
        if int(value) != value or not 0 <= value < 60 or value % self.time_delta:
            raise ValueError("start_minute must align with the source interval in [0, 59]")
        return int(value // self.time_delta)

    def _select_start_day(self):
        """Sample starts whose entire controlled trajectory fits in the source."""
        index = self.pv_data.index
        day_zero = index[0].normalize()
        days = int((index[-1].normalize() - day_zero).days) + 1
        candidates = getattr(self, "start_day_candidates", None)
        valid = np.arange(days) if candidates is None else np.asarray(candidates, dtype=int)
        if not hasattr(self, "time_delta"):
            self.time_delta = int((index[1] - index[0]).total_seconds() // 60)
        if not hasattr(self, "rng"):
            self.rng = np.random.default_rng(0)
        offset = pd.Timedelta(hours=getattr(self, "_episode_start_hour", 0),
                              minutes=getattr(self, "_episode_start_interval", 0) * self.time_delta)
        duration = pd.Timedelta(minutes=(self.episode_limit - 1) * self.time_delta)
        targets = day_zero + pd.to_timedelta(valid, unit="D") + offset
        valid = valid[(valid >= 0) & (targets >= index[0]) & (targets + duration <= index[-1])]
        if not valid.size:
            raise ValueError("start_day_candidates contains no valid complete episode start day")
        return int(self.rng.choice(valid))

    def _load_network(self):
        """load network
        """
        network_path = os.path.join(self.data_path, 'model.p')
        base_net = pp.from_pickle(network_path)
        return self._create_basenet(base_net)

    def _load_pv_data(self):
        """load pv data
        the sensor frequency is set to 3 or 15 mins as default
        """
        pv_path = os.path.join(self.data_path, 'pv_active.csv')
        pv = pd.read_csv(pv_path, index_col=None)
        pv.index = pd.to_datetime(pv.iloc[:, 0])
        pv.index.name = 'time'
        pv = pv.iloc[::1, 1:] * self.args.pv_scale
        return pv

    def _load_active_demand_data(self):
        """load active demand data
        the sensor frequency is set to 3 or 15 mins as default
        """
        demand_path = os.path.join(self.data_path, 'load_active.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand
    
    def _load_reactive_demand_data(self):
        """load reactive demand data
        the sensor frequency is set to 3 min as default
        """
        demand_path = os.path.join(self.data_path, 'load_reactive.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand

    def current_timestamp(self):
        return self.pv_data.index[
            self._episode_start_index + self._loaded_step
        ]

    def _resolve_episode_start_index(self):
        pv_index = self.pv_data.index
        if (
            not pv_index.equals(self.active_demand_data.index)
            or not pv_index.equals(self.reactive_demand_data.index)
        ):
            raise ValueError(
                "PV, active-demand, and reactive-demand indexes must match"
            )

        day_zero = pv_index[0].normalize()
        target = day_zero + pd.Timedelta(
            days=int(self._episode_start_day),
            hours=int(self._episode_start_hour),
            minutes=int(self._episode_start_interval) * self.time_delta,
        )
        location = pv_index.get_indexer([target])[0]
        strict = getattr(getattr(self, "args", None), "strict_episode_timestamps", False)
        if location >= 0:
            if strict or getattr(self, "episode_limit", 0) * self.time_delta == 24 * 60:
                actual = pv_index[location:location + self.episode_limit]
                expected = pd.date_range(target, periods=self.episode_limit,
                                         freq=pd.Timedelta(minutes=self.time_delta))
                if not actual.equals(expected):
                    raise ValueError("episode must contain a complete contiguous source trajectory")
            return int(location)

        if target == day_zero and not strict and getattr(self, "episode_limit", 0) * self.time_delta != 24 * 60:
            location = int(pv_index.searchsorted(target, side="left"))
            if location < len(pv_index):
                return location

        raise ValueError(f"episode start timestamp {target} is missing from source data")

    def _get_episode_pv_history(self):
        """return the pv history in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_index
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_pv_history = self.pv_data[start:start + nr_intervals].values
        return episode_pv_history
    
    def _get_episode_active_demand_history(self):
        """return the active power histories for all loads in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_index
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_demand_history = self.active_demand_data[start:start + nr_intervals].values
        return episode_demand_history
    
    def _get_episode_reactive_demand_history(self):
        """return the reactive power histories for all loads in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_index
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_demand_history = self.reactive_demand_data[start:start + nr_intervals].values
        return episode_demand_history

    def _get_pv_history(self):
        """returns pv history
        """
        t = self.steps
        history = self.history
        return self.pv_histories[t:t+history, :]

    def _get_active_demand_history(self):
        """return the history demand
        """
        t = self.steps
        history = self.history
        return self.active_demand_histories[t:t+history, :]
    
    def _get_reactive_demand_history(self):
        """return the history demand
        """
        t = self.steps
        history = self.history
        return self.reactive_demand_histories[t:t+history, :]

    def _set_demand_and_pv(self, add_noise=True):
        """optionally update the demand and pv production according to the histories with some i.i.d. gaussian noise
        """ 
        pv = copy.copy(self._get_pv_history()[0, :])
        self._last_nominal_pv_mw = pv.copy()

        # add uncertainty to pv data with unit truncated gaussian (only positive accepted)
        if add_noise:
            pv += self.pv_std * np.abs(self.rng.standard_normal(pv.shape))
        active_demand = copy.copy(self._get_active_demand_history()[0, :])

        # add uncertainty to active power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            active_demand += self.active_demand_std * np.abs(self.rng.standard_normal(active_demand.shape))
        reactive_demand = copy.copy(self._get_reactive_demand_history()[0, :])

        # add uncertainty to reactive power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            reactive_demand += self.reactive_demand_std * np.abs(self.rng.standard_normal(reactive_demand.shape))

        # update the record in the pandapower
        self._last_active_power_requested_mw = pv.copy()
        pv = self._apply_active_power_limit(pv)
        self._last_active_power_curtailment_mw = self._last_active_power_requested_mw - pv
        self.powergrid.sgen["p_mw"] = pv
        self.powergrid.load["p_mw"] = active_demand
        self.powergrid.load["q_mvar"] = reactive_demand
        self._pending_measurements = True

    def _statistics_frame(self, data):
        end_day = getattr(self.args, "statistics_end_day_exclusive", None)
        if end_day is None:
            return data
        if int(end_day) != end_day or end_day <= 0:
            raise ValueError("statistics_end_day_exclusive must be a positive integer")
        cutoff = self.pv_data.index[0].normalize() + pd.Timedelta(days=int(end_day))
        fitted = data.loc[data.index < cutoff]
        if fitted.empty:
            raise ValueError("statistics window contains no source samples")
        return fitted

    def _fit_input_statistics(self):
        """Fit physical-input noise amplitudes only on the declared prefix.

        Perturbations retain the historical positive half-normal distribution;
        these are source/load perturbations, not measurement errors.
        """
        self.active_demand_std = self._statistics_frame(self.active_demand_data).values.std(axis=0) / 100.0
        self.reactive_demand_std = self._statistics_frame(self.reactive_demand_data).values.std(axis=0) / 100.0
        self.pv_std = self._statistics_frame(self.pv_data).values.std(axis=0) / 100.0

    def _set_reactive_power_boundary(self):
        """Freeze physical ratings; reference kappa is independent of operating kappa."""
        self.factor = 1.2
        given = getattr(self.args, "inverter_s_max_mva", None)
        reference = getattr(self.args, "capacity_reference_pv_scale", None)
        fit = self._statistics_frame(self.pv_data)
        if given is not None:
            self.s_max = np.asarray(given, dtype=float)
            self.p_max = None
            self._capacity_source = getattr(self.args, "capacity_source", "given_ratings")
        else:
            self.p_max = fit.to_numpy().max(axis=0)
            if reference is not None:
                if not np.isfinite(reference) or reference <= 0 or self.args.pv_scale <= 0:
                    raise ValueError("capacity reference and operating pv_scale must be positive")
                self.p_max = self.p_max * (float(reference) / self.args.pv_scale)
            self.s_max = self.factor * self.p_max
            self._capacity_source = ("synthetic_training_peak" if
                getattr(self.args, "statistics_end_day_exclusive", None) is not None
                else "legacy_full_series_peak")
        if (self.s_max.shape != (len(self.base_powergrid.sgen),) or
                not np.all(np.isfinite(self.s_max)) or np.any(self.s_max < 0)):
            raise ValueError("inverter_s_max_mva must contain one finite non-negative rating per inverter")
        if getattr(self.args, "active_power_overflow", "reject") not in {"reject", "curtail"}:
            raise ValueError("active_power_overflow must be reject or curtail")
        if getattr(self.args, "nighttime_reactive_power", "enabled") not in {"enabled", "disabled"}:
            raise ValueError("nighttime_reactive_power must be enabled or disabled")

    def _apply_active_power_limit(self, active_power):
        active = np.asarray(active_power, dtype=float)
        if not np.all(np.isfinite(active)) or np.any(active < 0):
            raise ValueError("PV active power must be finite and non-negative")
        if np.any(active > self.s_max):
            if getattr(self.args, "active_power_overflow", "reject") == "curtail":
                return np.minimum(active, self.s_max)
            raise ValueError("PV active power exceeds inverter rating; select explicit curtailment or restrict the scenario")
        return active

    def reactive_power_capacity(self, active_power=None):
        """Physical Q capability; direct out-of-domain P is always rejected.

        Curtailment is applied when exogenous P is loaded, never hidden during
        action mapping. External users must supply the resulting feasible P.
        """
        if active_power is None:
            active_power = self.powergrid.sgen["p_mw"].to_numpy(dtype=float)
        active = np.asarray(active_power, dtype=float)
        if not np.all(np.isfinite(active)) or np.any(active < 0):
            raise ValueError("PV active power must be finite and non-negative")
        if np.any(active > self.s_max):
            raise ValueError("PV active power exceeds inverter rating before reactive action mapping")
        with np.errstate(over="raise", invalid="raise"):
            capacity = np.sqrt(self.s_max**2 - active**2)
        if getattr(self.args, "nighttime_reactive_power", "enabled") == "disabled":
            nominal = getattr(self, "_last_nominal_pv_mw", active)
            capacity = np.where((active > 1e-10) & (nominal > 1e-10), capacity, 0.0)
        return capacity

    def environment_metadata(self):
        fit = self._statistics_frame(self.pv_data)
        metadata = {
            "rng": {"type": type(self.rng.bit_generator).__name__, "seed": int(self.args.seed)},
            "observation": {
                "timing": self._observation_timing(),
                "demand_source": "bus_aggregated_in_service_scaled_load" if self._observation_timing() == "current_pre_control" else "legacy_res_bus_plus_generator_reconstruction",
                "held_q_rule": "clip_physical_q_to_current_alpha_capacity_before_precontrol_power_flow" if self._observation_timing() == "current_pre_control" else "legacy_no_precontrol_power_flow",
                "precontrol_power_flow": "lazy_on_get_obs_or_get_state" if self._observation_timing() == "current_pre_control" else None,
            },
            "reset": {"max_random_attempts": self._reset_attempt_limit(),
                      "deterministic_manual_attempts": 1,
                      "last_attempts": getattr(self, "_last_reset_attempts", None)},
            "statistics": {
                "end_day_exclusive": getattr(self.args, "statistics_end_day_exclusive", None),
                "source": "training_prefix" if getattr(self.args, "statistics_end_day_exclusive", None) is not None else "legacy_full_series",
                "first_timestamp": fit.index[0].isoformat(), "last_timestamp": fit.index[-1].isoformat(),
                "samples": len(fit), "noise_distribution": "positive_half_normal", "noise_std_divisor": 100.0,
                "pv_std_mw": self.pv_std.tolist(), "active_demand_std_mw": self.active_demand_std.tolist(),
                "reactive_demand_std_mvar": self.reactive_demand_std.tolist(),
            },
            "capacity": {
                "source": self._capacity_source, "s_max_mva": self.s_max.tolist(),
                "reference_pv_scale": getattr(self.args, "capacity_reference_pv_scale", None),
                "depends_on_operating_pv_scale": getattr(self.args, "inverter_s_max_mva", None) is None and getattr(self.args, "capacity_reference_pv_scale", None) is None,
                "synthetic_peak_multiplier": self.factor if self.p_max is not None else None,
                "active_power_overflow": getattr(self.args, "active_power_overflow", "reject"),
                "nighttime_reactive_power": getattr(self.args, "nighttime_reactive_power", "enabled"),
                "nighttime_zero_threshold_mw": 1e-10,
                "nighttime_detection": "nominal_pv_before_source_noise_or_zero_actual_pv",
            },
        }
        if hasattr(self, "_episode_start_index"):
            start = self.pv_data.index[self._episode_start_index]
            metadata["episode"] = {
                "start_timestamp": start.isoformat(),
                "end_timestamp": (start + pd.Timedelta(minutes=(self.episode_limit - 1) * self.time_delta)).isoformat(),
                "controlled_snapshots": int(self.episode_limit), "interval_minutes": self.time_delta,
                "configured_start_hour": getattr(self.args, "start_hour", None),
                "configured_start_minute": getattr(self.args, "start_minute", None),
            }
        return metadata

    def _get_clusters_info(self):
        """return the clusters of info
        the clusters info is divided by predefined zone
        distributed: each zone is equipped with several PV generators and each PV generator is an agent
        decentralised: each zone is controlled by an agent and each agent may have variant number of actions
        """
        clusters = dict()
        if self.args.mode == "distributed":
            for i in range(len(self.powergrid.sgen)):
                zone = self.powergrid.sgen["name"][i]
                sgen_bus = self.powergrid.sgen["bus"][i]
                pv = self.powergrid.sgen["p_mw"][i]
                q = self.powergrid.sgen["q_mvar"][i]
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"] == zone]
                clusters[f"sgen{i}"] = (zone_res_buses, zone, pv, q, sgen_bus)
        elif self.args.mode == "decentralised":
            for i in range(self.n_agents):
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"] == f"zone{i+1}"]
                sgen_res_buses = self.powergrid.sgen["bus"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                pv = self.powergrid.sgen["p_mw"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                q = self.powergrid.sgen["q_mvar"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                clusters[f"zone{i+1}"] = (zone_res_buses, pv, q, sgen_res_buses)

        if self._observation_timing() == "current_pre_control":
            for key, cluster in clusters.items():
                buses = cluster[0].copy()
                buses[["p_mw", "q_mvar"]] = self._bus_loads(buses.index)
                clusters[key] = (buses,) + cluster[1:]
        return clusters
    
    def _take_action(self, actions):
        """take the control variables
        the control variables we consider are the exact reactive power
        of each distributed generator
        """
        reactive_power, diagnostics = self._map_reactive_action(actions)
        self.powergrid.sgen["q_mvar"] = reactive_power
        self._last_action_diagnostics = diagnostics

        # solve power flow to get the latest voltage with new reactive power and old deamnd and PV active power
        try:
            pp.runpp(self.powergrid)
            self._pending_measurements = False
            return True
        except ppException as error:
            self._record_powerflow_failure(error, 'controlled_action')
            print ("The power flow for the reactive power penetration cannot be solved.")
            print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
            print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
            print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
            print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
            print (f"This is the res_bus: \n{self.powergrid.res_bus}")
            return False

    def _validate_action_mapping(self):
        mapping = getattr(self.args, "action_mapping", "capacity_adaptive")
        if mapping not in {"capacity_adaptive", "static_projected"}:
            raise ValueError(f"unsupported action_mapping: {mapping}")
        return mapping

    def _map_reactive_action(self, actions):
        """Map controlled actions to feasible inverter reactive power."""
        active_power = self.powergrid.sgen["p_mw"].to_numpy(dtype=float)
        if active_power.ndim != 1 or not np.all(np.isfinite(active_power)):
            raise ValueError("p_mw must be a finite one-dimensional vector")

        inverter_count = active_power.size
        s_max = np.asarray(self.s_max, dtype=float)
        if s_max.ndim == 0 and inverter_count == 1:
            s_max = s_max.reshape(1)
        if s_max.ndim != 1 or s_max.size != inverter_count:
            raise ValueError("s_max length must match the number of inverters")
        if not np.all(np.isfinite(s_max)) or np.any(s_max < 0.0):
            raise ValueError("s_max must contain finite non-negative values")

        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 0 and inverter_count == 1:
            actions = actions.reshape(1)
        if actions.ndim != 1 or actions.size != inverter_count:
            raise ValueError(
                "actions must be a one-dimensional vector with one value per inverter"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must contain only finite values")

        try:
            with np.errstate(over="raise", invalid="raise"):
                capacity = self.reactive_power_capacity(active_power)
        except FloatingPointError as error:
            raise ValueError("inverter capacity calculation must remain finite") from error

        mapping = self._validate_action_mapping()
        if mapping == "capacity_adaptive":
            candidate = capacity * actions
        else:
            candidate = s_max * actions
        if not np.all(np.isfinite(candidate)):
            raise ValueError("mapped reactive-power candidates must remain finite")

        alpha = float(getattr(self.args, "action_scale", 1.0))
        if not 0 <= alpha <= 1:
            raise ValueError("action_scale must lie in [0, 1] for physical action mapping")
        controlled_capacity = alpha * capacity
        executed = np.clip(candidate, -controlled_capacity, controlled_capacity)
        tolerance = 1e-10
        projected = np.abs(candidate - executed) > tolerance
        has_capacity = capacity > tolerance
        utilization = np.divide(
            np.abs(executed),
            capacity,
            out=np.zeros_like(capacity, dtype=float),
            where=has_capacity,
        )
        saturated = has_capacity & (
            np.abs(executed) >= capacity - tolerance
        )

        def mean_or_zero(values):
            return float(np.mean(values)) if values.size else 0.0

        has_control_limit = controlled_capacity > tolerance
        control_utilization = np.divide(
            np.abs(executed), controlled_capacity,
            out=np.zeros_like(controlled_capacity), where=has_control_limit,
        )
        control_saturated = has_control_limit & (np.abs(executed) >= controlled_capacity - tolerance)
        diagnostics = {
            "control_limit_utilization": mean_or_zero(control_utilization),
            "control_limit_saturation_ratio": mean_or_zero(control_saturated),
            "action_projection_ratio": mean_or_zero(projected),
            "reactive_capacity_utilization": mean_or_zero(utilization),
            "reactive_saturation_ratio": mean_or_zero(saturated),
        }
        return executed, diagnostics
    
    def _clip_reactive_power(self, reactive_actions, active_power):
        """clip the reactive power to the hard safety range
        """
        active_power = np.asarray(active_power, dtype=float)
        capacity = self.reactive_power_capacity(active_power)
        candidate = capacity * np.asarray(reactive_actions, dtype=float)
        alpha = float(getattr(self.args, "action_scale", 1.0))
        return np.clip(candidate, -alpha * capacity, alpha * capacity)
    
    def _calc_reward(self, info=None):
        """reward function
        consider 5 possible choices on voltage barrier functions:
            l1
            l2
            courant_beltrami
            bowl
            bump
        """
        if info is None:
            info = {}
        diagnostics = getattr(self, "_last_action_diagnostics", None)
        if diagnostics is not None:
            info.update(diagnostics)

        if hasattr(self.powergrid, "sgen") and hasattr(self, "s_max"):
            info["inverter_p_mw"] = self.powergrid.sgen["p_mw"].to_numpy(
                dtype=float, copy=True
            )
            info["inverter_q_mvar"] = self.powergrid.sgen["q_mvar"].to_numpy(
                dtype=float, copy=True
            )
            info["inverter_s_max_mva"] = np.asarray(
                self.s_max, dtype=float
            ).copy()

        info["hold_q_projection_ratio"] = float(getattr(self, "_last_hold_q_projection_ratio", 0.0))
        hold_adjustment = getattr(self, "_last_hold_q_adjustment_mvar", None)
        if hold_adjustment is not None:
            info["hold_q_adjustment_mvar"] = hold_adjustment.copy()
        curtailed = getattr(self, "_last_active_power_curtailment_mw", None)
        if curtailed is not None:
            info["inverter_active_power_curtailment_mw"] = curtailed.copy()
            info["active_power_curtailment_mw"] = float(np.sum(curtailed))
            info["active_power_curtailment_mwh"] = float(np.sum(curtailed)) * self.time_delta / 60.0

        # percentage of voltage out of control
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        percent_of_v_out_of_control = ( np.sum(v < self.v_lower) + np.sum(v > self.v_upper) ) / v.shape[0]
        info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control
        info["percentage_of_lower_than_lower_v"] = np.sum(v < self.v_lower) / v.shape[0]
        info["percentage_of_higher_than_upper_v"] = np.sum(v > self.v_upper) / v.shape[0]
        info["totally_controllable_ratio"] = 0. if percent_of_v_out_of_control > 1e-3 else 1.

        # voltage violation
        v_ref = 0.5 * (self.v_lower + self.v_upper)
        info["average_voltage_deviation"] = np.mean( np.abs( v - v_ref ) )
        info["average_voltage"] = np.mean(v)
        info["max_voltage_drop_deviation"] = np.max( (v < self.v_lower) * (self.v_lower - v) )
        info["max_voltage_rise_deviation"] = np.max( (v > self.v_upper) * (v - self.v_upper) )

        # line loss
        line_loss = np.sum(self.powergrid.res_line["pl_mw"])
        avg_line_loss = np.mean(self.powergrid.res_line["pl_mw"])
        info["total_line_loss"] = line_loss

        # reactive power (q) loss
        q = self.powergrid.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        q_loss = np.mean(np.abs(q))
        info["q_loss"] = q_loss

        # reward function
        if self.q_weight != None:
            components = reward_components(
                v,
                q,
                v_lower=self.v_lower,
                v_upper=self.v_upper,
                voltage_weight=self.voltage_weight,
                violation_weight=self.violation_weight,
                q_weight=self.q_weight,
            )
            info.update(components)
            loss = components["total_loss"]
        elif self.line_weight != None:
            v_loss = np.mean(self.voltage_barrier.step(v)) * self.voltage_weight
            violation_loss = (
                np.mean(squared_voltage_violation(v, self.v_lower, self.v_upper))
                * self.violation_weight
            )
            info["voltage_deviation_loss"] = v_loss
            info["voltage_violation_loss"] = violation_loss
            info["q_loss_weighted"] = 0.0
            loss = avg_line_loss * self.line_weight + v_loss + violation_loss
            info["total_loss"] = loss
            info["reward"] = -loss
        else:
            raise NotImplementedError("Please at least give one weight, either q_weight or line_weight.")
        reward = -loss

        # record destroy
        info["destroy"] = 0.0

        return reward, info

    def _get_res_bus_v(self):
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        return v
    
    def _get_res_bus_active(self):
        active = self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True)
        return active

    def _get_res_bus_reactive(self):
        reactive = self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True)
        return reactive

    def _get_res_line_loss(self):
        line_loss = self.powergrid.res_line["pl_mw"].sort_index().to_numpy(copy=True)
        return line_loss

    def _get_sgen_active(self):
        active = self.powergrid.sgen["p_mw"].to_numpy(copy=True)
        return active
    
    def _get_sgen_reactive(self):
        reactive = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)
        return reactive
    
    def _init_render(self):
        from .rendering_voltage_control_env import Viewer
        self.viewer = Viewer()
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=(mode == "rgb_array"))

    def res_pf_plot(self):
        if not os.path.exists("environments/var_voltage_control/plot_save"):
            os.mkdir("environments/var_voltage_control/plot_save")

        fig = pf_res_plotly(self.powergrid, 
                            aspectratio=(1.0, 1.0), 
                            filename="environments/var_voltage_control/plot_save/pf_res_plot.html", 
                            auto_open=False,
                            climits_volt=(0.9, 1.1),
                            line_width=5, 
                            bus_size=12,
                            climits_load=(0, 100),
                            cpos_load=1.1,
                            cpos_volt=1.0
                        )
        fig.write_image("environments/var_voltage_control/plot_save/pf_res_plot.jpeg")
        #fig.write_html("environments/var_voltage_control/plot_save/pf_res_plot.html")
