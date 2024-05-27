import traci
import numpy as np
import random
import timeit
import os
import torch

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class VisualSimulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration):
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._reward_episode = []
        self._queue_length_episode = []
        self._avg_speed_episode = []
        self._incoming_density_episode = []
        self._outgoing_density_episode = []
        self._pressure_episode = []
        self._cumulative_wait_store = []
        self._cumulative_waiting_time = 0
        self._traffic_generation = []  # Added to store traffic generation data

    def run(self, episode):
        """
        Runs the continuous visual simulation
        """
        start_time = timeit.default_timer()

        # Generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0

        while self._step < self._max_steps:
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait


            action = self._choose_action()


            self._set_green_phase(action)
            self._simulate(self._green_duration)

            self._set_yellow_phase(action)
            self._simulate(self._yellow_duration)

            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        total_speed = 0
        total_vehicles = 0

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)

            speed, num_vehicles = self._get_speed()
            incoming_density, outgoing_density, pressure = self._get_density_and_pressure()

            total_speed += speed
            total_vehicles += num_vehicles

            self._incoming_density_episode.append(incoming_density)
            self._outgoing_density_episode.append(outgoing_density)
            self._pressure_episode.append(pressure)

            if total_vehicles > 0:
                avg_speed = total_speed / total_vehicles
            else:
                avg_speed = 0
            self._avg_speed_episode.append(avg_speed)

            current_total_wait = self._collect_waiting_times()
            self._cumulative_wait_store.append(current_total_wait)

            num_vehicles = len(traci.vehicle.getIDList())
            self._traffic_generation.append(num_vehicles)  # Store the number of vehicles

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    # def _collect_waiting_times(self):
    #     """
    #     Retrieve the waiting time of every car in the incoming roads
    #     """
    #     incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    #     car_list = traci.vehicle.getIDList()
    #     total_waiting_time = 0
    #     for car_id in car_list:
    #         wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
    #         road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
    #         if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
    #             total_waiting_time += wait_time
    #     return total_waiting_time

    def _choose_action(self):
        """
        Choose the next action based on simple logic for continuous simulation
        """
        # Implement a simple action choice logic
        return (self._step // (self._green_duration + self._yellow_duration)) % 4

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_speed(self):
        """
        Calculate the total speed of vehicles
        """
        car_list = traci.vehicle.getIDList()
        total_speed = sum([traci.vehicle.getSpeed(car_id) for car_id in car_list])
        num_vehicles = len(car_list)

        return total_speed, num_vehicles

    def _get_density_and_pressure(self):
        """
        Calculate the incoming and outgoing lane densities and the pressure
        """
        incoming_lanes = ["E2TL", "N2TL", "W2TL", "S2TL"]
        outgoing_lanes = ["TL2E", "TL2N", "TL2W", "TL2S"]

        incoming_density = sum([traci.edge.getLastStepVehicleNumber(lane) for lane in incoming_lanes])
        outgoing_density = sum([traci.edge.getLastStepVehicleNumber(lane) for lane in outgoing_lanes])
        pressure = incoming_density - outgoing_density

        return incoming_density, outgoing_density, pressure



    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode

    @property
    def avg_speed_episode(self):
        return self._avg_speed_episode

    @property
    def incoming_density_episode(self):
        return self._incoming_density_episode

    @property
    def outgoing_density_episode(self):
        return self._outgoing_density_episode

    @property
    def pressure_episode(self):
        return self._pressure_episode

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store
    
    @property
    def traffic_generation(self):
        return self._traffic_generation

