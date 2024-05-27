from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
import torch
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    Model = TestModel(
        input_dim=config['num_states'],
        width=config['width_layers'],  # Added this line
        num_layers=config['num_layers'],  # Added this line
        output_dim=config['num_actions'],  # Added this line
        model_path=model_path,
        device=device
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.avg_speed_episode, filename='avg_speed', xlabel='Step', ylabel='Average Speed (m/s)')
    Visualization.save_data_and_plot(data=Simulation.incoming_density_episode, filename='incoming_density', xlabel='Step', ylabel='Incoming Lane Density (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.outgoing_density_episode, filename='outgoing_density', xlabel='Step', ylabel='Outgoing Lane Density (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.pressure_episode, filename='pressure', xlabel='Step', ylabel='Pressure (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Step', ylabel='Cumulative delay (s)')
