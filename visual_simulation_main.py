from __future__ import absolute_import
from __future__ import print_function
import os
import datetime
import optparse
import configparser
from shutil import copyfile
from visualization import Visualization
from visual_simulation import VisualSimulation
from generator import TrafficGenerator
from utils import import_visual_configuration, set_sumo, set_simple_vis_path


if __name__ == "__main__":
    config = import_visual_configuration(config_file='visual_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_simple_vis_path(config['models_path_name'], config['model_to_simple_vis'])

    # Initialize the Traffic Generator
    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path,
        dpi=96
    )

    # Create and run the Visual Simulation
    Simulation = VisualSimulation(
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
    )

    print('\n----- Benchmark episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Benchmark visualisation info saved at:", plot_path)

    copyfile(src='visual_settings.ini', dst=os.path.join(plot_path, 'visual_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.avg_speed_episode, filename='avg_speed', xlabel='Step', ylabel='Average Speed (m/s)')
    Visualization.save_data_and_plot(data=Simulation.incoming_density_episode, filename='incoming_density', xlabel='Step', ylabel='Incoming Lane Density (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.outgoing_density_episode, filename='outgoing_density', xlabel='Step', ylabel='Outgoing Lane Density (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.pressure_episode, filename='pressure', xlabel='Step', ylabel='Pressure (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Step', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.traffic_generation, filename='generation', xlabel='Step', ylabel='Number of Vehicles')