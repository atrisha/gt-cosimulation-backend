"""
Module to generate game model from vehicle maneuvers and trajectories.
Extracted from run_merge_before_intersection in run.py (up to line 139).
Instead of build_complete_tree, calls build_initial_reachability_states and returns trajectories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add path to game_theoretic_planner for constants module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'game_theoretic_planner'))

import numpy as np
import time
import json
from equilibrium.equilibria_calculation import *
from maps.States import TwoAgentSyntheticScenarioDef
import rg_constants
import traceback
from maps.map_info import MergeBeforeIntersection
from equilibrium.game_tree import *
import matplotlib.pyplot as plt
from shapely.geometry import linestring, Point
import pygambit as gambit
from pyproj import Transformer

def load_vehicle_data():
    """
    Load vehicle data from vehicle_data.json and extract agent parameters.
    Returns agent parameters for the first valid entry with both subject and relative vehicles.
    """
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(script_dir, 'vehicle_data.json')
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Iterate through entries to find one with both subject and relative vehicles
        for entry in data:
            if 'subject_vehicle' not in entry or 'relative_vehicle' not in entry:
                continue
                
            if entry['relative_vehicle'] is None:
                print("Skipping entry: relative_vehicle is null")
                continue
            
            # Extract subject vehicle (agent_1) data
            subject_vehicle = entry['subject_vehicle']
            agent1_speed_kph = subject_vehicle['speed']  # in kph
            agent1_speed_mps = agent1_speed_kph / 3.6    # convert to m/s

            # Convert all positions and centerline coordinates from lat/lon to UTM, zone 17T
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)  # WGS84 to UTM 17T
            
            # Convert agent positions from lat/lon to UTM
            agent1_lon, agent1_lat = subject_vehicle['position']
            agent1_utm_x, agent1_utm_y = transformer.transform(agent1_lon, agent1_lat)
            agent1_pos = (agent1_utm_x, agent1_utm_y)
            
            # Extract relative vehicle (agent_2) data
            relative_vehicle = entry['relative_vehicle']
            agent2_speed_kph = relative_vehicle['speed']  # in kph
            agent2_speed_mps = agent2_speed_kph / 3.6     # convert to m/s
            
            agent2_lon, agent2_lat = relative_vehicle['position']
            agent2_utm_x, agent2_utm_y = transformer.transform(agent2_lon, agent2_lat)
            agent2_pos = (agent2_utm_x, agent2_utm_y)
            
            # Extract centerlines from centerlines_ordered only
            centerlines_ordered = entry['centerlines_ordered']
            
            # Convert centerline coordinates from lat/lon to UTM
            target_lane_2_utm = []
            for lon, lat in centerlines_ordered['target_lane_2']:
                utm_x, utm_y = transformer.transform(lon, lat)
                target_lane_2_utm.append((utm_x, utm_y))

            agent1_waypoints = [agent1_pos] + target_lane_2_utm
            agent2_waypoints = [agent2_pos] + target_lane_2_utm


            print(f"Successfully loaded vehicle data:")
            print(f"Agent 1 (subject): speed={agent1_speed_kph:.1f} kph, pos={agent1_pos}")
            print(f"Agent 2 (relative): speed={agent2_speed_kph:.1f} kph, pos={agent2_pos}")
            print(f"Agent 1 waypoints: {len(agent1_waypoints)} points")
            print(f"Agent 2 waypoints: {len(agent2_waypoints)} points")
            
            return {
                'agent1_speed': agent1_speed_mps,
                'agent2_speed': agent2_speed_mps,
                'agent1_pos': agent1_pos,
                'agent2_pos': agent2_pos,
                'agent1_waypoints': agent1_waypoints,
                'agent2_waypoints': agent2_waypoints
            }
            
    except FileNotFoundError:
        print(f"Error: vehicle_data.json not found at {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return None
    except Exception as e:
        print(f"Error loading vehicle data: {e}")
        return None
    
    print("Warning: No valid entries found in vehicle_data.json")
    return None

def visualize_waypoints(agent1_centerline=None, agent2_centerline=None):
    """
    Visualize waypoints using matplotlib.
    If centerlines are provided, plot them. Otherwise, plot MergeBeforeIntersection waypoints.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    if agent1_centerline and agent2_centerline:
        # Plot provided centerlines
        agent1_x = [point[0] for point in agent1_centerline]
        agent1_y = [point[1] for point in agent1_centerline]
        agent2_x = [point[0] for point in agent2_centerline]
        agent2_y = [point[1] for point in agent2_centerline]
        
        plt.plot(agent1_x, agent1_y, 'bo-', label=f'Agent 1 Centerline ({len(agent1_centerline)} points)', markersize=6, linewidth=2)
        plt.plot(agent2_x, agent2_y, 'ro-', label=f'Agent 2 Centerline ({len(agent2_centerline)} points)', markersize=6, linewidth=2)
        
        # Mark start and end points
        plt.plot(agent1_x[0], agent1_y[0], 'bs', markersize=10, markeredgecolor='black', markeredgewidth=2, label='Agent 1 Start')
        plt.plot(agent1_x[-1], agent1_y[-1], 'b^', markersize=10, markeredgecolor='black', markeredgewidth=2, label='Agent 1 End')
        plt.plot(agent2_x[0], agent2_y[0], 'rs', markersize=10, markeredgecolor='black', markeredgewidth=2, label='Agent 2 Start')
        plt.plot(agent2_x[-1], agent2_y[-1], 'r^', markersize=10, markeredgecolor='black', markeredgewidth=2, label='Agent 2 End')
        
        plt.title('Agent Centerlines Visualization')
    else:
        # Fallback to MergeBeforeIntersection waypoints
        if hasattr(MergeBeforeIntersection, 'ol_waypoints') and MergeBeforeIntersection.ol_waypoints:
            ol_x = [point[0] for point in MergeBeforeIntersection.ol_waypoints]
            ol_y = [point[1] for point in MergeBeforeIntersection.ol_waypoints]
            plt.plot(ol_x, ol_y, 'bo-', label='OL Waypoints (Agent 1)', markersize=6, linewidth=2)
        
        if hasattr(MergeBeforeIntersection, 'mv_waypoints') and MergeBeforeIntersection.mv_waypoints:
            mv_x = [point[0] for point in MergeBeforeIntersection.mv_waypoints]
            mv_y = [point[1] for point in MergeBeforeIntersection.mv_waypoints]
            plt.plot(mv_x, mv_y, 'ro-', label='MV Waypoints (Agent 2)', markersize=6, linewidth=2)
        
        plt.title('MergeBeforeIntersection Waypoints Visualization')
    
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def solve_game(trajectories):
    """
    Build payoff table from trajectories and compute equilibrium strategies.
    
    Structure of trajectories:
    {'agent_1': {maneuver_name: maneuver_mode: [list of trajectories]}, 'agent_2': {...}}
    Each trajectory is a player's strategy.
    Construct the normal form game and call Gambit's pygambit API to get the equilibrium strategies.
    """
    import pygambit as gambit
    
    # Extract strategies for each agent
    agent1_strategies = []
    agent2_strategies = []
    
    # Flatten agent_1 trajectories into strategy list
    for maneuver_name, maneuver_modes in trajectories['agent_1'].items():
        for mode_name, trajectory_list in maneuver_modes.items():
            for i, trajectory in enumerate(trajectory_list):
                strategy_name = f"agent1_{maneuver_name}_{mode_name}_{i}"
                agent1_strategies.append((strategy_name, trajectory))
    
    # Flatten agent_2 trajectories into strategy list
    for maneuver_name, maneuver_modes in trajectories['agent_2'].items():
        for mode_name, trajectory_list in maneuver_modes.items():
            for i, trajectory in enumerate(trajectory_list):
                strategy_name = f"agent2_{maneuver_name}_{mode_name}_{i}"
                agent2_strategies.append((strategy_name, trajectory))
    
    print(f"Agent 1 has {len(agent1_strategies)} strategies")
    print(f"Agent 2 has {len(agent2_strategies)} strategies")
    
    # Create the normal form game
    num_agent1_strategies = len(agent1_strategies)
    num_agent2_strategies = len(agent2_strategies)
    
    if num_agent1_strategies == 0 or num_agent2_strategies == 0:
        print("Warning: One or both agents have no strategies. Cannot build payoff table.")
        return None
    
    # Create gambit game
    game = gambit.Game.new_table([num_agent1_strategies, num_agent2_strategies])
    game.title = "Vehicle Merging Game"
    
    # Set player names
    game.players[0].label = "Agent_1"
    game.players[1].label = "Agent_2"
    
    # Set strategy labels
    for i, (strategy_name, _) in enumerate(agent1_strategies):
        game.players[0].strategies[i].label = strategy_name
    
    for j, (strategy_name, _) in enumerate(agent2_strategies):
        game.players[1].strategies[j].label = strategy_name
    
    # Calculate payoffs for each strategy combination
    for i, (agent1_strategy_name, agent1_traj) in enumerate(agent1_strategies):
        for j, (agent2_strategy_name, agent2_traj) in enumerate(agent2_strategies):
            # Calculate payoffs based on trajectory interaction
            payoff1, payoff2 = calculate_trajectory_payoffs(agent1_traj, agent2_traj)
            
            # Set payoffs in the game matrix - correct pygambit syntax
            game[i, j][game.players[0]] = payoff1  # Payoff for player 0 (agent 1)
            game[i, j][game.players[1]] = payoff2  # Payoff for player 1 (agent 2)
    
    print("Payoff table constructed successfully")
    
    # Solve for Nash equilibria using Gambit
    try:
        # Use LCP solver to find Nash equilibria
        result = gambit.nash.lcp_solve(game, rational=False, stop_after=1)
        equilibria = result.equilibria
        
        if len(equilibria) > 0:
            eq = equilibria[0]  # Take the first equilibrium
            
            print(f"Found {len(equilibria)} equilibrium(a)")
            
            # Extract mixed strategy probabilities for each player
            agent1_probs = [float(eq[game.players[0]][strategy]) for strategy in game.players[0].strategies]
            agent2_probs = [float(eq[game.players[1]][strategy]) for strategy in game.players[1].strategies]
            
            print(f"Agent 1 mixed strategy probabilities: {agent1_probs}")
            print(f"Agent 2 mixed strategy probabilities: {agent2_probs}")
            
            # Sample strategies based on equilibrium probabilities
            sampled_agent1_strategy_idx = np.random.choice(range(len(agent1_strategies)), p=agent1_probs)
            sampled_agent2_strategy_idx = np.random.choice(range(len(agent2_strategies)), p=agent2_probs)
            
            sampled_agent1_strategy = agent1_strategies[sampled_agent1_strategy_idx]
            sampled_agent2_strategy = agent2_strategies[sampled_agent2_strategy_idx]
            
            print(f"Sampled Agent 1 strategy: {sampled_agent1_strategy[0]}")
            print(f"Sampled Agent 2 strategy: {sampled_agent2_strategy[0]}")
            
            game_results = {
                'game': game,
                'agent1_strategies': agent1_strategies,
                'agent2_strategies': agent2_strategies,
                'equilibrium': eq,
                'agent1_probs': agent1_probs,
                'agent2_probs': agent2_probs,
                'sampled_strategies': {
                    'agent1': sampled_agent1_strategy,
                    'agent2': sampled_agent2_strategy
                }
            }
            
            return game_results
        else:
            print("No equilibria found")
            return None
        
    except Exception as e:
        print(f"Error solving for equilibria: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def calculate_trajectory_payoffs(agent1_traj, agent2_traj):
    """
    Calculate payoffs for a pair of trajectories based on safety, efficiency, and comfort.
    
    Returns:
        tuple: (payoff_agent1, payoff_agent2)
    """
    # Simple payoff calculation based on trajectory properties
    # In a real implementation, this would consider collision risk, travel time, comfort, etc.
    
    # Base payoffs (higher is better)
    base_payoff = 10.0
    
    # Example payoff calculations:
    # 1. Collision avoidance bonus/penalty
    safety_util = calculate_collision_risk(agent1_traj, agent2_traj)
    
   
    agent_1_progress_util = np.log(linestring.LineString([(x[1],x[2]) for x in agent1_traj]).length +1)/4
    agent_2_progress_util = np.log(linestring.LineString([(x[1],x[2]) for x in agent2_traj]).length +1)/4

    safety_mixture_param = 0.75

    payoff1 = safety_mixture_param * safety_util + (1 - safety_mixture_param) * agent_1_progress_util
    payoff2 = safety_mixture_param * safety_util + (1 - safety_mixture_param) * agent_2_progress_util

    return payoff1, payoff2


def calculate_collision_risk(agent1_traj, agent2_traj):
    """
    Calculate collision risk between two trajectories.
    
    Returns:
        float: Risk value between 0 and 1
    """
    # Extract x,y coordinates from trajectories
    try:
        # Extract (x, y) coordinates from trajectories
        agent1_coords = [(x[1], x[2]) for x in agent1_traj] if len(agent1_traj) > 0 and len(agent1_traj[0]) > 2 else []
        agent2_coords = [(x[1], x[2]) for x in agent2_traj] if len(agent2_traj) > 0 and len(agent2_traj[0]) > 2 else []
        
        # Handle empty trajectories
        if not agent1_coords or not agent2_coords:
            return 0.5  # Moderate risk for empty trajectories
        
        # Enforce same size by truncating to minimum length
        min_length = min(len(agent1_coords), len(agent2_coords))
        agent1_coords = agent1_coords[:min_length]
        agent2_coords = agent2_coords[:min_length]
        
        # Calculate node-wise Euclidean distances
        distances = []
        for i in range(min_length):
            x1, y1 = agent1_coords[i]
            x2, y2 = agent2_coords[i]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(dist)
        
        # Calculate minimum distance
        min_dist = min(distances) if distances else float('inf')
        
       
        risk = np.log(min_dist + 1)/2
        
        # Clamp risk between 0 and 1
        safety_util = max(0.0, min(1.0, risk))
        return safety_util

    except (IndexError, TypeError, ValueError) as e:
        # Fallback for trajectory format issues
        print(f"Warning: Error in collision risk calculation: {e}")
        return 0.3  # Default moderate risk

def generate_game_model(run_id, agent1_id, agent2_id, agent1_vel, agent2_vel, agent1_pos, agent2_pos, agent1_centerline, agent2_centerline):
    """
    Generate game model from vehicle maneuvers and trajectories.
    Extracted from run_merge_before_intersection up to line 139.
    Instead of build_complete_tree, calls build_initial_reachability_states and returns trajectories.
    
    Args:
        run_id: Unique run identifier
        agent1_id: Agent 1 identifier
        agent2_id: Agent 2 identifier  
        agent1_vel: Agent 1 initial velocity (m/s)
        agent2_vel: Agent 2 initial velocity (m/s)
        agent1_pos: Agent 1 initial position (x, y)
        agent2_pos: Agent 2 initial position (x, y)
        agent1_centerline: Agent 1 centerline waypoints
        agent2_centerline: Agent 2 centerline waypoints
    """
    start_time = time.time()
    initialize_db = False
    freq = 0.5
    file_id = str(run_id) + '_' + str(agent1_id) + '-' + str(agent2_id) + '_' + str(agent1_vel).replace('.', ',') + '_' + str(agent2_vel).replace('.', ',')
    rg_constants.CURRENT_RG_FILE_ID = file_id
    rg_constants.SCENE_TYPE = ('synthetic', 'merge_before_intersection')
    rg_constants.ind_run_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_files')
    scene_def = TwoAgentSyntheticScenarioDef(initialize_db=initialize_db, file_id=file_id)
    
    scene_def.add_agent(agent_tag='agent_1', agent_id=agent1_id, agent_init_velocity_mps=agent1_vel, 
                       agent_waypoints=agent1_centerline, 
                       agent_waypoint_segments=MergeBeforeIntersection.ol_waypoint_segments, 
                       direction='L_S_W', file_id=file_id, initialize_db=True, start_ts=0, freq=freq,agent_type='vehicle')
    scene_def.add_agent(agent_tag='agent_2', agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, 
                       agent_waypoints=agent2_centerline, 
                       agent_waypoint_segments=MergeBeforeIntersection.mv_waypoint_segments, 
                       direction='L_S_W', file_id=file_id, initialize_db=True, start_ts=0, freq=freq,agent_type='vehicle')
    #visualize_waypoints()
    maneuver_map = {'agent_1': {'maneuvers': {'wait': None, 'turn': None}, 'agent_state': scene_def.agent1},
                   'agent_2': {'maneuvers': {'wait': None, 'turn': None}, 'agent_state': scene_def.agent2}}
    maneuver_constraints = scene_def.setup_trajectory_constraints(maneuver_map=maneuver_map)
    
    for k, v in maneuver_constraints['agent_2']['maneuvers'].items():
        setattr(v, 'path_degree', 5)
    
    maneuver_constraints['agent_1']['step_dist'], maneuver_constraints['agent_2']['step_dist'] = 2, 2
    time_horizon, init_time = 6, 0
    # Instead of build_complete_tree, call build_initial_reachability_states
    acts = Actions(maneuver_constraints)
    trajectories = acts.generate_actions(init_time,agent1_vel,agent2_vel,time_horizon,initialize_db)
    game_result = solve_game(trajectories)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    # From the trajectories dict, extract samples of trajectories into a payoff table
    # extract x[1],x[2] from game_result sampled_strategies for both agent, then plot it.
    
    
    agent1_traj = game_result['sampled_strategies']['agent1'][1]
    agent2_traj = game_result['sampled_strategies']['agent2'][1]
    
    agent1_coords = [(x[1], x[2]) for x in agent1_traj] if agent1_traj else []
    agent2_coords = [(x[1], x[2]) for x in agent2_traj] if agent2_traj else []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Trajectory subplot
    if agent1_coords:
        ax1.plot(*zip(*agent1_coords), 'b-o', label='Agent 1', linewidth=2)
    if agent2_coords:
        ax1.plot(*zip(*agent2_coords), 'r-o', label='Agent 2', linewidth=2)
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_title('Nash Equilibrium Trajectories')
    ax1.legend(), ax1.grid(True, alpha=0.3), ax1.axis('equal')
    
    # Speed subplot
    if agent1_traj:
        ax2.plot([x[3] * 3.6 for x in agent1_traj], 'b-o', label='Agent 1', linewidth=2)
    if agent2_traj:
        ax2.plot([x[3] * 3.6 for x in agent2_traj], 'r-o', label='Agent 2', linewidth=2)
    ax2.set_xlabel('Time Step'), ax2.set_ylabel('Speed (km/h)'), ax2.set_title('Speed Profiles')
    ax2.legend(), ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(), plt.show()
    
    return {
        'equilibrium_trajectories': (agent1_coords, agent2_coords),
        'maneuver_constraints': maneuver_constraints,
        'scene_def': scene_def,
        'file_id': file_id,
        'freq': freq,
        'agent1_pos': agent1_pos,
        'agent2_pos': agent2_pos,
        'agent1_centerline': agent1_centerline,
        'agent2_centerline': agent2_centerline
    }
    # --- End code ---

# If you want to run as a script

if __name__ == "__main__":
    # Example usage
    try:
        # Load vehicle data from JSON file
        vehicle_data = load_vehicle_data()
        
       
        # Use data from vehicle_data.json
        run_id = 1
        agent1_id = 1
        agent2_id = 2
        agent1_vel = vehicle_data['agent1_speed']
        agent2_vel = vehicle_data['agent2_speed']
        agent1_pos = vehicle_data['agent1_pos']
        agent2_pos = vehicle_data['agent2_pos']
        agent1_centerline = vehicle_data['agent1_waypoints']
        agent2_centerline = vehicle_data['agent2_waypoints']
        print("total distance for agent 1:", linestring.LineString(agent1_centerline).length)
        
        #plot the centerlines
        #visualize_waypoints(agent1_centerline, agent2_centerline)

        print(f"Initializing agents with:")
        print(f"Agent 1: pos={agent1_pos}, vel={agent1_vel:.2f} m/s ({agent1_vel*3.6:.1f} kph), centerline_points={len(agent1_centerline)}")
        print(f"Agent 2: pos={agent2_pos}, vel={agent2_vel:.2f} m/s ({agent2_vel*3.6:.1f} kph), centerline_points={len(agent2_centerline)}")

        result = generate_game_model(run_id, agent1_id, agent2_id, agent1_vel, agent2_vel, 
                                   agent1_pos, agent2_pos, agent1_centerline, agent2_centerline)
        
        
        print("Game model generated successfully!")
        
    except Exception as e:
        print(f"Error generating game model: {e}")
        import traceback
        traceback.print_exc()
