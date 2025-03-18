import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Compare model training performance
def compare_env(log_paths, labels, tag='rollout/ep_rew_mean', output_file=None):
    output_file = output_file if output_file else 'comparison.png'

    plt.figure(figsize=(12, 6))

    # Plot normalized data
    for log_path, label in zip(log_paths, labels):
        event_acc = event_accumulator.EventAccumulator(log_path)
        event_acc.Reload()

        if tag in event_acc.Tags()['scalars']:
            # Get steps and values from tensorboard
            raw_steps = [event.step for event in event_acc.Scalars(tag)]
            values = [event.value for event in event_acc.Scalars(tag)]
            
            # Stretch all values from 0-100%
            if len(raw_steps) > 0:
                max_step = max(raw_steps)
                normalized_steps = [step / max_step for step in raw_steps]
                plt.plot(normalized_steps, values, label=label)
    
    # A-axis tick labels
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['0%', '25%', '50%', '75%', '100%'])
    
    # Add chart labels
    plt.title(f'Comparison of {tag}')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save file
    plt.savefig(output_file)
    

# Compare model training performance
def compare_models(log_paths, labels, tag='rollout/ep_rew_mean', output_file=None):
    output_file = output_file if output_file else 'comparison.png'

    plt.figure(figsize=(12, 6))
    
    for log_path, label in zip(log_paths, labels):
        event_acc = event_accumulator.EventAccumulator(log_path)
        event_acc.Reload()
        
        # Check for tags
        if tag in event_acc.Tags()['scalars']:
            steps = [event.step for event in event_acc.Scalars(tag)]
            values = [event.value for event in event_acc.Scalars(tag)]
            plt.plot(steps, values, label=label)
    
    # Add chart labels
    plt.title(f'Comparison of {tag}')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save file
    plt.savefig(output_file)
    

if __name__ == "__main__":

    tracks = ['highway', 'roundabout']
    models = ['dqn', 'ppo', 'a2c']

    # Loop over environments and models
    for i in range(1):
        try:
            os.makedirs(f"./{tracks[i]}_charts/", exist_ok=True)
        except OSError as e:
            print(e)
        paths = []
        labels = []
        for j in range(3):
            
            # Build paths and names
            path1 = f'./{tracks[i]}_{models[j]}/{models[j].upper()}_1'
            path2 = f'./{tracks[i]}_{models[j]}/{models[j].upper()}_2'
            label1 = f'{tracks[i].upper()} {models[j].upper()} 1'
            label2 = f'{tracks[i].upper()} {models[j].upper()} 2'
            
            # Compare models of the same type
            compare_models(
                [path1, path2],
                [label1 ,label2],
                'rollout/ep_rew_mean',
                f'./{tracks[i]}_charts/{models[j]}_{tracks[i]}_comparison.png'
            )
        
            paths.append(path1)
            paths.append(path2)
            labels.append(label1)
            labels.append(label2)
    
        # Compare all models un-normalized steps
        compare_models(
            paths,
            labels,
            'rollout/ep_rew_mean',
            f'./{tracks[i]}_charts/{tracks[i]}_total_comparison.png'
        )

        # Compare all models normalized steps
        compare_env(
            paths,
            labels,
            'rollout/ep_rew_mean',
            f'./{tracks[i]}_charts/{tracks[i]}_normalized_comparison.png'
        )

    # Racetrack charts
    try:
        os.makedirs(f"./racetrack_charts/", exist_ok=True)
    except OSError as e:
        print(e)
    
    # Compare all models un-normalized steps
    compare_models(
        [f'./racetrack_trpo/TRPO_1',
         f'./racetrack_ppo/PPO_1',
         f'./racetrack_a2c/A2C_1'],
        ['RACETRACK TRPO', 'RACETRACK PPO', 'RACETRACK A2C'],
        'rollout/ep_rew_mean',
        f'./racetrack_charts/racetrack_total_comparison.png'
    )