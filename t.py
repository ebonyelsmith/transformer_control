import sys
sys.path.insert(0, 'analysis')
from acrobot_analysis import check_stability
import pandas as pd
import numpy as np

# List of stabilizing runs to test
test_runs = [
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_284408/run_005/run_005_l1_1.64_l2_1.19_m1_0.89_m2_1.18_context_20/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_265000/run_009/run_009_l1_1.65_l2_1.40_m1_0.86_m2_1.32_context_40/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_275000/run_004/run_004_l1_1.97_l2_1.38_m1_0.68_m2_1.00_context_40/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_280000/run_003/run_003_l1_1.98_l2_1.48_m1_0.55_m2_1.13_context_40/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_210000/run_003/run_003_l1_1.94_l2_1.04_m1_0.61_m2_1.02_context_20/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_225000/run_005/run_005_l1_1.94_l2_1.48_m1_0.53_m2_1.36_context_20/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_205000/run_003/run_003_l1_1.93_l2_1.46_m1_0.51_m2_1.36_context_20/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_205000/run_003/run_003_l1_1.93_l2_1.46_m1_0.51_m2_1.36_context_1/states.csv'
]

non_stabilizing_runs = [
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_190000/run_008/run_008_l1_1.90_l2_1.07_m1_0.89_m2_1.24_context_50/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_200000/run_006/run_006_l1_1.74_l2_1.27_m1_0.68_m2_1.40_context_30/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_200000/run_008/run_008_l1_1.56_l2_1.23_m1_0.78_m2_1.13_context_20/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_200000/run_002/run_002_l1_1.53_l2_1.09_m1_0.64_m2_1.15_context_50/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_200000/run_009/run_009_l1_1.52_l2_1.04_m1_0.94_m2_1.08_context_20/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_205000/run_001/run_001_l1_1.78_l2_1.49_m1_0.60_m2_1.33_context_1/states.csv',
    'acrobot/videos/acrobot_inference_gym_runs/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/step_205000/run_001/run_001_l1_1.78_l2_1.49_m1_0.60_m2_1.33_context_40/states.csv'
]

print('='*80)
print('Testing Stability Detection on Known Stabilizing Runs')
print('='*80)

all_passed = True
for i, run_path in enumerate(test_runs, 1):
    states = pd.read_csv(run_path)
    
    # Get file name for display
    run_name = run_path.split('/')[-2]
    
    # Check with different thresholds
    result_02 = check_stability(states, theta_threshold=0.2)
    result_03 = check_stability(states, theta_threshold=0.3)
    
    # Analyze final window
    theta1_final = states['theta1'].values[-50:]
    theta2_final = states['theta2'].values[-50:]
    
    theta1_mean = theta1_final.mean()
    theta2_mean = theta2_final.mean()
    theta1_std = theta1_final.std()
    theta2_std = theta2_final.std()
    
    print(f'\nRun {i}: {run_name}')
    print(f'  Final θ₁: {theta1_mean:.4f} ± {theta1_std:.4f} (target: π = {np.pi:.4f})')
    print(f'  Final θ₂: {theta2_mean:.4f} ± {theta2_std:.4f} (target: 0)')
    print(f'  Stable @ 0.2: {result_02}')
    print(f'  Stable @ 0.3 (DEFAULT): {result_03}')
    
    if not result_03:
        print('  ⚠️  WARNING: Not detected as stable with threshold 0.3!')
        all_passed = False
    else:
        print('  ✓ Correctly identified as stable')

print('\n' + '='*80)
print('Testing Stability Detection on Known NON-Stabilizing Runs')
print('='*80)

for i, run_path in enumerate(non_stabilizing_runs, 1):
    states = pd.read_csv(run_path)
    
    # Get file name for display
    run_name = run_path.split('/')[-2]
    
    # Check with different thresholds
    result_02 = check_stability(states, theta_threshold=0.2)
    result_03 = check_stability(states, theta_threshold=0.3)
    
    # Analyze final window
    theta1_final = states['theta1'].values[-50:]
    theta2_final = states['theta2'].values[-50:]
    
    theta1_mean = theta1_final.mean()
    theta2_mean = theta2_final.mean()
    theta1_std = theta1_final.std()
    theta2_std = theta2_final.std()
    
    print(f'\nRun {i}: {run_name}')
    print(f'  Final θ₁: {theta1_mean:.4f} ± {theta1_std:.4f} (target: π = {np.pi:.4f})')
    print(f'  Final θ₂: {theta2_mean:.4f} ± {theta2_std:.4f} (target: 0)')
    print(f'  Stable @ 0.2: {result_02}')
    print(f'  Stable @ 0.3 (DEFAULT): {result_03}')
    
    if result_03:
        print('  ⚠️  WARNING: Incorrectly detected as stable with threshold 0.3!')
        all_passed = False
    else:
        print('  ✓ Correctly identified as NOT stable')

print('\n' + '='*80)
if all_passed:
    print('✓ ALL TESTS PASSED: All runs correctly identified')
else:
    print('❌ SOME TESTS FAILED: Check the warnings above')
print('='*80)
