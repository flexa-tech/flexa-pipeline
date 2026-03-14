# QT-017: Clean grasping in calibration — debounce before grasp centroid anchor

## Problem
calibrate_workspace.py line 114 uses raw grasping (66% = 630 frames) for the centroid anchor.
The centroid is essentially the trajectory center, not the actual grasp location, causing systematic misalignment (RMS=0.099m).

## Task 1: Apply clean_grasping_signal in calibration

**File:** calibrate_workspace.py

1. Add import: `from trim_trajectory import clean_grasping_signal`
2. Before line 114, clean the grasping signal (debounce + onset only, no proximity gate):
   ```python
   grasping_arr = np.array(grasping, dtype=float)
   grasping_arr = clean_grasping_signal(grasping_arr)
   grasping = grasping_arr.tolist()
   ```
3. The cleaned grasping is already saved at line 162 (`"grasping": grasping`)

**Acceptance:** Grasp centroid uses fewer, post-onset frames. Better alignment with objects.
