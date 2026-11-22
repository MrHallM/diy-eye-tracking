# prep/

Turn each dataset into the Baballonia contract (see ../README.md): eye crop -> grayscale -> 128x128 -> EqualizeHist, 4-frame stacks (newest first), labels mapped to Baballonia's 0-1 pitch/yaw/lid ranges. One script per dataset (vrgaze.py first), writing to EyeTrackingLab/data/prepped/<dataset>/. Not written yet.
