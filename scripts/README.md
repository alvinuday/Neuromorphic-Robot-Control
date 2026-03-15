# Scripts Directory

## Quick Reference

### Utility & Diagnostics
- **check_dataset_cache.py**: Verify LeRobot dataset is downloaded and accessible
- **health_check_vla.py**: Check if VLA server is running and responsive
- **inspect_dataset.py**: Debug dataset structure and available columns
- **test_vla_server.py**: Basic VLA server connection test

### Phase Testing

#### Phase 12: VLA Server Benchmarking
- **phase12_quick_test.py**: Quick validation test (3 episodes per benchmark)
  - Run before full benchmarks to verify setup works
  - Usage: `python3 scripts/phase12_quick_test.py`

#### Phase 13: Real Sensor Fusion Ablation Study
- **phase13_final.py**: Production ablation study (30 episodes × 5 fusion modes)
  - Tests all fusion modes: RGB, RGB+Events, RGB+LiDAR, RGB+Proprio, Full Fusion
  - Uses real LeRobot images (3 cameras) + real robot state data
  - Extracts real sensor features (no synthetic/mocked data)
  - Outputs results to: `evaluation/results/phase13_30ep_real_*.json`
  - Usage: `python3 scripts/phase13_final.py`
  - Runtime: ~3-4 minutes for 150 total episodes

## Unit Tests

- **../test_encoder_simple.py**: Validate real sensor encoder on synthetic data
  - Usage: `python3 test_encoder_simple.py`
  - Should complete in <1 second with all 5 modes passing

## File Organization

```
scripts/
├── check_dataset_cache.py       # Utility
├── health_check_vla.py          # Utility  
├── inspect_dataset.py           # Utility
├── test_vla_server.py           # Utility
├── phase12_quick_test.py        # Phase 12 testing
├── phase13_final.py             # Phase 13 ablation study
└── README.md                    # This file
```

## Notes

- All phase variants and debug scripts have been removed to maintain a clean repo
- Only the final, working versions of scripts remain
- Archived logs are in `logs/archived/`
- Results are saved to `evaluation/results/`
