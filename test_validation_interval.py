#!/usr/bin/env python
"""Test validation interval logic to verify it triggers correctly."""

import sys

def test_validation_interval():
    """Test that validation triggers at every N iterations."""
    max_iters = 1000
    val_interval_iters = 20
    
    validation_steps = []
    
    for current_iter in range(1, max_iters + 1):
        # This is the exact condition from the training loop
        if current_iter % val_interval_iters == 0 or current_iter >= max_iters:
            validation_steps.append(current_iter)
    
    print(f"Configuration: max_iters={max_iters}, val_interval={val_interval_iters}")
    print(f"Total validation steps: {len(validation_steps)}")
    print(f"\nFirst 10 validation steps: {validation_steps[:10]}")
    print(f"Last 10 validation steps: {validation_steps[-10:]}")
    
    # Verify the pattern
    expected_steps = list(range(val_interval_iters, max_iters + 1, val_interval_iters))
    expected_steps.append(max_iters)  # Add final iteration
    expected_steps = sorted(set(expected_steps))  # Remove duplicates if max_iters is divisible
    
    print(f"\nExpected validation at: every {val_interval_iters} iterations")
    print(f"Validation happens at iterations: {validation_steps[:5]}... (every 20 steps)")
    
    if validation_steps == expected_steps:
        print("\n✓ Validation interval logic is CORRECT!")
        print(f"  Validates at iterations: 20, 40, 60, ..., {max_iters}")
        return True
    else:
        print("\n✗ Validation interval logic is WRONG!")
        print(f"  Expected: {expected_steps[:5]}...")
        print(f"  Got: {validation_steps[:5]}...")
        return False

if __name__ == '__main__':
    success = test_validation_interval()
    sys.exit(0 if success else 1)
