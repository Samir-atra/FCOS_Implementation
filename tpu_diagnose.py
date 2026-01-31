
import os
import sys
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print(f"Python Version: {sys.version}")
print("Environment Variables related to TPU:")
for k, v in os.environ.items():
    if "TPU" in k or "XLA" in k:
        print(f"  {k}: {v}")

print("\n--- Attempting TPU Initialization ---")

tpu = None
strategy = None

# Method 1: 'local' (VM)
print("\n[Method 1] trying TPUClusterResolver(tpu='local')...")
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    print("  Success! tpu.master():", tpu.master())
except Exception as e:
    print(f"  Failed: {e}")

# Method 2: Auto-detect
if not tpu:
    print("\n[Method 2] trying TPUClusterResolver()...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("  Success! tpu.master():", tpu.master())
    except Exception as e:
        print(f"  Failed: {e}")

if tpu:
    print("\n--- Initializing TPU System ---")
    try:
        # For TPU VMs, connect_to_cluster is often NOT needed or fails.
        # Check master
        master = tpu.master()
        if 'local' in master:
             print("  Master is local, skipping connect_to_cluster.")
        else:
             print("  Master is remote, calling connect_to_cluster...")
             tf.config.experimental_connect_to_cluster(tpu)
        
        print("  Calling initialize_tpu_system...")
        tf.tpu.experimental.initialize_tpu_system(tpu)
        
        print("  Creating TPUStrategy...")
        strategy = tf.distribute.TPUStrategy(tpu)
        print("  TPU Strategy created successfully!")
        print(f"  Number of replicas: {strategy.num_replicas_in_sync}")
        
        print("\n--- Running Test Computation ---")
        with strategy.scope():
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("  Computation result (AXB):")
            print(c)
            
    except Exception as e:
        print(f"  TPU System Error: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\nCRITICAL: Could not resolve TPU using any method.")
