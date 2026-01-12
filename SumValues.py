import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def getsum(log_dir, tag_name):
    """Extract scalar values from TensorBoard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get all scalar summaries
    scalars = event_acc.Scalars(tag_name) 
    for scalar_event in scalars:
        print(f"Step: {scalar_event.step}, Value: {scalar_event.value}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True, help="Path to TensorBoard log directory")
    ap.add_argument("--tag_name", type=str, required=True, help="Scalar tag name to extract")
    args = ap.parse_args()
    
    getsum(args.log_dir, args.tag_name)

if __name__ == "__main__":
    main()
