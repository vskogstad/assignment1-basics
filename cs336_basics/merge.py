from cs336_basics.train_model import load_checkpoint, save_checkpoint


# Loads checkpoints, merges them, then does full validation


def merge_checkpoints(base_name, num_checkpoints, merge_duration, decay_type):

    # Create a list of names for checkpoints based on input


    # Validate that checkpoints exist for wanted merging
    if not checkpoint_exists(num_checkpoints, base_name, merge_duration):
        return None
    # go through checkpoints one by one to build up a merged model state
    merged_state = {}
    for checkpoint in checkpoint_list:
        checkpoint_states = load
        for key in checkpoint_states[0].keys():


    # save merged checkpoint

    # validate merged checkpoint
    return validation_loss


def test_merging_stratergy(base_name, checkpoint_range: list, decay_types: list):
    # runs multiple merge_checkpoints to test various alternatives
    results = []
    for num_checkpoints in checkpoint_range:
        for decay in decay_types:
            result = merge_checkpoints(base_name, num_checkpoints, decay)
            results.append([num_checkpoints, decay, result])


    for result in results:
        print(result)

if __name__ == "__main__":
    test_merging_stratergy("wsm", [8, 12, 16, 20], ["linear"])