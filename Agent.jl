### This Code does ###

# This code defines the training procedure for the approximation of the complete solution of the damped two-level system with spontaneous emission based on the solution of its unitary part. The code uses the model-free, off-policy Reinforcement Learning method of Q-Learning to do so. 

### Preliminaries ###

if length(ARGS) != 1
    println("Usage: julia Agent.jl <output_file>")
    exit(1)
end

include("Variables.jl") # == Variables.
include("UnitaryQTable.jl") # == QTable.
include("Rewardfield.jl") # == Reward.
include("LocationsAndActions.jl") # == MovementFunctions.

# Enable terminal output of the Q_Table as a byte stream
using Serialization

# Take care of the right qualified paths, especially if a module is called inside another module!

### Program ###

for episode in 1:Variables.training_time
    if episode % 10000 == 0
        print("Episode: $episode\r")
    end

    # Define the starting location for later use
    row, col = MovementFunctions.initial_location()

    # Check if the current location is terminal
    while MovementFunctions.is_location_terminal(row, col) == false
        
        # Safe the row and column temporarily
        old_row = row
        old_col = col

        # Decide upon an action with respect to the ε-greedy algorithm
        current_action = MovementFunctions.next_action(row, col, Variables.ε)
        
        # Find the new row and column after the action is applied
        row, col = MovementFunctions.next_location(row, col, current_action)
        
        # Take the immediate reward and calculate the temporal difference error 
        immediate_reward = Reward.rewards[row, col]
        old_Q_value = MovementFunctions.QTable.Q_values[old_row, old_col, current_action]
        TD_error = immediate_reward + Variables.γ * maximum(MovementFunctions.QTable.Q_values[row, col, :]) - old_Q_value
        
        # Update the Q values with respect to the old Q value, the TD error and the learning rate and update the Q table as well
        new_Q_value = old_Q_value + Variables.α * TD_error
        MovementFunctions.QTable.Q_values[old_row, old_col, current_action] = new_Q_value
    end
end
println("Training completed.")

# Save the final Q-Table into a file 
serialize("QTable_check.txt", MovementFunctions.QTable.Q_values[:,:,:])

# Save the initial Q-Table into a file
# Note the difference to the first serialize where the qualified name is MovementFunctions.QTable. which indicates that the Q-Table called and altered in MovementFunctions is called, whereas here in the second serialize, the qualified name is just QTable. which means in the second serialize the initial Q-Table, created in the module QTable, written in the program UnitarQTable.jl, is called. This Q-Table only consists of the set of initial Q-values and thus a huge number of zeros instead of a version altered by the RL agent
serialize("InitQTable.txt", QTable.Q_values[:,:,:])

# Write the final path for the initial location into a file
# The path in the file is supposed to be near the analytical solution of the damped two-level system with spontaneous emission
MovementFunctions.collect_pathpoints(MovementFunctions.initial_location(), ARGS[1])
