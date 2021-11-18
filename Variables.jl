### This Code does ###

# This code stores the variabels needed for the Reinforcement Learning process

### Preliminaries ###

using DataFrames
using CSV

# Load as a module
module Variables

    using DataFrames
    using CSV

    # Keep in mind that for 0.05 env_cols needs to be 126 since the squares are getting smaller and thus more are needed to fill the same areas
    # However unitary_sol_range stays the same, since it is dependend on the stepsize of Unitary_solution.txt

    # DO NOT FORGET TO CHOOSE THE WANTED GRIDSQUARE-SIZE, INIT_Q_VALUES WITH DECAY, SLEEVE, STEADY STATE AND THE TRAINING HYPERPARAMETERS!

    ### Program ###

    # Choose one gridsize!
    # -------------------------------------
    #env_cols = 63
    #sidelength_gridsquare = 0.1

    env_cols = 126
    sidelength_gridsquare = 0.05
    # -------------------------------------

    # -------------------------------------
    sleeve_radius = 0.5
    lower_bound_sleeve = -0.5
    upper_bound_sleeve = 1.5
    # -------------------------------------
 
    # Calculate the number of rows in dependence of the size of the gridsquares and the sleeve radius
    env_rows = Int(round((abs(upper_bound_sleeve) 
		+ abs(lower_bound_sleeve))/sidelength_gridsquare, digits=0) + 1)

    # This variables value is always the same, since it depends on the stepsize in Unitary_solution.jl
    unitary_sol_range = 630

    # Variables that are needed for the Rewardfield and the steady state implementation
    # -------------------------------------
    steady_state = 0.5
    default_reward = -1
    final_steady_reward = 1
    steady_jump = 100
    steady_stepsize = 0.05
    # -------------------------------------

    # -------------------------------------
    init_Q_value = 15.0
    init_Q_value_decay = 0.05
    # -------------------------------------

    # -------------------------------------
    ε = 0.6
    α = 0.5
    γ = 0.7
    training_time = 5_000_000
    # -------------------------------------

    # The prefactor makes allowance for different sizes of the gridsquares in the computation of the rows and columns corresponding to the states and times respectively
    # Also important for the calculation of the right movement along the rows, based on the selected action
    # For sidelength_gridsquare = 0.1 this function equals 1 and for sidelength_gridsquare = 0.05 it equals 2
    # The 63 (unitary_sol_range ÷ 10) needs to be hardcoded, since the alternative would be env_cols, which changes with different gridsquare sizes and thus the resulting prefactor would change too!
    function prefactor(size_gridsquare)
        factor = 1 / ((unitary_sol_range ÷ 63) * size_gridsquare) 
			|> Int
    end

    # Call Unitary_solution.txt to make its values accessible and useable
    Unitary_table = CSV.File("Unitary_solution.txt",
				types=[Complex{Float64}, Complex{Float64}])
				 |> DataFrame

    # Define the number of values in the actionspace, the +1 makes allowance for purely horizontal movement
    num_actions = Int((round(maximum(real(Unitary_table.pee[:])), digits=1) - round(minimum(real(Unitary_table.pee[:])),
			 digits = 1))/sidelength_gridsquare + 1)
    
end
