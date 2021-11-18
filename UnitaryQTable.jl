### This Code does ###

# This code loads the numerical solution of the unitary part calculated with Unitary_solution.jl into a 3 dimensional array that represents the Q table of the Reinforcement Learning agent. While doing so it awards one state (2d vertical axis) at each timestep (2d horizontal axis) an "initial accumulative reward" which decays with an increase in timesteps, until it only awards zeros, that is it does not change the initial Q table values. Which array location gets awarded this "initial accumulative reward" depends on the size of the timesteps in Unitary_solution.jl, the size of the timesteps in the 3d array (because the timesteps in the solution files must be found that equal available array timesteps), the size of the statesteps in the 3d array, that is the range in which a numerical solution is appointed to a specific state and on the number of actions. At each (state,time) pair, which from now on is called location, the action that yields the next correct pair according to the numerical solution is awarded the "reward". For example suppose that a3 shifts down 3 states. So if we are in (s2,t7) (see below) and the numerical solution tells us that (s5,t8), then the "initial accumulative reward" will be saved in the array location [s2,t7,a3] which equals "For being in state s2 at time t7 the highest reward can be obtained by taking action a3.". 

# For better understanding, see below for an example of a 3d matrix as described above. It has 11 states/amplitudes, 11 timesteps and 6 possible actions. Each of the 11*11*6 = 726 Array locations will later (that is during training) will be awarded a reward value, such that for each time and each state, six values are competing against one another to tell the agent that their action is the best to take (in terms of the less negativ the value is the higher the reward and thus the better the action). 
#
#                        t1  t2  t3  t4  t5  t6  t7  t8  t9  t10  t11  a6 |
#                    t1  t2  t3  t4  t5  t6  t7  t8  t9  t10  t11 a5  |   |
#                t1  t2  t3  t4  t5  t6  t7  t8  t9  t10  t11  a4 |   |   |
#            t1  t2  t3  t4  t5  t6  t7  t8  t9  t10  t11  a3 |   |   |   |
#        t1  t2  t3  t4  t5  t6  t7  t8  t9  t10  t11  a2 |   |   |   |   |
#    t1  t2  t3  t4  t5  t6  t7  t8  t9  t10  t11  a1 |   |   |   |   |   |
#s1 |__||__||__||__||__||__||__||__||__||___||___|    |   |   |   |   |   | 
#s2 |__||__||__||__||__||__||__||__||__||___||___|    |   |   |   |   |   |
#s3 |__||__||__||__||__||__||__||__||__||___||___|    |   |   |   |   |   |
#s4 |__||__||__||__||__||__||__||__||__||___||___|    |   |   |   |   |   |
#s5 |__||__||__||__||__||__||__||__||__||___||___|    |   |   |   |   |   |
#s6 |__||__||__||__||__||__||__||__||__||___||___|    |   |   |   |   | # 
#s7 |__||__||__||__||__||__||__||__||__||___||___|    |   |   |   | #     
#s8 |__||__||__||__||__||__||__||__||__||___||___|    |   |   | #  
#s9 |__||__||__||__||__||__||__||__||__||___||___|    |   | #      
#s10|__||__||__||__||__||__||__||__||__||___||___|    | #       
#s11|__||__||__||__||__||__||__||__||__||___||___|  #       

# Note the following important difference:
# row refers to the rows of the Q table (sis above) and column refers to the columns of the Q table (tjs above)
# On the other hand, state refers to the values stored in Unitary_solution.txt[:,2] (output of Unitary_solution.jl) that is to the values that are taken as the amplitudes, while time refers to Unitary_solution.txt[1,:], that is the total number of cols
# So row and column are integer encodings for the actual values inside Unitary_solution.txt
# Also location refers to specific pairs (state,time)/(row,column)

### Preliminaries ###

# Load as module 
module QTable

    using DelimitedFiles

    include("Variables.jl") # == Variables.

    ### Program ###

    # Create the Q table with initial values 0.0 in each array location
    Q_values = zeros(Float64, Variables.env_rows, Variables.env_cols, Variables.num_actions)

    # Define the size of the steps between each value in Unitary_solution.txt that is of interest for Variables.sidelength_gridsquare = 0.1
    # This value is also important for Variables.sidelength_gridsquare 0.05, because the program was first fitted for Variables.sidelength_gridsquare = 0.1 and thus in some cases it still depends on its value as a sclaing basis for Variables.sidelength_gridsquare = 0.05 
    unitary_sol_stepsize = Variables.unitary_sol_range ÷ 63

    # Find the right rows for the values with Variables.unitary_sol_range ÷ Variables.env_cols difference in Unitary_solution.txt, that is order the states to the respective rows
    # This is done by assigning the first value in the Q table the highest possible state, e.g. this value occupies the first row ([1,t,a]) -> this yields that the lower bound sleeve value occupies the last row
    unitary_rows = let
        points = collect(1:(Variables.unitary_sol_range ÷ Variables.env_cols):length(Variables.Unitary_table.pee))
        # Calculate the number of the row that represents the state zero
        x̂ = 1 + Variables.prefactor(Variables.sidelength_gridsquare) * (unitary_sol_stepsize * Variables.upper_bound_sleeve)
        r = real.(Variables.Unitary_table.pee[points]) 
			|> x -> round.(x, digits=1)
        map(r) do x
            Int(x̂ - Variables.prefactor(Variables.sidelength_gridsquare) * (unitary_sol_stepsize * x))
        end 
    end

    # The unitary_distribution array gives the actions that will distribute the unitary solutions from Unitary_solution.txt across the initialized, all zeros Q table
    # 1 equals the maximum value of upward movement from the current position, which is given by round(maximum(real(Unitary_table.pee[:])), digits=1)/2 and the entry at the row equal to num_actions equals the maximum downward movement from the current position, so -round(maximum(real(Unitary_table.pee[:])), digits=1)/2 in this case. The +1 is added to offset x̂ to the value corresponding to zero vertical movement
    # Terminal states do not need to be considered here since the reward field (and terminal states) is not yet defined
    unitary_distribution = let 
        x̂ = Variables.num_actions ÷ 2 + 1
        diffs = diff(unitary_rows)
        push!(diffs, 0)
        map(x -> (x̂ + x), diffs)
    end

    # Write initial accumulative rewards into the Q table positions of the unitary solution with the right subsequent action and decay the initial accumulative reward up to zero
    let
        indices = mapslices(row -> CartesianIndex(row...), [unitary_rows 1:Variables.env_cols unitary_distribution], dims=[2])
        Q_values[indices] = max.(0, Variables.init_Q_value .-  Variables.init_Q_value_decay 
				.* collect(0:(Variables.env_cols-1))) 
    end

    # The possibility to access the initial Q-value distribution is added, which enables, in addition to plotting the unitary function, plotting a discretized version of the unitary solution in the form of the Q_values that are initially choosen from the above code to be ≠ 0 
    function initial_Q_plotable(file_name)
        unitary_rows_cp = unitary_rows[1:end-1]
        open(file_name, "w") do io
            writedlm(io, ["row" "col"])
            writedlm(io, [unitary_rows_cp 1:Variables.env_cols-1])
        end
    end

end
