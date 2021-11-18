### This Code does ###

# This code defines a 2 dimensional reward grid with sleeves on the outside which have a very high negativ reward, an area shaped like the unitary solution with a low negativ reward and a horizontal area/line with the best given reward that represents the steady state solution. NOTE THAT THE PUNISHMENT (high negativ reward) is the dominant part of the program, since it decides where the agent is allowed to go. This means that one has to take care in choosing Variables.sleeve_radius, since it is otherwise possible for the steady state solution to cross the punishment area. 

### Preliminaries ###

# Load as module
module Reward

    using DelimitedFiles

    include("Variables.jl") # == Variables

    ### Program ###

    # Convert a given state into the corresponding row with state = Variabels.upper_bound_sleeve <=> row = 1 and state = Variables.lower_bound_sleeve <=> row = Variables.env_rows  
    # This is similar to unitary_rows in UnitaryQTable.jl
    # Note that the 63 is hardcoded for a reason and the program only works for gridsquare sizes of 0.1 and 0.05
    function state_to_row_converter(state)
        # The prefactor makes allowance for the different sizes of the gridsquares
        x̂ = 1 + Variables.prefactor(Variables.sidelength_gridsquare) * ((Variables.unitary_sol_range ÷ 63) * Variables.upper_bound_sleeve)
        row = x̂ - Variables.prefactor(Variables.sidelength_gridsquare) * ((Variables.unitary_sol_range ÷ 63) * state) |> Int
        if row <= 0
            return "Error: State out of considered range"
        elseif row > Variables.env_rows
            return "Error: State out of considered range"
        end
        return row
    end


    function custom_ceil(x, base)
        if base == 0.1
            return ceil(x, digits=-1)
        elseif base == 0.05
            return (ceil(2 * x,digits=-1) / 10) * 5 |> Int
        end
    end

    # Convert a given timestep of Unitary_solution.txt into the corresponding column with time = 1 <=> column = 1 and time = Varibales.unitary_sol_range <=> column = env_cols
    function time_to_col_converter(time)
        
        # The prefactor makes allowance for different sizes of the gridsquares
        # First check if the considered time lies in the first column
        # The -1 avoids jumps into the next ceil for the values in question, while the +1 yields the correct denominator for the modulo operation
        # Variables.sidelength_gridsquare = 0.1 needs a different, more specific treatment in comparison to Variables.sidelength_grisquare = 0.05, since for Variables.sidelength_gridsquare = 0.1 after consideration of the fixed stepsize of the Runge-Kutta 4ᵗʰ order of 0.01 it occurs that some time values will lie in the middle of two gridsquare, e.g. all multiples of 0.05. These values thus need to be considered for the gridsaure to the left AND to the right.
        # For Variables.sidelength_gridsquare = 0.05 this problem does not occur, since the lines between gridsquares lie at multiples of 0.025 and the nearest values with the 0.01 stepping are 0.02 and 0.03 and thus lie solely in their respective gridsquare     
        if time % (custom_ceil(time - 1, Variables.sidelength_gridsquare ) + 1) == 0 && Variables.sidelength_gridsquare == 0.1 && ceil(time, digits = -1)
		 == 10
           col = ceil((Variables.prefactor(Variables.sidelength_gridsquare) * time), digits=-1) 
		/ (Variables.unitary_sol_range ÷ 63) |> Int
            return [col]
        
        # Check if the considered time is a border value, that is a value that needs to lie in two neighbouring columns, but by definition is only able to lie in one
        # If the time considered is a border value, the respective column as well as the one immediately before it needs to be considered
        elseif time % (custom_ceil(time - 1, Variables.sidelength_gridsquare) + 1) == 0 && Variables.sidelength_gridsquare == 0.1
            col = ceil((Variables.prefactor(Variables.sidelength_gridsquare) * time), digits=-1) 
		/ (Variables.unitary_sol_range ÷ 63) |> Int
            return [col-1,col]
        
        # Assign columns to all other times that do not lie in column 1 and are no border values 
        else    
            col = ceil((Variables.prefactor(Variables.sidelength_gridsquare) * time), digits=-1) 
		/ (Variables.unitary_sol_range ÷ 63) |> Int
            return [col]
        end
        
        if col <= 0
            return "Error: Time out of considered range"
        elseif col > Variables.env_cols
            return "Error: Time out of considered range"
        end
    end

    # Define how the steady state is implemented, e.g. where does it begin, where does it end, what value is assigned to it, does it fade or not
    # steady_jump is inclusive, that is steady_jump = 5 means that the steady state will be inserted beginning at col 6
    function steady_state_fading(default_reward, final_steady_reward, steady_jump, steady_stepsize)
        steady_state_reward = default_reward
        steady_state_vector = []
        if steady_jump == 0
            for count in 1:Variables.env_cols
                append!(steady_state_vector, round(min(steady_state_reward, final_steady_reward),
		 digits=5))
                steady_state_reward += steady_stepsize
            end
        else
            for id in 1:Variables.env_cols
                if id < steady_jump
                    append!(steady_state_vector, default_reward)
                elseif id >= steady_jump
                    append!(steady_state_vector, round(min(steady_state_reward, final_steady_reward), digits=5))
                    steady_state_reward += steady_stepsize
                end
            end
        end
        return steady_state_vector
    end        

    # --------------------------------------------------------------------------
    # Initialize the basis for the rewardfield and load the steady state solution
    rewards = zeros(Float64, Variables.env_rows, Variables.env_cols)

    rewards[:,:] .= Variables.default_reward

    # Decide how the steady state is introduced, first case is a constant steady state starting at a given position, second case is a fading steady state jumping n values before starting to increase towards a final steady state reward

    # Case 1
    rewards[(Variables.steady_state |> state_to_row_converter),100:end] .= Variables.final_reward
    
    # Case 2
   # rewards[(Variables.steady_state) |> state_to_row_converter, :] = steady_state_fading(Variables.default_reward, Variables.final_steady_reward, Variables.steady_jump, Variables.steady_stepsize)
    # --------------------------------------------------------------------------

    # Erasing the first n ∈ {3,5} values out of the DataFrame makes allowance for the initial value, which does not need a reward, since it will always be initialized instead of being choosen by the agent.
    # Although five/three values get erased, there will still be Variables.env_cols time steps. One could think that this poses a hindrance, but the values that will be rounded into time step 63/126 by time_to_col_converter will eventually be the terminal states and thus their rewards can be omitted in considerations just as the ones of the initial value.
   Unitary_table_time_modification = Variables.Unitary_table 
	|> x -> delete!(x,
	 if Variables.sidelength_gridsquare == 0.1
        1:5
    else
        1:3
    end)

    # Define a customized round function for the gridsize of Variables.sidelength_gridsquare = 0.05
    function custom_round(x)
        return 0.05 * round(x/0.05)
    end

    # Define a custom truncate function to make the following code easier to write
    function custom_trunc(x, gridsize)
        if gridsize == 0.1
            return trunc(x, digits = 1)
        elseif gridsize == 0.05
            return trunc(custom_round(x), digits = 2)
        end
    end

    # Load the sleeve into the rewardfield, this leaves an area with the initial small negativ reward that is roughly shaped like the unitary solution out of Unitary_solution.jl
    # The code makes use of the time_to_col_converter to keep track of the grid boundaries along the x-axis
    # To keep track of the grid boundaries along the y-axis, each gridpoints' y-range is considered upon the value of Unitary_table_time_modification.pjj, with jj ∈ {ee, eg, ge, gg}
    for id in 1:length(Unitary_table_time_modification.pee)
        if (real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius) > custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) + Variables.sidelength_gridsquare / 2
            ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter 
            for ic in (Variables.upper_bound_sleeve |> state_to_row_converter):(ix - 1)
                rewards[ic,(id |> time_to_col_converter)] .= -100
            end
        elseif (real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius) < custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) - Variables.sidelength_gridsquare / 2
            ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
            for ic in (Variables.upper_bound_sleeve |> state_to_row_converter):(ix + 1)
                rewards[ic,(id |> time_to_col_converter)] .= -100
            end
        else
            ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
            for ic in (Variables.upper_bound_sleeve |> state_to_row_converter):ix
                rewards[ic,(id |> time_to_col_converter)] .= -100
            end
        end
        if (real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius) > custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) + Variables.sidelength_gridsquare / 2
            ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
            for ik in (ix - 1):(Variables.lower_bound_sleeve |> state_to_row_converter)
                rewards[ik,(id |> time_to_col_converter)] .= -100
            end
        elseif (real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius) < custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) - Variables.sidelength_gridsquare / 2
            ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
            for ik in (ix + 1):(Variables.lower_bound_sleeve |> state_to_row_converter)
                rewards[ik,(id |> time_to_col_converter)] .= -100
            end
        else
            ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
            for ik in ix:(Variables.lower_bound_sleeve |> state_to_row_converter)
                rewards[ik,(id |> time_to_col_converter)] .= -100
            end
        end
    end

    # Add a new column full of zeros in the beginning, so that the time-positions of the gridworld and the rewardfield match again, after the first timestep has been erased above
    rewards = [0 .* collect(1:Variables.env_rows) rewards]
    
    # Delete the last column to have identical sizes for the Gridworld/Q table and the Rewardfield
    rewards = rewards[1:end, 1:end-1]

    # This function is essentially time_to_col_converter(time), but note the slight difference in the returns, whereas a return in time_to_col_converter(time) for example reads return [col], here it reads return col
    # Omitting the square brackets yields a better handling of the data for saving it in files from which one is subsequently able to plot the data compared with to the returns with the square brackets
    # This difference is needed since time_to_col_converter(time) creates computer code from which the agent learns ("machine code"), so [a, [b,c]] = x actually assigns x to row a in columns b and c, while here in time_to_col_converter_plotable(time) it is necessary to save the data in a format that yields numbers inside the file in which it will be stored, so one will be able to later on extract and thus plot the data easily
    function time_to_col_converter_plotable(time)
        
        # The prefactor makes allowance for different sizes of the gridsquares
        # First check if the considered time lies in the first column
        # The -1 avoids jumps into the next ceil for the values in question, while the +1 yields the correct denominator for the modulo operation
        if time % (custom_ceil(time - 1, Variables.sidelength_gridsquare ) + 1) == 0 && Variables.sidelength_gridsquare == 0.1 
		&& ceil(time, digits = -1) == 10
            col = ceil((Variables.prefactor(Variables.sidelength_gridsquare) * time), digits=-1) / (Variables.unitary_sol_range ÷ 63) |> Int
            return col
        
        # Check if the considered time is a border value, that is a value that needs to lie in two neighbouring columns, but by definition is only able to lie in one
        # If the time considered is a border value, the respective column as well as the one immediately before it needs to be considered
        elseif time % (custom_ceil(time - 1, Variables.sidelength_gridsquare) + 1) == 0 && Variables.sidelength_gridsquare == 0.1    
            col = ceil((Variables.prefactor(Variables.sidelength_gridsquare) * time), digits=-1) / (Variables.unitary_sol_range ÷ 63) |> Int
            return col-1,col
        
        # Assign columns to all other times that do not lie in column 1 and are no border values 
        else    
            col = ceil((Variables.prefactor(Variables.sidelength_gridsquare) * time), digits=-1) / (Variables.unitary_sol_range ÷ 63) |> Int
            return col
        end
        
        if col <= 0
            return "Error: Time out of considered range"
        elseif col > Variables.env_cols
            return "Error: Time out of considered range"
        end
    end

    # Add the possibility of plotting not only a continuous but also a discrete sleeve
    # The .+ 1 adds the shift that is done in line 177 for the machine read code
    # The following functions need to be called explicitly inside the terminal after including the module, otherwise the data will not be available!
    function discrete_sleeves_top(file_name)
        discrete_grid_values = []
        for id in 1:length(Unitary_table_time_modification.pee)
            if (real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius) > custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) + Variables.sidelength_gridsquare / 2
                ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter 
                for ic in (Variables.upper_bound_sleeve |> state_to_row_converter):(ix - 1)
                    if length(time_to_col_converter_plotable(id)) == 2
                        holder = collect(time_to_col_converter_plotable(id))
                        append!(discrete_grid_values, [[ic,holder[1] .+ 1]])
                        append!(discrete_grid_values, [[ic,holder[2] .+ 1]])
                    else
                        append!(discrete_grid_values, [[ic,(id |> time_to_col_converter_plotable) .+ 1]])
                    end
                end
            elseif (real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius) < custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) - Variables.sidelength_gridsquare / 2
                ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
                for ic in (Variables.upper_bound_sleeve |> state_to_row_converter):(ix + 1)
                    if length(time_to_col_converter_plotable(id)) == 2
                        holder = collect(time_to_col_converter_plotable(id))
                        append!(discrete_grid_values, [[ic,holder[1] .+ 1]])
                        append!(discrete_grid_values, [[ic,holder[2] .+ 1]])
                    else
                        append!(discrete_grid_values, [[ic,(id |> time_to_col_converter_plotable) .+ 1]])
                    end
                end
            else
                ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) + Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
                for ic in (Variables.upper_bound_sleeve |> state_to_row_converter):ix
                    if length(time_to_col_converter_plotable(id)) == 2
                        holder = collect(time_to_col_converter_plotable(id))
                        append!(discrete_grid_values, [[ic,holder[1] .+ 1]])
                        append!(discrete_grid_values, [[ic,holder[2] .+ 1]])
                    else
                        append!(discrete_grid_values, [[ic,(id |> time_to_col_converter_plotable) .+ 1]])
                    end
                end
            end
        end
        open(file_name, "w") do io
            writedlm(io, ["row" "col"])
            writedlm(io, discrete_grid_values)
        end
    end

    function discrete_sleeves_bottom(file_name)
        discrete_grid_values = []
        for id in 1:length(Unitary_table_time_modification.pee)
            if (real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius) > custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables. sidelength_gridsquare) + Variables.sidelength_gridsquare / 2
                ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
                for ik in (ix - 1):(Variables.lower_bound_sleeve |> state_to_row_converter)
                    if length(time_to_col_converter_plotable(id)) == 2
                        holder = collect(time_to_col_converter_plotable(id))
                        append!(discrete_grid_values, [[ik,holder[1] .+ 1]])
                        append!(discrete_grid_values, [[ik,holder[2] .+ 1]])
                    else
                        append!(discrete_grid_values, [[ik,(id |> time_to_col_converter_plotable) .+ 1]])
                    end
                end
            elseif (real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius) < custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables. sidelength_gridsquare) - Variables.sidelength_gridsquare / 2
                ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
                for ik in (ix + 1):(Variables.lower_bound_sleeve |> state_to_row_converter)
                    if length(time_to_col_converter_plotable(id)) == 2
                        holder = collect(time_to_col_converter_plotable(id))
                        append!(discrete_grid_values, [[ik,holder[1] .+ 1]])
                        append!(discrete_grid_values, [[ik,holder[2] .+ 1]])
                    else
                        append!(discrete_grid_values, [[ik,(id |> time_to_col_converter_plotable) .+ 1]])
                    end
                end
            else
                ix = custom_trunc(real(Unitary_table_time_modification.pee[id]) - Variables.sleeve_radius, Variables.sidelength_gridsquare) |> state_to_row_converter
                for ik in ix:(Variables.lower_bound_sleeve |> state_to_row_converter)
                    if length(time_to_col_converter_plotable(id)) == 2
                        holder = collect(time_to_col_converter_plotable(id))
                        append!(discrete_grid_values, [[ik,holder[1] .+ 1]])
                        append!(discrete_grid_values, [[ik,holder[2] .+ 1]])
                    else
                        append!(discrete_grid_values, [[ik,(id |> time_to_col_converter_plotable) .+ 1]])
                    end
                end
            end
        end
        open(file_name, "w") do io
            writedlm(io, ["row" "col"])
            writedlm(io, discrete_grid_values)
        end
    end

end
