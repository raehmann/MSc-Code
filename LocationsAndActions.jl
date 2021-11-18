### This Code does ###

# This file provides all the necessary functions to move around the Rewardfield and Q table with respect to the defined boundaries

module MovementFunctions

    using DelimitedFiles

    include("Variables.jl")#  == Variables.
    include("UnitaryQTable.jl") # == QTable.
    include("Rewardfield.jl") # == Reward.

    # Location refers to a state-time pair

    ### Program ###

    # Check if the location is terminal, that is, is it inside the sleeve, exceeding the time or grid limit or good to go
    function is_location_terminal(current_row, current_col)
        nrows, ncols = size(Reward.rewards)
        inbounds = 1 <= current_row <= nrows && 1 <= current_col < ncols
        valid = Reward.rewards[current_row, current_col] != -100
        return !inbounds || !valid
    end

    # Save the row and column of the initial location for later use
    function initial_location()
    
        # -------------------------------------
        # Hardcode
	#initial_row = 18 
        #initial_col = 1
        
        # For initial location that is equivalent to (0,1)
        initial_row = Int(Variables.env_rows - round(Variables.sleeve_radius / Variables.sidelength_gridsquare))
        initial_col = 1
        # -------------------------------------
    
        return initial_row, initial_col
    end

    # Define the possible actions the agent can choose from in each time step
    # Use of Symbols makes the comprehension faster
    actions = [Symbol("up_$id") for id in reverse(1:(Variables.num_actions ÷ 2))]
    push!(actions, Symbol("even"))
    append!(actions, [Symbol("down_$id") for id in 1:(Variables.num_actions ÷ 2)])

    # Depending on the current location and with respect to an ε-greedy algorithm, decide on which action to take
    function next_action(current_row, current_col, ε)

        # If the random number is smaller than ε of ε-greedy, explore otherwise exploit the existing knowledge <=> probability of exploration or exploitation, as long as 0 ≤ ε ≤ 1
        if rand(Float64) < ε
            return rand(1:Variables.num_actions)
        
        # Make a distinction between possible multiple maximum action arguments, since argmax only ever chooses the argument of the first maximum to occur
        # This adds an additional variation layer, so that the agent is able to "exploit more freely" in the sense that for multi-maximum values and thus hypothetical multi-argmax values a sense of variation is added instead of choosing the argument of the first maximum value to occur each time
        elseif length(findall(x -> x == maximum(QTable.Q_values[current_row, current_col, :]), QTable.Q_values[current_row, current_col, :])) > 1
            
            # Find the indices of the maximum values in the 3D Q_values array 
            maximum_positions = findall(x -> x == maximum(QTable.Q_values[current_row, current_col, :]), QTable.Q_values[current_row, current_col, :])
            
            # Select one of the indices of the array that represents the indices of the maximum values in the 3D Q_values array
            choosen_maximum = rand(1:length(findall(x -> x == maximum(QTable.Q_values[current_row, current_col, :]), QTable.Q_values[current_row, current_col, :])))
            return maximum_positions[choosen_maximum] 
        
        # If only one maximum value exists, take its argument
        else 
            return argmax(QTable.Q_values[current_row, current_col, :])
        end
    end

    # Convert a given index of the actions array into the corresponding movement along the rows of the Q table
    # Take care of the signs! Up has positive, down has negativ signs, this needs to be changed when applying the results
    function action_index_to_row_movement_converter(action_index)
        x̂ = 1 + (Variables.num_actions ÷ 2)
        movement = x̂ - action_index * Variables.prefactor(Variables.sidelength_gridsquare) * Variables.sidelength_gridsquare * (Variables.unitary_sol_range ÷ 63) |> x -> round(Int64, x)
    end

    # Check if a choosen action will cause a movement that leads the agent outside the gridworld borders given by Variables.env_rows and Variables.env_cols
    function is_movement_valid(next_row, next_col)
        return 1 <= next_row <= Variables.env_rows && 1 <= next_col <= Variables.env_cols
    end

    # Depending on the current row, column and action, calculate the new position inside the Q table
    function next_location(current_row, current_col, current_action)
        current_action = current_action
        
        # Introduce a variable that checks if the next_location() is inside the bounds set by Variables.env_rows and Variables.env_cols, although this is already implemented in is_location_terminal() it seems not to be sufficient, that is why it is added again here 
        bound_checker = false
        while bound_checker == false
            next_row = current_row
            next_col = current_col
            
            # Keep in mind that the index positions do not match the movement index, e.g. up_5 has index 1, this is taken care of by the second part of the Symbol construction
            if actions[current_action] == Symbol("up_", ((Variables.num_actions ÷ 2) + 1) - current_action)
                next_col += 1
                
                # The aforementioned, unintuitive sign inversion
                next_row -= action_index_to_row_movement_converter(current_action) |> Int
            
                # Keep in mind that the index positions do not match the movement index, e.g. down_5 has index 11, this is taken care of by the second part of the Symbol construction
            elseif actions[current_action] == Symbol("down_", current_action - ((Variables.num_actions ÷ 2) + 1)) 
                next_col += 1
                
                # The aforementioned, unintuitive sign inversion
                next_row -= action_index_to_row_movement_converter(current_action) |> Int
            elseif actions[current_action] == :even
                next_col += 1
            end
            
            # Update bound_checker and, depending on the outcome, choose a new action or approve of the next location
            bound_checker = is_movement_valid(next_row, next_col)
            if bound_checker == false
                current_action = next_action(current_row, current_col, Variables.ε)
            else
                return next_row, next_col
            end
        end
    end

    # Search for the path that the agent goes if only the best actions are selected and write it in a file
    # To select only the best actions, ε is choosen to be 0.0, that is there is no probability for exploration, only exploitation takes place
    function collect_pathpoints(initial_location, file_name)
        current_row, current_col = initial_location
        if is_location_terminal(current_row, current_col) == true
            return "Error: Initial value is wrong."
        else
            path = []
            append!(path, [[current_row, current_col]])

            while !is_location_terminal(current_row, current_col) 
                current_action = next_action(current_row, current_col, 0.0)
                current_row, current_col = next_location(current_row, current_col, current_action)
                append!(path, [[current_row, current_col]])
            end
            open(file_name, "w") do io
                writedlm(io, ["row" "col"])
                writedlm(io, path)
            end
        end
    end

    # -----------------------------------------------
    # Plot the unitary solution
     # This function needs to be called explicitly after the module was loaded into the terminal, otherwise the data will not be available!
    #collect_pathpoints(initial_location(), ARGS[1])
    # -----------------------------------------------

    # Make the possible locations the agent can occupy accessible, so a markertest plot can be created
    # This function needs to be called explicitly after the module was loaded into the terminal, otherwise the data will not be available!
    function markertest(file_name)
        markers_at = []
        for id in 1:Variables.env_rows
            for ix in 2:Variables.env_cols
                if !is_location_terminal(id, ix) #&& is_movement_valid(id, ix)
                    append!(markers_at, [[id, ix]])
                end
            end
        end
        open(file_name, "w") do io
            writedlm(io, ["row" "col"])
            writedlm(io, markers_at)
        end
    end
    
end
