### This Code does ###

# This code calculates the unitary solution of a damped two level system with spontaneous emission via 4th-order Runge-Kutta and writes it in a file so that the numerical solution is available.

### Premliminaries ###

using DifferentialEquations
using DelimitedFiles

### Program ###

# ----------------------------------
# Define the Rabi-frequency
Omega = 10
# ----------------------------------

# Baseline for nondetuned damped system. Counting 1 = pee; 2 = peg; 3 = pge; 4 = pgg.
# Be careful with the function variables: first derivative, then actual function, then parameters (if needed) then time. See documentation!
function Baseline(dp, p, params, t)
    Omega = params
    dp[1] = im * Omega/2 * (p[2] - p[3])
    dp[2] = im * Omega/2 * (p[1] - p[4])
    dp[3] = im * Omega/2 * (p[4] - p[1])
    dp[4] = im * Omega/2 * (p[3] - p[2])
end

# ------------------------------------------------------
p0 = [0.0+0.0*im;0.0+0.0*im;0.0+0.0*im;1.0+0.0*im]
tspan = (0.0,2*pi)
params = Omega
prob = ODEProblem(Baseline, p0, tspan, params)
runkut = solve(prob, RK4(), dt = 0.01, adaptive=false)
# --------------------------------------------------------

# Write the solution in a file
# The range is given by 2pi/0.01 + 2, such that the first step which eqauls zero as well as the subseqent step that does not fall into the range of the actual step size but yields from 6.28 to 2pi are included
open("Unitary_solution.txt", "w") do io
    writedlm(io, ["t"   "pee"])
    for id in 1:630
        writedlm(io, [runkut.t[id] runkut[1, id]])
    end
end
