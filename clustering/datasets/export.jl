#!/usr/local/bin/julia

using Gadfly

using Bullseye

function savePlot{T}(data::Array{Float64,2}, labels::Array{T, 1},
                     filename::String)
    draw(PDF("$(filename).pdf", 6inch, 6inch),
         plot(x = data[1,:], y = data[2,:], color = labels));
end


data, labels = bullseye(3000, 2, [0.1, 0.4, 0.7, 1.0], [0.1, 0.1, 0.1, 0.1]);
savePlot(data, labels, "bullseye");
