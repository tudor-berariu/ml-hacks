module convnet

function init_weights(config::Array{Dict{String, Any}, 1}, T)
    # Dictionary of weights
    weights = Dict{Int, Array{T}}();
    # Size of inputs and number of maps from previous layer
    prev_width = config[1]["width"];
    prev_maps = 1;
    # Init weights
    for l in 1:length(config)-1
        layer = config[l+1];
        if layer["type"] == "CONV"
            weights[l] = rand(T, layer["maps"],
                              prev_maps,
                              layer["r"] * layer["r"]+1) - 0.5;
            prev_maps = layer["maps"];
            prev_width = prev_width - layer["r"] + 1;
        elseif layer["type"] == "MAXPOOL"
            prev_width = prev_width / layer["r"];
        elseif layer["type"] == "FULL"
            weights[l] = rand(layer["m"],
                              convert(Int64,prev_maps*prev_width*prev_width+1));
            prev_maps = layer["m"];
            prev_width = 1;
        end
    end
    return weights
end

function prepare_network(config::Array{Dict{String, Any}, 1}, batchSize, T)
    A = Dict{Int, Array{T}}();
    Z = Dict{Int, Array{T}}();
    E = Dict{Int, Array{T}}();
    # Size of inputs and number of maps from previous layer
    prev_width = config[1]["width"];
    prev_maps = 1;
    # Init weights
    for l in 1:length(config)-1
        layer = config[l+1];
        if layer["type"] == "CONV"
            A[l] = zeros(T, batchSize,
                         layer["maps"],
                         (prev_width - layer["r"] + 1),
                         (prev_width - layer["r"] + 1));
            Z[l] = copy(A[l]);
            E[l] = copy(A[l]);
            prev_maps = layer["maps"];
            prev_width = prev_width - layer["r"] + 1;
        elseif layer["type"] == "MAXPOOL"
            prev_width = convert(Int64, prev_width / layer["r"]);
            Z[l] = zeros(T, batchSize, prev_maps, prev_width, prev_width);
            E[l] = zeros(T, batchSize, prev_maps, prev_width, prev_width);
        elseif layer["type"] == "FULL"
            A[l] = zeros(T, batchSize, layer["m"]);
            Z[l] = zeros(T, batchSize, layer["m"]);
            E[l] = zeros(T, batchSize, layer["m"]);
            prev_maps = layer["m"];
            prev_width = 1;
        end
    end
    return A, Z, E
end

function describe_weights{T}(weights::Dict{Int, Array{T}})
    for i in weights
        println("Layer #", i[1], " has size ", size(i[2]));
    end
end

function learn{T}(config::Array{Dict{String, Any}, 1},
                    A::Dict{Int, Array{T}},
                    Z::Dict{Int, Array{T}},
                    E::Dict{Int, Array{T}},
                    weights::Dict{Int, Array{T}},
                    x::Array{T,2})
    for l = 1:length(config)-1
        println("Computng layer ", l);
    end
end

layers = Dict{String, Any}[{"type" => "INPUTS", "width" => 28},
                           {"type" => "CONV", "r" => 5, "maps" => 6},
                           {"type" => "MAXPOOL", "r" => 2},
                           {"type" => "CONV", "r" => 5, "maps" => 13},
                           {"type" => "MAXPOOL", "r" => 2},
                           {"type" => "FULL", "m" => 400},
                           {"type" => "FULL", "m" => 10}
                           ];


function learn()
    W = init_weights(layers, Float32);
    A, Z, E = prepare_network(layers, 100, Float32);
    learn(layers, W, rand(Float32, layers[1]["width"], layers[1]["width"]));
end

end
